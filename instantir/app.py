import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import gradio as gr
#import spaces
from PIL import Image

from diffusers import DDPMScheduler
from schedulers.lcm_single_step_scheduler import LCMSingleStepScheduler

from module.ip_adapter.utils import load_adapter_to_pipe
from pipelines.sdxl_instantir import InstantIRPipeline
import gc

print(f"version={torch.__version__}")


def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        # ratio = min_side / min(h, w)
        # w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image

from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="InstantX/InstantIR", filename="models/adapter.pt", local_dir=".")
hf_hub_download(repo_id="InstantX/InstantIR", filename="models/aggregator.pt", local_dir=".")
hf_hub_download(repo_id="InstantX/InstantIR", filename="models/previewer_lora_weights.bin", local_dir=".")

instantir_path = f'./models'

sdxl_repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
dinov2_repo_id = "facebook/dinov2-large"
lcm_repo_id = "latent-consistency/lcm-lora-sdxl"

if torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "mps"
    torch_dtype = torch.float32
else:
    device = "cpu"
    torch_dtype = torch.float32

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

PROMPT = "照片写实风格，高度细节，超高清细节，32k分辨率，\
超高清，极致细节表现，皮肤毛孔细节，\
超高清晰度，完美无变形，\
使用佳能EOS R相机拍摄，电影感，高对比度，专业色彩分级。"

NEG_PROMPT = "模糊，焦点不清晰，深度模糊，过度平滑，\
素描，油画，卡通，CG风格，3D渲染，虚幻引擎，\
脏，乱，最差质量，低质量，帧，绘画，插画，素描，艺术，\
水印，签名，JPEG伪影，变形，低分辨率"

def unpack_pipe_out(preview_row, index):
    return preview_row[index][0]

def dynamic_preview_slider(sampling_steps):
    print(sampling_steps)
    return gr.Slider(label="Restoration Previews", value=sampling_steps-1, minimum=0, maximum=sampling_steps-1, step=1)

def dynamic_guidance_slider(sampling_steps):
    return gr.Slider(label="Start Free Rendering", value=sampling_steps, minimum=0, maximum=sampling_steps, step=1)

def show_final_preview(preview_row):
    return preview_row[-1][0]

#@spaces.GPU(duration=70) #[uncomment to use ZeroGPU]
@torch.no_grad()
def instantir_restore(
    lq, prompt="", steps=30, cfg_scale=7.0, guidance_end=1.0,
    creative_restoration=False, seed=3407, height=1024, width=1024, preview_start=0.0, cpu_offload=False, progress=gr.Progress(track_tqdm=True)):



    # Load pretrained models.
    print("Initializing pipeline...")
    pipe = InstantIRPipeline.from_pretrained(
        sdxl_repo_id,
        torch_dtype=torch_dtype,
    )

    # Image prompt projector.
    print("Loading LQ-Adapter...")
    load_adapter_to_pipe(
        pipe,
        f"{instantir_path}/adapter.pt",
        dinov2_repo_id,
    )

    # Prepare previewer
    lora_alpha = pipe.prepare_previewers(instantir_path)
    print(f"use lora alpha {lora_alpha}")
    lora_alpha = pipe.prepare_previewers(lcm_repo_id, use_lcm=True)
    print(f"use lora alpha {lora_alpha}")
    pipe.to(device=device, dtype=torch_dtype)
    pipe.scheduler = DDPMScheduler.from_pretrained(sdxl_repo_id, subfolder="scheduler")
    lcm_scheduler = LCMSingleStepScheduler.from_config(pipe.scheduler.config)

    # Load weights.
    print("Loading checkpoint...")
    aggregator_state_dict = torch.load(
        f"{instantir_path}/aggregator.pt",
        map_location="cpu"
    )
    pipe.aggregator.load_state_dict(aggregator_state_dict, strict=True)
    pipe.aggregator.to(device=device, dtype=torch_dtype)

    print("******loaded")

    if creative_restoration:
        if "lcm" not in pipe.unet.active_adapters():
            pipe.unet.set_adapter('lcm')
    else:
        if "previewer" not in pipe.unet.active_adapters():
            pipe.unet.set_adapter('previewer')

    pipe.enable_vae_tiling()

#    if cpu_offload:
#        pipe.enable_model_cpu_offload()
#        #pipe.enable_sequential_cpu_offload()


    if isinstance(guidance_end, int):
        guidance_end = guidance_end / steps
    elif guidance_end > 1.0:
        guidance_end = guidance_end / steps
    if isinstance(preview_start, int):
        preview_start = preview_start / steps
    elif preview_start > 1.0:
        preview_start = preview_start / steps

    w, h = lq.size
    if w == h :
        lq = [resize_img(lq.convert("RGB"), size=(width, height))]
    else:
        lq = [resize_img(lq.convert("RGB"), size=None)]
   
    generator = torch.Generator(device=device).manual_seed(seed)
    timesteps = [
        i * (1000//steps) + pipe.scheduler.config.steps_offset for i in range(0, steps)
    ]
    timesteps = timesteps[::-1]

    prompt = PROMPT if len(prompt)==0 else prompt
    neg_prompt = NEG_PROMPT

    out = pipe(
        prompt=[prompt]*len(lq),
        image=lq,
        num_inference_steps=steps,
        generator=generator,
        timesteps=timesteps,
        negative_prompt=[neg_prompt]*len(lq),
        guidance_scale=cfg_scale,
        control_guidance_end=guidance_end,
        preview_start=preview_start,
        previewer_scheduler=lcm_scheduler,
        return_dict=False,
        save_preview_row=True,
    )
    for i, preview_img in enumerate(out[1]):
        preview_img.append(f"preview_{i}")

    del pipe
    gc.collect()
    print(f"TORCH={torch}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()
    return out[0][0], out[1]


examples = [
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "An astronaut riding a green horse",
    "A delicious ceviche cheesecake slice",
]

css="""
#col-container {
    margin: 0 auto;
    max-width: 640px;
}
"""

with gr.Blocks() as demo:
    with gr.Accordion("使用说明", open=False):
        gr.Markdown(
        """
        # InstantIR: 基于即时生成参考的盲图像修复

        ### **InstantIR 的官方 🤗 Gradio 演示**
        ### **InstantIR 不仅可以帮助您修复损坏的图像，还可以根据您的文本提示进行创意重建。查看高级用法获取更多详情！**
        
        ---

        ## 基本用法：修复您的图像
        1. 上传您想要修复的图像；
        2. 可选择调整 `步数` `CFG比例` 参数。通常步数越高效果越好，但建议不要超过50以保证效率；
        3. 点击 `InstantIR 魔法！`。

        ---

        ## 高级用法：
        ### 浏览修复变体：
        1. InstantIR 处理后，拖动 `修复预览` 滑块探索其他进行中的版本；
        2. 如果您喜欢其中某个版本，将 `开始自由渲染` 滑块设置为相同的值以获得更精细的结果。
        ### 创意修复：
        1. 勾选 `创意修复` 复选框；
        2. 在 `修复提示` 文本框中输入您的文本提示；
        3. 将 `开始自由渲染` 滑块设置为中等值（约为 `步数` 的一半）以为 InstantIR 创作提供足够空间。
        """)
    with gr.Row():
        with gr.Column():
            lq_img = gr.Image(label="待修复图像", type="pil")      
            
            with gr.Row():
                steps = gr.Number(label="步数", value=30, step=1)
                cfg_scale = gr.Number(label="CFG比例", value=7.0, step=0.1)
            
            with gr.Row():
                height = gr.Number(label="高度", value=1024, step=1, visible=False)
                width = gr.Number(label="宽度", value=1024, step=1, visible=False)
                seed = gr.Number(label="随机种子", value=42, step=1)
            # guidance_start = gr.Slider(label="Guidance Start", value=1.0, minimum=0.0, maximum=1.0, step=0.05)
            guidance_end = gr.Slider(label="开始自由渲染", value=30, minimum=0, maximum=30, step=1)
            preview_start = gr.Slider(label="预览开始", value=0, minimum=0, maximum=30, step=1)
            prompt = gr.Textbox(label="修复提示（可选）", placeholder="")
            mode = gr.Checkbox(label="创意修复", value=False)
            cpu_offload = gr.Checkbox(label="CPU卸载", info="如果您有大量GPU显存，取消选中此选项可加快生成速度", value=False, visible=False)

    
            with gr.Row():
                restore_btn = gr.Button("InstantIR 魔法！")
                clear_btn = gr.ClearButton()
            gr.Examples(
                    examples = ["../assets/lady.png", "../assets/man.png", "../assets/dog.png", "../assets/panda.png", "../assets/sculpture.png", "../assets/cottage.png", "../assets/Naruto.png", "../assets/Konan.png"],
                    inputs = [lq_img]
                )
        with gr.Column():
            output = gr.Image(label="InstantIR 修复结果", type="pil")
            index = gr.Slider(label="修复预览", value=29, minimum=0, maximum=29, step=1)
            preview = gr.Image(label="预览", type="pil")
       
    pipe_out = gr.Gallery(visible=False)
    clear_btn.add([lq_img, output, preview])
    restore_btn.click(
        instantir_restore, inputs=[
            lq_img, prompt, steps, cfg_scale, guidance_end,
            mode, seed, height, width, preview_start, cpu_offload
        ],
        outputs=[output, pipe_out], api_name="InstantIR"
    )
    steps.change(dynamic_guidance_slider, inputs=steps, outputs=guidance_end)
    output.change(dynamic_preview_slider, inputs=steps, outputs=index)
    index.release(unpack_pipe_out, inputs=[pipe_out, index], outputs=preview)
    output.change(show_final_preview, inputs=pipe_out, outputs=preview)

demo.queue().launch()
