import os
import sys
import gradio as gr
import pillow_heif
import torch
import devicetorch
import subprocess
import gc
import psutil  # for system stats - gpu/cpu etc
import random
import shutil

from PIL import Image
from typing import List
from pathlib import Path
from datetime import datetime
from gradio_imageslider import ImageSlider
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoProcessor

from refiners.fluxion.utils import manual_seed
from refiners.foundationals.latent_diffusion import Solver, solvers
from enhancer import ESRGANUpscaler, ESRGANUpscalerCheckpoints
from system_monitor import SystemMonitor
from message_manager import MessageManager

import warnings
# Filter out the timm deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
# Filter the GenerationMixin inheritance warning
warnings.filterwarnings("ignore", message=".*has generative capabilities.*")
# Filter the PyTorch flash attention warning
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

pillow_heif.register_heif_opener()
pillow_heif.register_avif_opener()

message_manager = MessageManager()

last_seed = None
save_path = "../outputs"   # Can be changed to a preferred directory: "C:\path\to\save_folder"
os.makedirs(save_path, exist_ok=True)
MAX_GALLERY_IMAGES = 30
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif', '.avif'}


CHECKPOINTS = ESRGANUpscalerCheckpoints(
    unet=Path(
        hf_hub_download(
            repo_id="refiners/juggernaut.reborn.sd1_5.unet",
            filename="model.safetensors",
            revision="347d14c3c782c4959cc4d1bb1e336d19f7dda4d2",
        )
    ),
    clip_text_encoder=Path(
        hf_hub_download(
            repo_id="refiners/juggernaut.reborn.sd1_5.text_encoder",
            filename="model.safetensors",
            revision="744ad6a5c0437ec02ad826df9f6ede102bb27481",
        )
    ),
    lda=Path(
        hf_hub_download(
            repo_id="refiners/juggernaut.reborn.sd1_5.autoencoder",
            filename="model.safetensors",
            revision="3c1aae3fc3e03e4a2b7e0fa42b62ebb64f1a4c19",
        )
    ),
    controlnet_tile=Path(
        hf_hub_download(
            repo_id="refiners/controlnet.sd1_5.tile",
            filename="model.safetensors",
            revision="48ced6ff8bfa873a8976fa467c3629a240643387",
        )
    ),
    esrgan=Path(
        hf_hub_download(
            repo_id="philz1337x/upscaler",
            filename="4x-UltraSharp.pth",
            revision="011deacac8270114eb7d2eeff4fe6fa9a837be70",
        )
    ),
    negative_embedding=Path(
        hf_hub_download(
            repo_id="philz1337x/embeddings",
            filename="JuggernautNegative-neg.pt",
            revision="203caa7e9cc2bc225031a4021f6ab1ded283454a",
        )
    ),
    negative_embedding_key="string_to_param.*",
    
    loras={
        "more_details": Path(
            hf_hub_download(
                repo_id="philz1337x/loras",
                filename="more_details.safetensors",
                revision="a3802c0280c0d00c2ab18d37454a8744c44e474e",
            )
        ),
        "sdxl_render": Path(
            hf_hub_download(
                repo_id="philz1337x/loras",
                filename="SDXLrender_v2.0.safetensors",
                revision="a3802c0280c0d00c2ab18d37454a8744c44e474e",
            )
        ),
    },
)

device = torch.device(devicetorch.get(torch))
dtype = devicetorch.dtype(torch, "bfloat16")
enhancer = ESRGANUpscaler(checkpoints=CHECKPOINTS, device=device, dtype=dtype)


def generate_prompt(image: Image.Image, caption_detail: str = "<CAPTION>") -> str:
    """
    使用 Florence-2 为图片生成详细描述。
    """
    if image is None:
        message_manager.add_warning("请先加载图片!")
        return gr.Warning("请先加载图片!")
        
    try:
        message_manager.add_message(f"开始使用 Florence-2 生成描述，详细程度: {caption_detail}")
        device = torch.device(devicetorch.get(torch))
        torch_dtype = devicetorch.dtype(torch, "bfloat16")
        
        gc.collect()
        devicetorch.empty_cache(torch)
        message_manager.add_message("正在加载 Florence-2 模型...")

        # Load model in eval mode immediately
        model = AutoModelForCausalLM.from_pretrained(
            "multimodalart/Florence-2-large-no-flash-attn", 
            torch_dtype=torch_dtype,
            trust_remote_code=True
        ).eval()
        
        processor = AutoProcessor.from_pretrained(
            "multimodalart/Florence-2-large-no-flash-attn",
            trust_remote_code=True
        )
        message_manager.add_success("Florence-2 模型加载成功")

        # Move model to device after eval mode
        model = devicetorch.to(torch, model)
        message_manager.add_message("正在使用 Florence-2 处理图片...")

        # Process the image
        inputs = processor(
            text=caption_detail, 
            images=image.convert("RGB"), 
            return_tensors="pt"
        )
        
        # Convert inputs to the correct dtype and move to device
        inputs = {
            k: v.to(device=device, dtype=torch_dtype if v.dtype == torch.float32 else v.dtype) 
            for k, v in inputs.items()
        }

        # Generate caption with no grad
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=2
            )
            
            # Move generated_ids to CPU immediately
            generated_ids = generated_ids.cpu()

        # Clear inputs from GPU
        inputs = {k: v.cpu() for k, v in inputs.items()}
        devicetorch.empty_cache(torch)
        
        # Process the generated text
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=caption_detail,
            image_size=(image.width, image.height)
        )
        
        # Clean up the caption and add enhancement-specific terms
        raw_caption = parsed_answer[caption_detail]
        caption_text = clean_caption(raw_caption)
        enhanced_prompt = f"masterpiece, best quality, highres, {caption_text}"
        
        message_manager.add_message("原始描述: " + raw_caption)
        message_manager.add_success(f"生成的提示词: {enhanced_prompt}")

        # Aggressive cleanup
        del generated_ids
        del inputs
        model.cpu()
        del model
        del processor
        gc.collect()
        devicetorch.empty_cache(torch)
        message_manager.add_message("清理 Florence-2 资源")
            
        return enhanced_prompt
        
    except Exception as e:
        # Ensure cleanup happens even on error
        gc.collect()
        devicetorch.empty_cache(torch)
        message_manager.add_error(f"生成描述时出错: {str(e)}")
        return gr.Warning(f"Error generating prompt: {str(e)}")
        
        
def clean_caption(text: str) -> str:
    """
    通过删除常见前缀、填充短语和悬空描述来清理描述文本。
    """
    # 要删除的常见前缀
    replacements = [
        "图片显示 ",
        "图片是 ",
        "图片描绘 ",
        "这张图片显示 ",
        "这张图片描绘 ",
        "照片显示 ",
        "照片描绘 ",
        "图显示 ",
        "图像描绘 ",
        "整体氛围 ",
        "图片的氛围 ",
        "有一个 ",
        "我们可以看到 ",
    ]
    
    cleaned_text = text
    for phrase in replacements:
        cleaned_text = cleaned_text.replace(phrase, "")
    
    # 删除情绪/氛围相关片段
    mood_patterns = [
        ". 氛围是 ",
        ". 气氛是 ",
        ". 图片的感觉是 ",
        ". 整体感觉是 ",
        ". 基调是 ",
    ]
    
    for pattern in mood_patterns:
        if pattern in cleaned_text:
            cleaned_text = cleaned_text.split(pattern)[0]
    
    # 删除尾部片段
    while cleaned_text.endswith((" 是", " 有", " 和", " 与", " 的")):
        cleaned_text = cleaned_text.rsplit(" ", 1)[0]
    
    return cleaned_text.strip()


def get_seed(seed_value: int, reuse: bool) -> int:
    """处理种子生成和重用逻辑。"""
    global last_seed
    
    if reuse and last_seed is not None:
        return last_seed
    
    if seed_value == -1:
        generated_seed = random.randint(0, 10_000)
        last_seed = generated_seed
        return generated_seed
    
    last_seed = seed_value
    return seed_value

    
def process(
    input_image: Image.Image,
    prompt: str = "",
    negative_prompt: str = "",
    seed: int = -1,
    reuse_seed: bool = False,
    upscale_factor: int = 2,
    controlnet_scale: float = 0.6,
    controlnet_decay: float = 1.0,
    condition_scale: int = 6,
    tile_width: int = 112,
    tile_height: int = 144,
    denoise_strength: float = 0.35,
    num_inference_steps: int = 18,
    solver: str = "DDIM",
    auto_save_enabled: bool = True,  
) -> tuple[Image.Image, Image.Image]:
    try:
        # 输入验证
        if input_image is None:
            message_manager.add_warning("请先加载图片!")
            return gr.Warning("请先加载图片!")
            
        actual_seed = get_seed(seed, reuse_seed)
        message_manager.add_message(f"开始增强，使用种子值 {actual_seed}")
        message_manager.add_message(f"放大倍数: {upscale_factor}x")
        
        # 处理前清理内存
        gc.collect()
        devicetorch.empty_cache(torch)
        
        manual_seed(actual_seed)
        solver_type: type[Solver] = getattr(solvers, solver)

        # 使用 no_grad 上下文
        with torch.no_grad():
            message_manager.add_message("正在处理图片...")
            enhanced_image = enhancer.upscale(
                image=input_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                upscale_factor=upscale_factor,
                controlnet_scale=controlnet_scale,
                controlnet_scale_decay=controlnet_decay,
                condition_scale=condition_scale,
                tile_size=(tile_height, tile_width),
                denoise_strength=denoise_strength,
                num_inference_steps=num_inference_steps,
                loras_scale={"more_details": 0.5, "sdxl_render": 1.0},
                solver_type=solver_type,
            )

        global latest_result
        latest_result = enhanced_image
        message_manager.add_success("增强完成!")
        
        if auto_save_enabled:
            save_output(enhanced_image, True)
        
        # 处理后清理内存
        gc.collect()
        devicetorch.empty_cache(torch)
        message_manager.add_message("已清理资源")
        
        return (input_image, enhanced_image)
        
    except Exception as e:
        message_manager.add_error(f"处理过程中出错: {str(e)}")
        gc.collect()
        devicetorch.empty_cache(torch)
        return gr.Warning(f"Error during processing: {str(e)}")

        
def batch_process_images(
    files,
    prompt: str = "",
    negative_prompt: str = "",
    seed: int = -1,
    reuse_seed: bool = False,
    upscale_factor: int = 2,
    controlnet_scale: float = 0.6,
    controlnet_decay: float = 1.0,
    condition_scale: int = 6,
    tile_width: int = 112,
    tile_height: int = 144,
    denoise_strength: float = 0.35,
    num_inference_steps: int = 18,
    solver: str = "DDIM",
    progress=gr.Progress()
) -> tuple[str, List[str], tuple[Image.Image, Image.Image]]:
    """
    使用增强器处理多个图片并直接保存到批处理子文件夹。
    """
    def generate_summary():
        """生成批处理摘要的辅助函数"""
        summary = [
            f"处理完成!",
            f"成功处理: {results['successful']} 张图片",
            f"失败: {results['failed']} 张图片",
            f"跳过: {results['skipped']} 张图片",
            f"\n保存到文件夹: {batch_folder}"
        ]
        
        if results['error_files']:
            summary.append("\n错误列表:")
            summary.extend(results['error_files'])
            
        return "\n".join(summary)

    if not files:
        message_manager.add_warning("未选择要批量处理的文件")
        return "请上传一些图片进行处理。", [], (None, None)
        
    results = {
        'successful': 0,
        'failed': 0,
        'skipped': 0,
        'processed_files': [],
        'error_files': []
    }
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif', '.avif'}
    
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    batch_folder = os.path.join(save_path, f"batch_{timestamp}")
    os.makedirs(batch_folder, exist_ok=True)
    message_manager.add_message(f"创建批处理文件夹: {batch_folder}")
    
    current_image_pair = (None, None)
    
    try:
        total_files = len(files)
        message_manager.add_message(f"开始批量处理 {total_files} 个文件")
        
        for i, file in enumerate(files, 1):
            try:
                # Update progress
                progress(i/total_files, f"Processing {i}/{total_files}")
                message_manager.add_message(f"正在处理第 {i}/{total_files} 个文件: {file.name}")
                
                # Check file extension
                file_ext = os.path.splitext(file.name)[1].lower()
                if file_ext not in valid_extensions:
                    message_manager.add_warning(f"跳过不支持的文件: {file.name}")
                    results['skipped'] += 1
                    results['error_files'].append(f"{os.path.basename(file.name)} (不支持的格式)")
                    
                    if i == total_files:
                        message_manager.add_success("批量处理完成")
                        yield generate_summary(), update_gallery(), current_image_pair
                    else:
                        yield (
                            f"正在处理第 {i}/{total_files} 个文件: {file.name}",
                            update_gallery(),
                            current_image_pair
                        )
                    continue
                
                # Load and process image
                input_image = Image.open(file.name).convert("RGB")
                
                # 处理前清理内存
                gc.collect()
                devicetorch.empty_cache(torch)
                
                # Process with the same parameters as single image processing
                actual_seed = get_seed(seed, reuse_seed)
                manual_seed(actual_seed)
                solver_type: type[Solver] = getattr(solvers, solver)
                
                with torch.no_grad():
                    enhanced_image = enhancer.upscale(
                        image=input_image,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        upscale_factor=upscale_factor,
                        controlnet_scale=controlnet_scale,
                        controlnet_scale_decay=controlnet_decay,
                        condition_scale=condition_scale,
                        tile_size=(tile_height, tile_width),
                        denoise_strength=denoise_strength,
                        num_inference_steps=num_inference_steps,
                        loras_scale={"more_details": 0.5, "sdxl_render": 1.0},
                        solver_type=solver_type,
                    )
                
                # Update the current image pair for the slider
                current_image_pair = (input_image, enhanced_image)
                
                # Save enhanced image to batch folder
                original_name = Path(file.name).stem
                enhanced_filename = f"{original_name}_enhanced.png"
                output_path = os.path.join(batch_folder, enhanced_filename)
                enhanced_image.save(output_path, "PNG")
                
                # Update results
                results['successful'] += 1
                results['processed_files'].append(enhanced_filename)
                message_manager.add_success(f"已保存: {enhanced_filename}")
                
                # For the last file, show the summary instead of progress
                if i == total_files:
                    message_manager.add_success("批量处理完成")
                    yield generate_summary(), update_gallery(), current_image_pair
                else:
                    yield (
                        f"正在处理第 {i}/{total_files} 个文件: {file.name}",
                        update_gallery(),
                        current_image_pair
                    )
                
                # 处理后清理内存
                gc.collect()
                devicetorch.empty_cache(torch)
                
            except Exception as e:
                message_manager.add_error(f"Error processing {file.name}: {str(e)}")
                results['failed'] += 1
                results['error_files'].append(f"{os.path.basename(file.name)} ({str(e)})")
                
                if i == total_files:
                    message_manager.add_success("批量处理完成")
                    yield generate_summary(), update_gallery(), current_image_pair
        
        # Final return for gradio
        return generate_summary(), update_gallery(), current_image_pair
        
    except Exception as e:
        error_msg = f"批量处理错误: {str(e)}"
        message_manager.add_error(error_msg)
        return error_msg, [], (None, None)
            
            
def open_output_folder() -> None:
    folder_path = os.path.abspath(save_path)
    
    try:
        os.makedirs(folder_path, exist_ok=True)
    except Exception as e:
        message_manager.add_error(f"创建文件夹时出错: {str(e)}")
        return
        
    try:
        if os.name == 'nt':  # Windows
            subprocess.run(['explorer', folder_path])
        elif os.name == 'posix':  # macOS and Linux
            subprocess.run(['xdg-open' if os.name == 'posix' else 'open', folder_path])
        message_manager.add_success(f"已打开输出文件夹: {folder_path}")
    except Exception as e:
        message_manager.add_error(f"打开文件夹时出错: {str(e)}")


def save_output(image: Image.Image = None, auto_saved: bool = False) -> List[str]:
    """保存图片并返回更新后的图库数据"""
    if image is None:
        if not globals().get('latest_result'):
            message_manager.add_warning("没有可保存的图片! 请先增强图片。")
            return []
        image = latest_result
        
    try:
        # Create output directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"enhanced_{timestamp}.png"
        filepath = os.path.join(save_path, filename)
        
        # Save the image
        image.save(filepath, "PNG")
        
        save_type = "自动保存" if auto_saved else "已保存"
        message = f"图片{save_type}为: {filename}"
        message_manager.add_success(message)
        
        # Return updated gallery data
        return update_gallery()
        
    except Exception as e:
        error_msg = f"保存图片时出错: {str(e)}"
        message_manager.add_error(error_msg)
        return []
        
        
def process_and_update(*args):
    """处理输出和图库更新的包装函数"""
    result = process(*args)  # 获取滑块图片
    return result, update_gallery()  # 获取当前图库状态
    
    
def update_gallery() -> List[str]:
    """使用保存路径和批处理文件夹中的最新图片更新图库。"""
    try:
        # 从主保存路径和批处理子文件夹获取所有图片
        batch_folders = [d for d in os.listdir(save_path) 
                        if os.path.isdir(os.path.join(save_path, d)) 
                        and d.startswith('batch_')]
        
        # 收集主文件夹和所有批处理文件夹中的图片
        all_images = []
        
        # 主文件夹图片
        main_images = [
            os.path.join(save_path, f) 
            for f in os.listdir(save_path) 
            if f.lower().endswith(tuple(VALID_EXTENSIONS))
        ]
        all_images.extend(main_images)
        
        # 批处理文件夹图片
        for batch_folder in batch_folders:
            folder_path = os.path.join(save_path, batch_folder)
            batch_images = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(tuple(VALID_EXTENSIONS))
            ]
            all_images.extend(batch_images)
        
        # 按最新优先排序并限制数量
        all_images.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return all_images[:MAX_GALLERY_IMAGES]
        
    except Exception as e:
        message_manager.add_error(f"更新图库时出错: {str(e)}")
        return []


css = """
/* 图片的具体调整 */
.image-container .image-custom {
    max-width: 100% !important;
    max-height: 80vh !important;
    width: auto !important;
    height: auto !important;
}

/* 居中 ImageSlider 容器并保持滑块的完整宽度 */
.image-container .image-slider-custom {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
}

/* 滑块容器的样式 */
.image-container .image-slider-custom > div {
    width: 100% !important;
    max-width: 100% !important;
    max-height: 80vh !important;
}

/* 确保前后图片保持宽高比 */
.image-container .image-slider-custom img {
    max-height: 80vh !important;
    width: 100% !important;
    height: auto !important;
    object-fit: contain !important;
}

/* 滑块手柄的样式 */
.image-container .image-slider-custom .image-slider-handle {
    width: 2px !important;
    background: white !important;
    border: 2px solid rgba(0, 0, 0, 0.6) !important;
}

/* 控制台滚动样式 */
.console-scroll textarea {
    max-height: 12em !important;  /* Approximately 8 lines of text */
    overflow-y: auto !important;  /* Enables vertical scrolling */
}

/* 状态特定样式 */
.batch-status textarea {
    min-height: 12em !important;  /* Ensures minimum height for welcome message */
}

/* 提示指南样式 */
.prompt-guide textarea {
    white-space: pre-wrap !important;  /* Preserves formatting but allows wrapping */
    padding-left: 1em !important;      /* Base padding for all text */
    text-indent: -1em !important;      /* Negative indent for first line */
}

"""

# 存储最新的处理结果
latest_result = None

with gr.Blocks(css=css) as demo:
    with gr.Row():
        with gr.Column(elem_classes="image-container"):
            with gr.Tabs() as tabs:
                with gr.TabItem("单张图片") as single_tab:
                    input_image = gr.Image(type="pil", label="输入图片", elem_classes=["image-custom"])
                    run_button = gr.ClearButton(
                        components=None,
                        value="增强图片",
                        variant="primary"
                    )
                with gr.TabItem("批量处理") as batch_tab:
                    batch_welcome = """✨ 欢迎使用批量处理! ✨

📸 拖放多个图片进行批量增强:
    
• 所有增强设置将应用于每张图片
   (提示词、降噪、种子等 - 注意:种子默认随机)
• 图片将保存到带时间戳的批处理文件夹
• 增强效果会在预览窗口显示
• 在此处跟踪进度 + 主控制台显示详细信息

🚀 准备好了吗? 拖入图片并点击'开始批量处理'!"""

                    input_files = gr.File(
                        file_count="multiple",
                        label="加载图片",
                        scale=2
                    )
                    batch_status = gr.Textbox(
                        label="批处理状态",
                        value=batch_welcome,
                        interactive=False,
                        show_copy_button=True,
                        autoscroll=True,
                        elem_classes="batch-status"
                    )
                    batch_button = gr.Button(
                        "开始批量处理",
                        variant="primary"
                    )
           
        with gr.Column(elem_classes="image-container"):
            output_slider = ImageSlider(
                interactive=False,
                label="前后对比",
                elem_classes=["image-slider-custom"]
            )
            run_button.add(output_slider)
            with gr.Row():
                save_result = gr.Button("保存结果", scale=2)
                auto_save = gr.Checkbox(label="自动保存", value=True)
                open_folder_button = gr.Button("打开输出文件夹", scale=2)

    with gr.Accordion("提示词设置", open=False):
        with gr.Row():
            with gr.Tabs():
                with gr.TabItem("提示词"):
                    prompt = gr.Textbox(
                        label="增强提示词",
                        value="masterpiece, best quality, highres",
                        show_label=True
                    )
                with gr.TabItem("指南"):
                    prompt_guide = gr.Textbox(
    value="""💡 提示词指南

🎯 附加提示词是可选的！
• 它们的工作方式类似于其他AI生成应用中的img2img和controlnet提示词
• 默认设置对一般增强效果很好
• 使用提示词来引导AI进行特定的改进
• 保持提示词简单并专注于你想要增强的部分

📝 示例提示词:
• "sharp details, high quality" - 提高清晰度和细节
• "vivid colors, high contrast" - 获得更生动的效果
• "soft lighting, smooth details" - 获得更柔和的增强效果
• "perfect eyes", "green eyes", "detailed fingernails" 等 - 关注特定细节

💭 提示:
• 从默认提示词开始，需要时再添加特定引导
• Florence2自动提示是完全可选的。添加它主要是因为为什么不呢😆""", 

                        label="使用提示词",
                        interactive=False,
                        show_label=False,
                        lines=12,
                        elem_classes="prompt-guide"
                    )
            with gr.Column(scale=1):
                caption_detail = gr.Radio(
                    choices=["<CAPTION>","<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"],
                    value="<CAPTION>",
                    label="Florence-2 描述详程度",
                    info="选择图像分析的详细程度"
                )
                generate_prompt_btn = gr.Button("📝 生成提示词", variant="primary")
        with gr.Row():
            negative_prompt = gr.Textbox(
                label="反向提示词",
                value="worst quality, low quality, normal quality",
            )
        with gr.Row():
            with gr.Column(scale=8):
                seed = gr.Slider(
                    minimum=-1,
                    maximum=10_000,
                    step=1,
                    value=-1,
                    label="种子值 (-1为随机)"
                )
            with gr.Column(scale=1):
                reuse_seed = gr.Checkbox(label="重用上次种子值", value=False)
                
    with gr.Accordion("基本选项", open=False):
        with gr.Row():    
            upscale_factor = gr.Slider(
                minimum=1,
                maximum=4,
                value=2,
                step=0.2,
                label="放大倍数",
            )
            denoise_strength = gr.Slider(
                minimum=0.05,
                maximum=1,
                value=0.15,
                step=0.05,
                label="降噪强度", 
                info="设为最小值获得更传统的放大效果",
            )
            num_inference_steps = gr.Slider(
                minimum=1,
                maximum=50,
                value=20,
                step=1,
                label="推理步数",
            )
    with gr.Accordion("高级选项", open=False):
        with gr.Row(): 
            controlnet_scale = gr.Slider(
                minimum=0,
                maximum=1.5,
                value=0.6,
                step=0.1,
                label="ControlNet强度",
            )
            controlnet_decay = gr.Slider(
                minimum=0.5,
                maximum=1,
                value=1.0,
                step=0.025,
                label="ControlNet衰减",
            )
            condition_scale = gr.Slider(
                minimum=2,
                maximum=20,
                value=6,
                step=1,
                label="条件缩放",
            )
        with gr.Row(): 
            tile_width = gr.Slider(
                minimum=64,
                maximum=200,
                value=112,
                step=1,
                label="潜空间切片宽度",
            )
            tile_height = gr.Slider(
                minimum=64,
                maximum=200,
                value=144,
                step=1,
                label="潜空间切片高度",
            )
            solver = gr.Radio(
                choices=["DDIM", "DPMSolver"],
                value="DDIM",
                label="求解器",
            )
    with gr.Accordion("系统信息和控制台", open=True):            
        with gr.Row():       
            # Status Info (for cpu/gpu monitor)
            resource_monitor = gr.Textbox(
                label="系统监视器",
                lines=8,
                interactive=False,
                # value=get_welcome_message()
            )  
            console_out = gr.Textbox(
                label="控制台",
                lines=8,
                interactive=False,
                show_copy_button=True,
                autoscroll=True,    # Enables automatic scrolling to newest messages
                elem_classes="console-scroll"  # Add custom class for styling
            )
 
    with gr.Accordion("图库", open=False):     
        with gr.Row():
            gallery = gr.Gallery(
                label="最近增强",
                show_label=True,
                elem_id="gallery",
                columns=5,
                rows=6,
                height="80vh",  # Use viewport height instead of fixed pixels
                object_fit="contain",
                allow_preview=True,
                show_share_button=False,
                show_download_button=True,
                preview=True,
            )
            
    # Event handlers
    
    generate_prompt_btn.click(
        fn=generate_prompt,
        inputs=[input_image, caption_detail],
        outputs=[prompt]
    )
    
    run_button.click(
        fn=process_and_update,
        inputs=[
            input_image,
            prompt,
            negative_prompt,
            seed,
            reuse_seed,
            upscale_factor,
            controlnet_scale,
            controlnet_decay,
            condition_scale,
            tile_width,
            tile_height,
            denoise_strength,
            num_inference_steps,
            solver,
            auto_save,
        ],
        outputs=[output_slider, gallery]
    )
    
    batch_button.click(
        fn=batch_process_images,
        inputs=[
            input_files,
            prompt,
            negative_prompt,
            seed,
            reuse_seed,
            upscale_factor,
            controlnet_scale,
            controlnet_decay,
            condition_scale,
            tile_width,
            tile_height,
            denoise_strength,
            num_inference_steps,
            solver,
        ],
        outputs=[batch_status, gallery, output_slider]
    )
    
    save_result.click(
        fn=save_output,
        inputs=None,
        outputs=[gallery]
    )
    
    open_folder_button.click(
        fn=open_output_folder,
        inputs=None,
        outputs=gr.Text(visible=False) 
    )
    
    def update_console():
        return message_manager.get_messages()
    
    # Initialize the timer and set up its tick event
    demo.load(
        fn=lambda: (SystemMonitor.get_system_info(), update_console()),
        outputs=[resource_monitor, console_out],
        every=1  # Updates every 1 second
    )
    
demo.launch(share=False)
