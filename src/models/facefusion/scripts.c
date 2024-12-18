#include "scripts.h"

// facefusion_start.sh 内容
const char* FACEFUSION_START = 
    "#!/bin/bash\n"
    "\n"
    "# Network acceleration\n"
    ". /etc/network_turbo || true\n"
    "\n"
    "# 确保 bash 环境\n"
    "if [ -z \"$BASH_VERSION\" ]; then\n"
    "    exec bash \"$0\" \"$@\"\n"
    "fi\n"
    "\n"
    "# 先进入用户主目录\n"
    "cd $HOME\n"
    "\n"
    "# 初始化 conda\n"
    "if [ -f \"/root/miniconda3/etc/profile.d/conda.sh\" ]; then\n"
    "    . \"/root/miniconda3/etc/profile.d/conda.sh\"\n"
    "else\n"
    "    echo \"Error: Could not find conda installation at /root/miniconda3\"\n"
    "    exit 1\n"
    "fi\n"
    "\n"
    "# Configure conda\n"
    ". $HOME/miniconda3/etc/profile.d/conda.sh || . $HOME/anaconda3/etc/profile.d/conda.sh\n"
    "\n"
    "# 激活 conda 环境\n"
    "conda activate facefusion || {\n"
    "    echo \"Error: Failed to activate conda environment 'facefusion'\"\n"
    "    exit 1\n"
    "}\n"
    "\n"
    "\n"
    "# 检测 NVIDIA GPU\n"
    "DEVICE=\"cuda\"\n"
    "\n"
    "# 运行应用\n"
    "cd $HOME/facefusion/\n"
    "export GRADIO_SERVER_PORT=6006\n"
    "python facefusion.py run";

// facefusion_install.sh 内容
const char* FACEFUSION_INSTALL = 
    "#!/bin/bash\n"
    "\n"
    "# Network acceleration\n"
    ". /etc/network_turbo || true\n"
    "\n"
    "# Ensure script is run with bash\n"
    "if [ -z \"$BASH_VERSION\" ]; then\n"
    "    exec bash \"$0\" \"$@\"\n"
    "fi\n"
    "\n"
    "# Install system dependencies\n"
    "apt update\n"
    "apt install -y portaudio19-dev python3-pyaudio\n"
    "\n"
    "# Install PyTorch and dependencies\n"
    "conda install -c conda-forge cuda-runtime=12.4.1 -y\n"
    "conda install -c conda-forge cudnn=9.2.1.18 -y\n"
    "conda install -c conda-forge ffmpeg -y\n"
    "\n"
    "# Configure conda\n"
    ". $HOME/miniconda3/etc/profile.d/conda.sh || . $HOME/anaconda3/etc/profile.d/conda.sh\n"
    "\n"
    "# Create and activate conda environment\n"
    "conda clean --all -y  # 清理所有缓存\n"
    "conda create -n facefusion python=3.10 -y\n"
    "conda activate facefusion\n"
    "\n"
    "# Clone repository and enter directory\n"
    "cd $HOME\n"
    "rm -rf facefusion\n"
    "git clone https://github.com/facefusion/facefusion --branch 3.0.1 --single-branch facefusion\n"
    "cd facefusion\n"
    "\n"
    "pip install tensorrt==10.5.0 --extra-index-url https://pypi.nvidia.com\n"
    "\n"
    "python install.py --onnxruntime cuda\n"
    "\n"
    "cd $HOME/facefusion/facefusion\n"
    "curl -o wording.py https://raw.githubusercontent.com/liangnan93u16/autodl/refs/heads/main/facefusion/facefusion/wording.py\n"
    "\n"
    "";

