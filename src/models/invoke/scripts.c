#include "scripts.h"

// invoke_install.sh 内容
const char* INVOKE_INSTALL = 
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
    "# Configure conda\n"
    ". $HOME/miniconda3/etc/profile.d/conda.sh || . $HOME/anaconda3/etc/profile.d/conda.sh\n"
    "\n"
    "# Create and activate conda environment\n"
    "conda clean --all -y  # 清理所有缓存\n"
    "conda create -n InvokeAI python=3.10 -y\n"
    "conda activate InvokeAI\n"
    "\n"
    "# Clone repository and enter directory\n"
    "git clone https://github.com/peanutcocktail/FacePoke InvokeAI\n"
    "cd InvokeAI\n"
    "\n"
    "# Install PyTorch and dependencies\n"
    "# pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121\n"
    "\n"
    "# Install project and additional dependencies\n"
    "pip install \"InvokeAI[xformers]\" --upgrade --use-pep517 --extra-index-url https://download.pytorch.org/whl/cu124";

// invoke_start.sh 内容
const char* INVOKE_START = 
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
    "conda activate InvokeAI || {\n"
    "    echo \"Error: Failed to activate conda environment 'InvokeAI'\"\n"
    "    exit 1\n"
    "}\n"
    "\n"
    "\n"
    "# 检测 NVIDIA GPU\n"
    "if command -v nvidia-smi &> /dev/null; then\n"
    "    DEVICE=\"cuda\"\n"
    "else\n"
    "    DEVICE=\"cpu\"\n"
    "fi\n"
    "\n"
    "# 运行应用\n"
    "cd $HOME/InvokeAI/\n"
    "export INVOKEAI_PORT=6006\n"
    "invokeai-web";

