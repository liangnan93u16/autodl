#include "scripts.h"

// fish_speech_install.sh 内容
const char* FISH_SPEECH_INSTALL = 
    "#!/bin/bash\n"
    "\n"
    "# Network acceleration\n"
    ". /etc/network_turbo || true\n"
    "\n"
    "# Install system dependencies\n"
    "apt-get update\n"
    "apt-get install -y portaudio19-dev python3-pyaudio\n"
    "\n"
    "# Configure conda\n"
    "source $HOME/miniconda3/etc/profile.d/conda.sh || source $HOME/anaconda3/etc/profile.d/conda.sh\n"
    "\n"
    "# Create and activate conda environment\n"
    "conda create -n fish-speech python=3.10 -y\n"
    "conda activate fish-speech\n"
    "\n"
    "# Clone repository and enter directory\n"
    "git clone https://github.com/fishaudio/fish-speech fish-speech\n"
    "cd fish-speech\n"
    "\n"
    "# Install PyTorch and dependencies\n"
    "pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121\n"
    "\n"
    "# Install project and additional dependencies\n"
    "pip install -e .\n"
    "pip install cachetools livekit livekit-agents\n"
    "\n"
    "# Download model checkpoints\n"
    "huggingface-cli download fishaudio/fish-speech-1.4 --local-dir checkpoints/fish-speech-1.4\n"
    "";

// fish_speech_start.sh 内容
const char* FISH_SPEECH_START = 
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
    "# 激活 conda 环境\n"
    "conda activate fish-speech || {\n"
    "    echo \"Error: Failed to activate conda environment 'fish-speech'\"\n"
    "    exit 1\n"
    "}\n"
    "\n"
    "# 检测 NVIDIA GPU\n"
    "if command -v nvidia-smi &> /dev/null; then\n"
    "    DEVICE=\"cuda\"\n"
    "else\n"
    "    DEVICE=\"cpu\"\n"
    "fi\n"
    "\n"
    "# 确保目录存在\n"
    "mkdir -p $HOME/fish-speech/tools\n"
    "\n"
    "# 进入工作目录\n"
    "cd $HOME/fish-speech/tools\n"
    "\n"
    "# 下载 webui2.py\n"
    "curl -o webui2.py https://raw.githubusercontent.com/liangnan93u16/autodl/refs/heads/main/fish_speech/webui2.py\n"
    "\n"
    "# 创建软链接（如果不存在）\n"
    "if [ ! -L \"checkpoints\" ]; then\n"
    "    ln -sf ../checkpoints checkpoints\n"
    "fi\n"
    "\n"
    "# 运行应用\n"
    "python webui2.py --device $DEVICE\n"
    "";

