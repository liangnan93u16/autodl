#include "scripts.h"

// allegrotxt2vid_install.sh 内容
const char* ALLEGROTXT2VID_INSTALL = 
    "#!/bin/bash\n"
    "\n"
    "# Network acceleration\n"
    ". /etc/network_turbo || true\n"
    "\n"
    "apt-get update\n"
    "apt-get install sudo\n"
    "\n"
    "# Install system dependencies\n"
    "apt-get update\n"
    "apt-get install -y portaudio19-dev python3-pyaudio\n"
    "# Create and activate conda environment\n"
    "conda create -n Allegro-txt2vid python=3.10 -y\n"
    "conda activate Allegro-txt2vid\n"
    "\n"
    "# Configure conda\n"
    ". $HOME/miniconda3/etc/profile.d/conda.sh || . $HOME/anaconda3/etc/profile.d/conda.sh\n"
    "\n"
    "# Clone repository and enter directory\n"
    "git clone https://github.com/pinokiofactory/Allegro-txt2vid Allegro-txt2vid\n"
    "cd Allegro-txt2vid\n"
    "\n"
    "# Install PyTorch and dependencies\n"
    "pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 xformers --index-url https://download.pytorch.org/whl/cu121\n"
    "\n"
    "# Install project and additional dependencies\n"
    "pip install gradio devicetorch\n"
    "pip install -r requirements.txt";

// allegrotxt2vid_start.sh 内容
const char* ALLEGROTXT2VID_START = 
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
    "conda activate Allegro-txt2vid || {\n"
    "    echo \"Error: Failed to activate conda environment 'Allegro-txt2vid'\"\n"
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
    "# 进入工作目录\n"
    "cd $HOME/Allegro-txt2vid\n"
    "\n"
    "# 运行应用\n"
    "export GRADIO_SERVER_PORT=6006\n"
    "python gradio_app.py\n"
    "";

