#include "scripts.h"

// bolt_install.sh 内容
const char* BOLT_INSTALL = 
    "#!/bin/bash\n"
    "\n"
    "APP=\"bolt\"\n"
    "# Network acceleration\n"
    ". /etc/network_turbo || true\n"
    "\n"
    "# Install system dependencies\n"
    "apt update\n"
    "apt install -y portaudio19-dev python3-pyaudio curl socat\n"
    "\n"
    "# Install Node.js and npm\n"
    "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -\n"
    "apt install -y nodejs\n"
    "\n"
    ". $HOME/miniconda3/etc/profile.d/conda.sh || . $HOME/anaconda3/etc/profile.d/conda.sh\n"
    "\n"
    "# Create and activate conda environment\n"
    "conda clean --all -y  # 清理所有缓存\n"
    "conda create -n $APP python=3.10 -y || {\n"
    "    echo \"Failed to create conda environment\"\n"
    "    exit 1\n"
    "}\n"
    "\n"
    "# Ensure conda environment is activated successfully\n"
    "conda activate $APP || {\n"
    "    echo \"Failed to activate conda environment\"\n"
    "    exit 1\n"
    "}\n"
    "\n"
    "# Clone repository and enter directory\n"
    "cd $HOME\n"
    "rm -rf $APP\n"
    "git clone https://github.com/coleam00/bolt.new-any-llm $APP\n"
    "cd $HOME/$APP\n"
    "\n"
    "# Install npm dependencies\n"
    "# Configure npm to use Taobao registry\n"
    "unset http_proxy && unset https_proxy\n"
    "npm config set strict-ssl false\n"
    "npm config set registry https://registry.npmmirror.com\n"
    "npm install\n"
    "\n"
    "pip install litellm[proxy]\n"
    "\n"
    ". /etc/network_turbo || true\n"
    "cd $HOME/$APP/app/components/chat\n"
    "curl -o BaseChat.tsx https://raw.githubusercontent.com/liangnan93u16/autodl/refs/heads/main/bolt/BaseChat.tsx";

// bolt_start.sh 内容
const char* BOLT_START = 
    "#!/bin/bash\n"
    "\n"
    "APP=\"bolt\"\n"
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
    "conda activate $APP || {\n"
    "    echo \"Error: Failed to activate conda environment '$APP'\"\n"
    "    exit 1\n"
    "}\n"
    "\n"
    "\n"
    "# 检测 NVIDIA GPU\n"
    "DEVICE=\"cuda\"\n"
    "\n"
    "# 运行应用\n"
    "cd $HOME/$APP/\n"
    "export NODE_TLS_REJECT_UNAUTHORIZED=0\n"
    "npm config set strict-ssl false\n"
    "npm run dev &\n"
    "\n"
    "# 等待几秒确保 npm run dev 启动完成\n"
    "sleep 2\n"
    "\n"
    "echo \"'$APP'启动成功,通过http://localhost:6006访问\"\n"
    "# 端口转发\n"
    "socat TCP-LISTEN:6006,reuseaddr,fork TCP:localhost:5173";

