#include "scripts.h"

// logocreator_start.sh 内容
const char* LOGOCREATOR_START = 
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
    ". $HOME/miniconda3/etc/profile.d/conda.sh || . $HOME/anaconda3/etc/profile.d/conda.sh\n"
    "\n"
    "# 激活 conda 环境\n"
    "conda activate logocreator || {\n"
    "    echo \"Error: Failed to activate conda environment 'logocreator'\"\n"
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
    "# 创建项目目录\n"
    "cd $HOME/logocreator\n"
    "npm config set strict-ssl false\n"
    "#npm start\n"
    "npm run dev -- -p 6006";

// logocreator_install.sh 内容
const char* LOGOCREATOR_INSTALL = 
    "#!/bin/bash\n"
    "\n"
    "# Network acceleration\n"
    ". /etc/network_turbo || true\n"
    "\n"
    "# Install system dependencies\n"
    "apt update\n"
    "\n"
    "# Install Node.js and npm\n"
    "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -\n"
    "apt install -y nodejs\n"
    "\n"
    ". $HOME/miniconda3/etc/profile.d/conda.sh || . $HOME/anaconda3/etc/profile.d/conda.sh\n"
    "\n"
    "conda clean --all -y  # 清理所有缓存\n"
    "conda create -n logocreator python=3.10 -y \n"
    "conda activate logocreator \n"
    "\n"
    "# Clone repository and enter directory\n"
    "cd $HOME\n"
    "rm -rf logocreator\n"
    "git clone https://github.com/Nutlope/logocreator.git logocreator\n"
    "cd logocreator\n"
    "\n"
    "# new env\n"
    "rm -rf .env\n"
    "cp .env.example .env\n"
    "\n"
    "# Install npm dependencies\n"
    "# Configure npm to use Taobao registry\n"
    "unset http_proxy && unset https_proxy\n"
    "npm config set strict-ssl false\n"
    "npm config set registry https://registry.npmmirror.com\n"
    "\n"
    "npm install\n"
    "\n"
    "# 提示用户输入 API key\n"
    "echo \"请输入您的 TOGETHER_API_KEY:\"\n"
    "read api_key\n"
    "\n"
    "# 更新 .env 文件中的 API key\n"
    "sed -i \"s/TOGETHER_API_KEY=/TOGETHER_API_KEY=$api_key/\" .env\n"
    "\n"
    "echo \"TOGETHER_API_KEY 已更新到 .env 文件中\"\n"
    "\n"
    "# 创建 .env.local 文件并获取 Clerk keys\n"
    "rm -rf .env.local\n"
    "touch .env.local\n"
    "\n"
    "echo \"请输入您的 NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY:\"\n"
    "read clerk_pub_key\n"
    "\n"
    "echo \"请输入您的 CLERK_SECRET_KEY:\"\n"
    "read clerk_secret_key\n"
    "\n"
    "# 将 keys 写入 .env.local 文件\n"
    "echo \"NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=$clerk_pub_key\" >> .env.local\n"
    "echo \"CLERK_SECRET_KEY=$clerk_secret_key\" >> .env.local\n"
    "\n"
    "echo \".env.local 文件已创建并更新完成\"\n"
    "";

