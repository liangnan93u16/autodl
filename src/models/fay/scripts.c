#include "scripts.h"

// fay_install.sh 内容
const char* FAY_INSTALL = 
    "#!/bin/bash\n"
    "\n"
    "# 将环境名称定义为常量\n"
    "APP=\"fay\"\n"
    "\n"
    "# 网络加速\n"
    ". /etc/network_turbo || true\n"
    "\n"
    "# 更新软件包列表\n"
    "apt update\n"
    "apt-get update && apt-get install -y libxtst6 libxrender1 libxi6 socat\n"
    "\n"
    "# 配置conda环境\n"
    ". $HOME/miniconda3/etc/profile.d/conda.sh || . $HOME/anaconda3/etc/profile.d/conda.sh\n"
    "\n"
    "# 创建并激活conda环境\n"
    "conda clean --all -y  # 清理所有缓存\n"
    "conda create -n $APP python=3.10 -y\n"
    "conda activate $APP\n"
    "\n"
    "# 克隆代码仓库\n"
    "cd $HOME\n"
    "rm -rf $APP\n"
    "git clone https://github.com/xszyou/Fay $APP\n"
    "cd $APP\n"
    "\n"
    "# 安装 portaudio 依赖\n"
    "apt-get install -y libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg\n"
    "\n"
    "# 安装依赖包\n"
    "pip install -r requirements.txt\n"
    "\n"
    "cd $HOME/$APP/asr/funasr\n"
    "pip install torch\n"
    "pip install modelscope\n"
    "pip install testresources\n"
    "pip install websockets\n"
    "pip install torchaudio\n"
    "pip install FunASR";

// fay_start.sh 内容
const char* FAY_START = 
    "#!/bin/bash\n"
    "\n"
    "# 将环境名称定义为常量\n"
    "APP=\"fay\"\n"
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
    "# 检测 NVIDIA GPU\n"
    "if command -v nvidia-smi &> /dev/null; then\n"
    "    DEVICE=\"cuda\"\n"
    "else\n"
    "    DEVICE=\"cpu\"\n"
    "fi\n"
    "\n"
    "cd $HOME/$APP/asr/funasr\n"
    "nohup python -u ASR_server.py --host \"0.0.0.0\" --port 10197 --ngpu 0 > ASR_server.log 2>&1 & \n"
    "echo \"ASR_server.log 启动成功\"\n"
    "\n"
    "# 运行应用\n"
    "cd $HOME/$APP/\n"
    "nohup python -u main.py > fay.log 2>&1 & \n"
    "echo \"fay.log 启动成功\"\n"
    "\n"
    "# 记录进程ID到文件中,方便后续管理\n"
    "echo $! > fay.pid\n"
    "\n"
    "socat TCP-LISTEN:6006,reuseaddr,fork TCP:localhost:5000";

