
# 支持自动安装的开源项目
- [applio ]([url](https://github.com/IAHispano/Applio))
- fish-speech

# 环境要求
适配网站：https://www.autodl.com/home

基础镜像=>Miniconda=>conda3=>3.10(ubuntu22.04)=>Cuda版本11.8

PS:只在这个环境测试过，其他的Miniconda应该也可以的。

# 执行命令
wget https://raw.githubusercontent.com/liangnan93u16/AutoDLInstallScript/main/auto_dl_install && chmod +x auto_dl_install && ./auto_dl_install

# 安装说明
1. 自动下载对应开源项目的代码
2. 自动安装依赖环境
3. 自动安装：torch==2.4.1 torchaudio==2.4.1

