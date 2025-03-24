# Meeting Minutes Generator
一个基于Whisper和DeepSeek R1的本地会议记录生成工具，支持多发言人识别。

## 安装
1. `pip install -r requirements.txt`
2. 下载DeepSeek R1模型到`models/`目录。
3. 运行`streamlit run app.py`。

## 使用
- 上传音频文件，指定模型路径，点击生成按钮。

在VS Code中的使用建议
项目初始化：
在VS Code中打开MeetingMinutesGenerator/文件夹。
创建虚拟环境：
bash

收起

自动换行

复制
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
代码调试：
使用VS Code的Python扩展，设置断点调试app.py。
配置launch.json运行Streamlit：
json

收起

自动换行

复制
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Run Streamlit",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/app.py",
            "args": ["run", "${file}"],
            "console": "integratedTerminal",
            "justMyCode": true
        }
    ]
}
文件管理：
将音频文件放入data/input/，运行后检查data/output/。