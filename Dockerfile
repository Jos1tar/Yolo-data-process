# 使用官方 Python 3.11 轻量级镜像作为基础
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 升级 pip 到最新版本，这是一个好习惯
RUN pip install --no-cache-dir --upgrade pip

# --- 修改部分开始 ---
# 一次性复制所有 requirements 文件
COPY requirements*.txt ./

# 在一个指令中安装所有依赖
# pip 可以接受多个 -r 参数来安装来自不同文件的包
RUN pip install --no-cache-dir -r requirements-base.txt -r requirements.txt
# --- 修改部分结束 ---

# 将项目代码复制到工作目录中
# 这一步放在安装依赖之后，可以利用缓存
# 这样代码的改动不会导致依赖被重新安装
COPY . .

# 声明容器对外暴露的端口
EXPOSE 8000

# 定义容器启动时执行的命令
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]