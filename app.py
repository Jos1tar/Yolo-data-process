from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import os
import numpy as np

import preProcessor

MODEL_PATH = os.environ.get("MODEL_PATH", "yolov8n.onnx")

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "greeting service"}


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(i) for i in obj)
    elif isinstance(obj, (np.int32, np.int64, np.float32, np.float64)):
        return obj.item()  # 转为 Python 原生数值类型
    else:
        return obj

@app.post("/YOLOpredict/")
async def predict(file: UploadFile = File(...)):
    # 只取文件名部分，防止 "../../xxx" 这类文件名写到上传目录之外
    filename = os.path.basename(file.filename or "upload.jpg")
    os.makedirs("runs/uploaded", exist_ok=True)
    temp_path = os.path.join("runs/uploaded", filename)

    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = preProcessor.ini(temp_path, model_path=MODEL_PATH)

    result = convert_to_serializable(result)

    return JSONResponse(content=result)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
