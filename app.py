from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import os
import numpy as np
import model_trans
import preProcessor
import logging
app = FastAPI()

UPLOAD_DIR = "uploads"
ONNX_OUTPUT_DIR = "runs/onnx"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ONNX_OUTPUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.get("/")
async def root():
    return {"message": "greeting service"}


@app.post("/test_upload/")
async def test_upload(file: UploadFile = File(...)):
    return {"filename": file.filename}


@app.post("/convert/yolo_to_onnx/")
async def convert_pt_to_onnx(file: UploadFile = File(...)):
    try:
        # 1. 保存上传的 .pt 文件
        pt_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(pt_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 2. 生成ONNX模型路径名
        onnx_name = os.path.splitext(file.filename)[0] + ".onnx"

        # 3. 调用转换函数
        onnx_path = model_trans.convert_yolov8_to_onnx(pt_path, output_dir=ONNX_OUTPUT_DIR, onnx_name=onnx_name)

        return JSONResponse({
            "message": "转换成功",
            "onnx_model_path": onnx_path
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


def convert_to_serializable(result):
    """
    将 numpy 推理结果转换为 JSON 可序列化的格式。

    输入:
        result: ndarray, shape=(N, 6)，每一行为[x1, y1, x2, y2, conf, cls_id]

    输出:
        List[dict]，例如:
        [
            {"bbox": [x1, y1, x2, y2], "confidence": 0.92, "class_id": 0},
            ...
        ]
    """
    if isinstance(result, np.ndarray):
        serializable_result = []
        for row in result:
            x1, y1, x2, y2, conf, cls_id = row.tolist()
            serializable_result.append({
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "confidence": round(conf, 4),
                "class_id": int(cls_id)
            })
        return serializable_result

    elif isinstance(result, list):
        # 已经是 Python list 的情况（保险）
        return result

    else:
        raise TypeError("输入 result 类型不支持，必须是 np.ndarray 或 list。")


@app.post("/YOLOpredict/")
async def predict(file: UploadFile = File(...)):
    temp_path = f"runs/uploaded/{file.filename}"
    os.makedirs("runs/uploaded", exist_ok=True)
    os.makedirs("runs/detect/myPredict", exist_ok=True)

    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = preProcessor.ini(temp_path)

    # 检查 result 的类型和内容
    if not isinstance(result, (list, tuple)) or len(result) < 3:
        return JSONResponse(
            content={"error": "预处理返回的结果格式不正确"},
            status_code=500
        )

    # 转换检测结果为可序列化格式
    result = list(result)  # 确保 result 是可变的
    result[2] = convert_to_serializable(result[2])

    return JSONResponse(content=result)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
