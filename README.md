# YOLO inference service (ONNX Runtime / OpenVINO)

A FastAPI service for YOLOv8 object detection. The model is exported from PyTorch to ONNX (and optionally to
OpenVINO IR), and all pre- and post-processing is done in NumPy and OpenCV, so the service does not depend on
ultralytics at runtime.

## Pipeline

1. Letterbox: resize the long side to 640 keeping the aspect ratio, pad with gray (114) to 640x640,
   BGR to RGB, scale to [0, 1], HWC to NCHW.
2. Inference with ONNX Runtime (`preProcessor.ini`) or OpenVINO (`preProcessor.run_inference`).
   Output shape is `(1, 84, 8400)`: cx, cy, w, h and 80 class scores per candidate.
3. Post-processing (`opPostprocessor.py`): take the best class per candidate, filter by confidence,
   per-class NMS, then undo the letterbox (subtract padding, divide by scale, clip to the image).
4. Draw boxes and save the annotated image.

On the sample images the output matches ultralytics running the same ONNX file to within 0.1 px and 0.001
confidence.

## Model

```
pip install ultralytics
python model_trans.py          # yolov8n.pt -> yolov8n.onnx -> openvino_models/yolov8n.xml
```

Put `yolov8n.onnx` next to `app.py`, or point `MODEL_PATH` to it.

## Run

```
pip install -r requirements.txt
uvicorn app:app --port 8000
```

```
curl -F "file=@runs/bus.jpg" http://127.0.0.1:8000/YOLOpredict/
```

The response contains `[x1, y1, x2, y2, conf, class_id]` for each box in original image coordinates and the path
of the annotated image (`runs/detect/myPredict/`).

Docker:

```
docker build -t yolo-service .
docker run -p 8000:8000 -v $(pwd)/yolov8n.onnx:/app/yolov8n.onnx yolo-service
```

## Files

```
app.py               FastAPI endpoint /YOLOpredict/
preProcessor.py      letterbox, ONNX Runtime / OpenVINO inference, request pipeline
opPostprocessor.py   decoding, NMS, coordinate restore, drawing
model_trans.py       PyTorch -> ONNX -> OpenVINO IR export
runs/                sample inputs and outputs
```

Other branches: `multibatch` (batched preprocessing and inference) and `with_nsight` (ONNX to TensorRT engine
conversion and TensorRT inference).

## Changes in Nov 2026

- The request handler called a method that did not exist (`cod_trf`) and passed the wrong arguments to
  `post_process`, so every request returned an error. It now goes through `post_pipeline`.
- `plt.show()` was called on every request and blocked the server; the visualization is now optional.
- The resize used `math.ceil`, which could produce 641 px and a negative padding. Downscaling now uses linear
  interpolation like ultralytics (it used `INTER_AREA`, which shifted confidences by about 0.01).
- Coordinate restore recomputed the scale and padding with different rounding than the preprocessing, giving
  1 px offsets; it now uses the values recorded during the letterbox.
- The OpenVINO conversion embedded ImageNet mean/std and a channel swap on top of the Python-side `/255` and
  BGR to RGB, so the input was normalized twice. The IR no longer embeds preprocessing, and the deprecated `mo`
  API was replaced by `openvino.convert_model`.
- The ONNX Runtime session was recreated on every request; it is now cached.
- Uploaded file names are reduced to their base name before saving.
- The Dockerfile referenced a missing `requirements-base.txt`; `requirements.txt` was trimmed (it contained
  unrelated packages and three conflicting OpenCV builds).
