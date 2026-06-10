from ultralytics import YOLO

# 1. 모델 선언 (인터넷에서 yolov8n.pt 자동 다운로드)
model = YOLO("yolov8n.pt")

# 2. ONNX 포맷으로 변환 및 저장 (opset=12 지정으로 OpenCV 호환성 확보)
model.export(format="onnx", opset=12)