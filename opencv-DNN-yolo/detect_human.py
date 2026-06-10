import cv2
from ultralytics import YOLO

# 1. YOLOv8 가벼운 Nano 모델 로드 (최초 실행 시 자동 다운로드)
model = YOLO("yolov8n.pt")

# 2. USB 카메라 연결 (기본 카메라 인덱스는 0, 작동 안 할 시 1 또는 2)
cap = cv2.VideoCapture(0)

# 카메라 해상도 설정 (성능 최적화를 위해 640x480 권장)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: USB 카메라를 열 수 없습니다.")
    exit()

print("사람 인식 시작... 종료하려면 'q' 키를 누르세요.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. YOLOv8 추론 수행 (stream=True 설정을 통해 메모리 효율 가속)
    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # 클래스 ID 확인
            cls = int(box.cls[0])

            # YOLO COCO 데이터셋에서 Class 0은 'person'(사람)임
            if cls == 0:
                # 신뢰도(Confidence Score)
                conf = float(box.conf[0])

                # 신뢰도가 50% 이상인 경우만 화면에 표시
                if conf > 0.5:
                    # 바운딩 박스 좌표 추출 (좌상단 x1, y1, 우하단 x2, y2)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # 화면에 사각형 및 텍스트 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Person: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 4. 결과 영상 출력
    cv2.imshow("Raspberry Pi 5 - Person Detection", frame)

    # 'q' 키 입력 시 루프 탈출 및 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()