import cv2
import sys

# 1. 실제 MP4 동영상 파일 로드
video_path = "./images/bs.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"오류: 동영상 파일({video_path})을 열 수 없습니다.")
    sys.exit()

# 동영상의 FPS(초당 프레임 수)를 가져와 지연 시간 계산 (원본 속도 유지용)
fps = cap.get(cv2.CAP_PROP_FPS)
delay = max(1, int(1000 / fps))

# 배경 분리 객체 생성 (이전 실습 연계)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

while True:
    retval, frame = cap.read()
    
    # 동영상이 끝나면 루프 탈출 (NoneType 에러 방지)
    if not retval:
        print("동영상 상영이 완료되었습니다.")
        break
        
    # 전경 마스크 생성 및 모폴로지 노이즈 제거 (모래채 연산)
    fg_mask = fgbg.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    
    # 흰색 객체의 외곽선(Contours) 검출
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # [중요] 원근 변환이나 드로잉 시 원본 프레임 훼손을 막기 위해 복사본 생성
    drawing_frame = frame.copy()
    
    for contour in contours:
        # 너무 작은 크기의 노이즈(예: 먼지, 불빛 흔들림)는 거르고 큰 객체만 선별
        if cv2.contourArea(contour) < 500:
            continue
            
        # 외곽선을 수직으로 감싸는 최소 크기의 사각형 좌표(x, y, 너비, 높이) 추출
        # x, y: 사각형의 좌측 상단 꼭지점 좌표
        x, y, w, h = cv2.boundingRect(contour)
        
        # 2. 원본 이미지에 빨간색 바운딩 박스 그리기
        # cv2.rectangle(이미지, 좌측상단좌표, 우측하단좌표, 색상(BGR), 두께)
        # 색상: (0, 0, 255) -> Red, 두께: 2픽셀
        cv2.rectangle(drawing_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # 3. 박스 상단에 상태 표시 텍스트 쓰기
        # cv2.putText(이미지, 텍스트, 시작좌표, 폰트종류, 글자크기, 색상, 두께, 라인타입)
        # y - 10 위치에 배치하여 사각형 박스 바로 위에 글자가 뜨도록 유도
        text_position = (x, y - 10)
        cv2.putText(drawing_frame, "Target Detected", text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        
    # 4. 모니터링 창 출력
    cv2.imshow("CCTV Box Detection", drawing_frame)
    
    # ESC 키 누르면 중단
    if cv2.waitKey(delay) & 0xFF == 27:
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()