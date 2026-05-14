import cv2
import numpy as np

def process_lane_detection(frame):
    """
    입력된 프레임에서 흰색 계열의 차선을 감지하고 시각화합니다.
    
    Args:
        frame (np.array): OpenCV로 읽어온 BGR 형식의 이미지 프레임.
        
    Returns:
        np.array: 차선이 마스킹되고 원본 프레임에 오버레이된 결과 이미지.
    """
    # 1. 전처리: BGR -> HSV 색 공간 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. 흰색 차선 마스킹 (HSV 기반)
    # 흰색 영역을 포괄하는 범위 설정
    lower_white = np.array([0, 50, 150])
    upper_white = np.array([180, 255, 255])
    
    # 마스크 생성 (전체 프레임 크기)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # 3. 노이즈 제거 및 차선 강조
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    # 4. ROI (Region of Interest) 마스킹 (오류 수정 지점)
    height, width = frame.shape[:2]
    # ROI 정의 (중앙 하단)
    y1, y2 = int(0.6 * height), height # Y축: 70% 높이부터 바닥까지
    x1, x2 = int(0.2 * width), int(0.8 * width)
    
    # 1. 전체 프레임 크기의 검정색 바탕 (검출되지 않을 영역)을 준비합니다.
    # 이 영역은 검은색(0)이어야 합니다.
    roi_mask = np.zeros_like(mask_closed)
    
    # 2. ROI 영역에만 흰색(255)을 채웁니다. (여기서 크기 불일치 오류가 해결됨)
    cv2.rectangle(roi_mask, (x1, y1), (x2, y2), 255, -1)
    
    # 3. 최종 ROI 마스크: [전체 마스크] AND [ROI 마스크]
    mask_roi = cv2.bitwise_and(mask_closed, roi_mask)
    
    # 5. 엣지 검출 및 허프 변환을 통한 선 검출
    edges = cv2.Canny(mask_roi, 50, 150)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30, minLineLength=30, maxLineGap=10)
    
    # 결과 시각화를 위해 원본 프레임 복사
    final_output = frame.copy()
    
    # 6. 검출된 차선을 빨간색으로 그리기
    if lines is not None:
        for line in lines:
            x_start, y_start, x_end, y_end = line[0]
            cv2.line(final_output, (x_start, y_start), (x_end, y_end), (0, 0, 255), 3)
            
    # ROI 영역을 녹색 사각형으로 표시 (두께 2)
    cv2.rectangle(final_output, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return final_output

# ======================================================================
# 메인 실행 루프
# ======================================================================
if __name__ == "__main__":
    # 웹캠 캡처 시작
    VIDEO_PATH = './temp/drive_yy.mp4' # 입력 동영상 파일 경로
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("오류: 동영상을 열 수 없습니다. 파일 경로를 확인하세요.")
        exit()
        
    print("차선 감지를 시작합니다. 'q' 키를 눌러 종료하세요.")
    
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("프레임 수신 실패. 종료합니다.")
            break
        
        # 차선 감지 함수 호출
        processed_frame = process_lane_detection(frame)
        
        # 결과 표시
        cv2.imshow('Lane Detection (Press Q to Quit)', processed_frame)
        
        # 키 입력 확인
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    # 종료 및 리소스 해제
    cap.release()
    cv2.destroyAllWindows()