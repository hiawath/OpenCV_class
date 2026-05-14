"""
실시간 차선 감지 프로그램
- ROI: 화면 하단 30%
- ROI 영역: 녹색 박스 표시
- 감지된 차선: 적색 선으로 시각화
사용법: python lane_detection.py [동영상 파일 경로]
         python lane_detection.py  (웹캠 사용)
"""

import cv2
import numpy as np
import sys


def region_of_interest(frame):
    """하단 30% ROI 마스크 생성"""
    height, width = frame.shape[:2]
    roi_top = int(height * 0.70)  # 상단 70% 제외 → 하단 30%

    mask = np.zeros_like(frame)
    # 하단 30% 사각형 영역
    roi_vertices = np.array([[
        (0, roi_top),
        (width, roi_top),
        (width, height),
        (0, height)
    ]], dtype=np.int32)

    if len(frame.shape) == 3:
        cv2.fillPoly(mask, roi_vertices, (255, 255, 255))
    else:
        cv2.fillPoly(mask, roi_vertices, 255)

    return cv2.bitwise_and(frame, mask), roi_top


def detect_edges(frame):
    """엣지 감지 파이프라인"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 가우시안 블러로 노이즈 제거
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny 엣지 감지
    edges = cv2.Canny(blurred, 50, 150)
    return edges


def average_lines(frame, lines):
    """감지된 선들을 좌/우 차선으로 분류 및 평균화"""
    height, width = frame.shape[:2]

    left_fit = []   # 기울기 음수 → 왼쪽 차선
    right_fit = []  # 기울기 양수 → 오른쪽 차선

    if lines is None:
        return None, None

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 == x1:
            continue  # 수직선 제외
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # 너무 수평한 선 제외 (노이즈)
        if abs(slope) < 0.3:
            continue

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_line = make_coordinates(frame, np.mean(left_fit, axis=0)) if left_fit else None
    right_line = make_coordinates(frame, np.mean(right_fit, axis=0)) if right_fit else None

    return left_line, right_line


def make_coordinates(frame, line_params):
    """기울기·절편으로부터 화면 좌표 계산"""
    height, width = frame.shape[:2]
    roi_top = int(height * 0.70)

    slope, intercept = line_params
    y1 = height
    y2 = roi_top  # ROI 상단까지만

    if slope == 0:
        return None

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    # 화면 밖으로 나가는 좌표 클리핑
    x1 = np.clip(x1, 0, width)
    x2 = np.clip(x2, 0, width)

    return np.array([x1, y1, x2, y2])


def draw_lanes(frame, left_line, right_line):
    """적색 차선 및 반투명 채우기 그리기"""
    height, width = frame.shape[:2]
    line_image = np.zeros_like(frame)

    if left_line is not None:
        x1, y1, x2, y2 = left_line
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 6)

    if right_line is not None:
        x1, y1, x2, y2 = right_line
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 6)

    # 두 차선 사이 반투명 영역 (선택)
    if left_line is not None and right_line is not None:
        pts = np.array([
            [left_line[0],  left_line[1]],
            [left_line[2],  left_line[3]],
            [right_line[2], right_line[3]],
            [right_line[0], right_line[1]]
        ], dtype=np.int32)
        cv2.fillPoly(line_image, [pts], (0, 0, 80))

    return cv2.addWeighted(frame, 1.0, line_image, 0.9, 0)


def draw_roi_box(frame, roi_top):
    """ROI 영역 녹색 박스 표시"""
    height, width = frame.shape[:2]
    cv2.rectangle(frame,
                  (0, roi_top),
                  (width - 1, height - 1),
                  (0, 255, 0), 2)
    cv2.putText(frame, "ROI", (10, roi_top - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def draw_info(frame, fps, left_detected, right_detected):
    """화면 상단 정보 오버레이"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    left_color  = (0, 255, 0) if left_detected  else (0, 0, 255)
    right_color = (0, 255, 0) if right_detected else (0, 0, 255)

    left_txt  = "Left Lane:  Detected" if left_detected  else "Left Lane:  Not Found"
    right_txt = "Right Lane: Detected" if right_detected else "Right Lane: Not Found"

    cv2.putText(frame, left_txt,  (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, left_color,  2)
    cv2.putText(frame, right_txt, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, right_color, 2)
    return frame


def process_frame(frame):
    """프레임 단위 차선 감지 파이프라인"""
    height = frame.shape[0]
    roi_top = int(height * 0.70)

    # 1. 엣지 감지
    edges = detect_edges(frame)

    # 2. ROI 마스킹
    roi_edges, _ = region_of_interest(edges)

    # 3. 허프 변환으로 선 감지
    lines = cv2.HoughLinesP(
        roi_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=60,
        maxLineGap=100
    )

    # 4. 좌/우 차선 평균화
    left_line, right_line = average_lines(frame, lines)

    # 5. 차선 그리기
    result = draw_lanes(frame, left_line, right_line)

    # 6. ROI 녹색 박스
    result = draw_roi_box(result, roi_top)

    return result, left_line is not None, right_line is not None


def main():
    # 입력 소스 결정
    if len(sys.argv) > 1:
        source = sys.argv[1]
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[오류] 파일을 열 수 없습니다: {source}")
            sys.exit(1)
        print(f"[정보] 동영상 파일 재생: {source}")
    else:
            # 웹캠 캡처 시작
        VIDEO_PATH = './temp/drive_yy.mp4' # 입력 동영상 파일 경로
        cap = cv2.VideoCapture(VIDEO_PATH)
       
        if not cap.isOpened():
            print("[오류] 웹캠을 열 수 없습니다.")
            sys.exit(1)
        print("[정보] 웹캠 사용 중...")

    # 창 설정
    cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Lane Detection", 960, 540)

    # FPS 측정용
    fps_timer = cv2.getTickCount()
    fps = 0.0
    frame_count = 0

    print("[안내] 종료하려면 'q' 또는 ESC를 누르세요.")
    print("[안내] 일시정지/재개: Space")

    paused = False

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):   # q 또는 ESC → 종료
            break
        if key == ord(' '):         # Space → 일시정지 토글
            paused = not paused

        if paused:
            cv2.putText(cv2.getWindowImageRect("Lane Detection") and
                        np.zeros((50, 200, 3), dtype=np.uint8) or np.zeros((1, 1, 3), dtype=np.uint8),
                        "PAUSED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            continue

        ret, frame = cap.read()
        if not ret:
            print("[정보] 동영상 재생 완료 또는 프레임 읽기 실패.")
            break

        # FPS 계산 (30프레임마다 갱신)
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = (cv2.getTickCount() - fps_timer) / cv2.getTickFrequency()
            fps = 30 / elapsed
            fps_timer = cv2.getTickCount()

        # 차선 감지 처리
        result, left_ok, right_ok = process_frame(frame)

        # 정보 오버레이
        result = draw_info(result, fps, left_ok, right_ok)

        cv2.imshow("Lane Detection", result)

    cap.release()
    cv2.destroyAllWindows()
    print("[완료] 프로그램이 종료되었습니다.")


if __name__ == "__main__":
    main()