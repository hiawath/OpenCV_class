"""
실시간 차선 감지 프로그램 v2
개선사항:
  1. 좌/우 차선 교차(소실점) 이후 연장선 제거
  2. 시간적 스무딩(EMA) + 미감지 시 이전 프레임 유지 → 점선 차선도 안정적으로 표시
사용법:
  python lane_detection.py [동영상 파일 경로]
  python lane_detection.py          ← 웹캠 사용
"""

import cv2
import numpy as np
import sys


# ────────────────────────────────────────────────
# 설정값
# ────────────────────────────────────────────────
SMOOTH_ALPHA    = 0.15   # EMA 계수 (작을수록 부드럽고 반응 느림)
MAX_MISS_FRAMES = 15     # 미감지 허용 프레임 수 (이후엔 선 숨김)


# ────────────────────────────────────────────────
# 시간적 스무딩 클래스
# ────────────────────────────────────────────────
class LaneSmoother:
    """
    EMA(지수 이동 평균)로 차선 파라미터(slope, intercept)를 안정화.
    점선 구간처럼 잠깐 미감지되어도 이전 값을 유지해 깜빡임 방지.
    """
    def __init__(self, alpha=SMOOTH_ALPHA, max_miss=MAX_MISS_FRAMES):
        self.alpha        = alpha
        self.max_miss     = max_miss
        self.left_params  = None   # (slope, intercept)
        self.right_params = None
        self.left_miss    = 0
        self.right_miss   = 0

    def update(self, left_raw, right_raw):
        # 왼쪽 차선
        if left_raw is not None:
            if self.left_params is None:
                self.left_params = left_raw
            else:
                s = self.alpha * left_raw[0] + (1 - self.alpha) * self.left_params[0]
                i = self.alpha * left_raw[1] + (1 - self.alpha) * self.left_params[1]
                self.left_params = (s, i)
            self.left_miss = 0
        else:
            self.left_miss += 1
            if self.left_miss > self.max_miss:
                self.left_params = None

        # 오른쪽 차선
        if right_raw is not None:
            if self.right_params is None:
                self.right_params = right_raw
            else:
                s = self.alpha * right_raw[0] + (1 - self.alpha) * self.right_params[0]
                i = self.alpha * right_raw[1] + (1 - self.alpha) * self.right_params[1]
                self.right_params = (s, i)
            self.right_miss = 0
        else:
            self.right_miss += 1
            if self.right_miss > self.max_miss:
                self.right_params = None

        return self.left_params, self.right_params


# ────────────────────────────────────────────────
# 소실점(두 직선의 교점) 계산
# ────────────────────────────────────────────────
def vanishing_point(left_params, right_params):
    """y = slope*x + intercept 두 직선의 교점 반환. 평행 또는 불가능이면 None."""
    if left_params is None or right_params is None:
        return None
    sl, bl = left_params
    sr, br = right_params
    denom = sl - sr
    if abs(denom) < 1e-6:
        return None
    x_vp = (br - bl) / denom
    y_vp = sl * x_vp + bl
    return int(x_vp), int(y_vp)


# ────────────────────────────────────────────────
# ROI
# ────────────────────────────────────────────────
def region_of_interest(frame):
    """하단 30% 사각 ROI 마스크"""
    height, width = frame.shape[:2]
    roi_top = int(height * 0.70)
    mask = np.zeros_like(frame)
    pts  = np.array([[(0, roi_top), (width, roi_top),
                      (width, height), (0, height)]], dtype=np.int32)
    fill = (255, 255, 255) if len(frame.shape) == 3 else 255
    cv2.fillPoly(mask, pts, fill)
    return cv2.bitwise_and(frame, mask), roi_top


# ────────────────────────────────────────────────
# 엣지 감지
# ────────────────────────────────────────────────
def detect_edges(frame):
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 40, 130)
    return edges


# ────────────────────────────────────────────────
# Hough 선 → 좌/우 raw 파라미터
# ────────────────────────────────────────────────
def extract_lane_params(frame, lines):
    """
    Hough 선들을 좌/우로 분류해 평균 (slope, intercept) 반환.
    중앙 x 위치도 함께 사용해 오분류 방지.
    """
    if lines is None:
        return None, None

    width = frame.shape[1]
    left_fit, right_fit = [], []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 == x1:
            continue
        slope     = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        if abs(slope) < 0.3:            # 수평 노이즈 제거
            continue
        mid_x = (x1 + x2) / 2
        if slope < 0 and mid_x < width * 0.6:
            left_fit.append((slope, intercept))
        elif slope > 0 and mid_x > width * 0.4:
            right_fit.append((slope, intercept))

    left_params  = tuple(np.mean(left_fit,  axis=0)) if left_fit  else None
    right_params = tuple(np.mean(right_fit, axis=0)) if right_fit else None
    return left_params, right_params


# ────────────────────────────────────────────────
# 좌표 생성 + 소실점 클리핑
# ────────────────────────────────────────────────
def params_to_line(frame, params, y_bottom, y_top):
    """(slope, intercept) → (x1,y1,x2,y2). y_bottom~y_top 범위로 클리핑."""
    if params is None:
        return None
    slope, intercept = params
    if abs(slope) < 1e-6:
        return None
    width    = frame.shape[1]
    x_bottom = int(np.clip(int((y_bottom - intercept) / slope), 0, width - 1))
    x_top    = int(np.clip(int((y_top    - intercept) / slope), 0, width - 1))
    return np.array([x_bottom, y_bottom, x_top, y_top])


def clip_lines_at_vanishing(frame, left_params, right_params, roi_top):
    """
    소실점의 y 좌표를 선분의 상단 한계로 사용.
    → 교차점 너머로 이어지는 연장선을 자동으로 제거.
    소실점이 ROI 밖이거나 없으면 roi_top을 상단 한계로 사용.
    """
    height   = frame.shape[0]
    y_bottom = height - 1

    vp = vanishing_point(left_params, right_params)

    if vp is not None:
        _, y_vp = vp
        # 소실점이 ROI 안(roi_top < y_vp < height)에 있을 때만 적용
        y_top = y_vp if roi_top < y_vp < height else roi_top
    else:
        y_top = roi_top

    left_line  = params_to_line(frame, left_params,  y_bottom, y_top)
    right_line = params_to_line(frame, right_params, y_bottom, y_top)
    return left_line, right_line, vp


# ────────────────────────────────────────────────
# 시각화
# ────────────────────────────────────────────────
def draw_lanes(frame, left_line, right_line):
    overlay = np.zeros_like(frame)

    if left_line is not None:
        x1, y1, x2, y2 = left_line
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 6)

    if right_line is not None:
        x1, y1, x2, y2 = right_line
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 6)

    if left_line is not None and right_line is not None:
        pts = np.array([
            [left_line[0],  left_line[1]],
            [left_line[2],  left_line[3]],
            [right_line[2], right_line[3]],
            [right_line[0], right_line[1]]
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 0, 60))

    return cv2.addWeighted(frame, 1.0, overlay, 0.85, 0)


def draw_vanishing_point(frame, vp):
    """소실점을 노란 원으로 표시"""
    if vp is None:
        return frame
    cv2.circle(frame, vp, 8,  (0, 255, 255), -1)
    cv2.circle(frame, vp, 13, (0, 200, 200),  2)
    cv2.putText(frame, "VP", (vp[0] + 15, vp[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return frame


def draw_roi_box(frame, roi_top):
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, roi_top), (width - 1, height - 1), (0, 255, 0), 2)
    cv2.putText(frame, "ROI", (10, roi_top - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def draw_info(frame, fps, left_miss, right_miss):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, 85), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    def color(m):
        if m == 0:                    return (0, 255, 0)      # 초록: 감지됨
        elif m <= MAX_MISS_FRAMES:    return (0, 200, 255)    # 주황: 보간 중
        else:                         return (0, 0, 255)      # 적색: 미감지

    def label(m):
        if m == 0:                    return "Detected"
        elif m <= MAX_MISS_FRAMES:    return f"Holding ({m}f)"
        else:                         return "Not Found"

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Left Lane:  {label(left_miss)}",  (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color(left_miss),  2)
    cv2.putText(frame, f"Right Lane: {label(right_miss)}", (10, 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color(right_miss), 2)
    return frame


# ────────────────────────────────────────────────
# 프레임 처리 파이프라인
# ────────────────────────────────────────────────
def process_frame(frame, smoother):
    height  = frame.shape[0]
    roi_top = int(height * 0.70)

    edges               = detect_edges(frame)
    roi_edges, _        = region_of_interest(edges)

    # 점선 대응: minLineLength ↓, maxLineGap ↑
    lines = cv2.HoughLinesP(
        roi_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=20,
        maxLineGap=150
    )

    left_raw, right_raw         = extract_lane_params(frame, lines)
    left_params, right_params   = smoother.update(left_raw, right_raw)
    left_line, right_line, vp   = clip_lines_at_vanishing(
                                      frame, left_params, right_params, roi_top)

    result = draw_lanes(frame, left_line, right_line)
    result = draw_vanishing_point(result, vp)
    result = draw_roi_box(result, roi_top)

    return result, smoother.left_miss, smoother.right_miss


# ────────────────────────────────────────────────
# 진입점
# ────────────────────────────────────────────────
def main():
    if len(sys.argv) > 1:
        source = sys.argv[1]
        cap    = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[오류] 파일을 열 수 없습니다: {source}")
            sys.exit(1)
        print(f"[정보] 동영상 파일: {source}")
    else:
        # 웹캠 캡처 시작
        VIDEO_PATH = './temp/drive_yy.mp4' # 입력 동영상 파일 경로
        cap = cv2.VideoCapture(VIDEO_PATH)

        if not cap.isOpened():
            print("[오류] 웹캠을 열 수 없습니다.")
            sys.exit(1)
        print("[정보] 웹캠 사용")

    cv2.namedWindow("Lane Detection v2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Lane Detection v2", 960, 540)

    smoother    = LaneSmoother()
    fps_timer   = cv2.getTickCount()
    fps         = 0.0
    frame_count = 0
    paused      = False

    print("[Q / ESC] 종료    [Space] 일시정지 / 재개")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord(' '):
            paused = not paused

        if paused:
            continue

        ret, frame = cap.read()
        if not ret:
            # 동영상 끝 → 처음부터 반복
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            smoother = LaneSmoother()
            continue

        frame_count += 1
        if frame_count % 30 == 0:
            elapsed   = (cv2.getTickCount() - fps_timer) / cv2.getTickFrequency()
            fps       = 30 / max(elapsed, 1e-6)
            fps_timer = cv2.getTickCount()

        result, lm, rm = process_frame(frame, smoother)
        result         = draw_info(result, fps, lm, rm)

        cv2.imshow("Lane Detection v2", result)

    cap.release()
    cv2.destroyAllWindows()
    print("[완료]")


if __name__ == "__main__":
    main()