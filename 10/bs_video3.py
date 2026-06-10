import cv2
import sys
import numpy as np
import os

# ─────────────────────────────────────────────
# 설정값 (필요에 따라 변경)
# ─────────────────────────────────────────────
VIDEO_PATH      = "./temp/background_subtraction.mp4"
MIN_AREA        = 800        # 이 픽셀 면적보다 작은 객체는 노이즈로 무시
MOG2_HISTORY    = 300        # 배경 학습에 사용할 프레임 수
MOG2_THRESHOLD  = 40         # 전경/배경 판별 임계값
DETECT_SHADOWS  = False      # 그림자 별도 표시 여부
BOX_COLOR       = (0, 0, 255)    # 바운딩 박스 색상 (BGR → 빨간색)
TEXT_COLOR      = (0, 255, 255)  # 텍스트 색상 (BGR → 노란색)
WINDOW_NAME     = "CCTV Vehicle Detection  [ESC: 종료 / r: 재시작]"

# ── 차선 제외 필터 파라미터 ──────────────────
BRIGHTNESS_THRESH = 200   # 이 값 이상인 픽셀은 헤드라이트/차선으로 판단해 마스크에서 제거
MIN_ASPECT_RATIO  = 0.25  # 바운딩박스 w/h 비율이 이보다 작으면 차선(세로로 긴 형태) 제외
MAX_ASPECT_RATIO  = 8.0   # 비율이 이보다 크면 가로로 너무 긴 차선 형태로 제외
MIN_SOLIDITY      = 0.3   # 면적/볼록껍질 비율 — 너무 파편화된 윤곽(차선)은 제외
MIN_BOX_HEIGHT    = 20    # 바운딩박스 높이가 이보다 낮으면 차선 가능성 높으므로 제외
# ─────────────────────────────────────────────


def create_demo_video(path: str, duration_sec: int = 12, fps: int = 25) -> None:
    """
    실제 영상 파일이 없을 때 테스트용 합성 영상을 생성합니다.
    헤드라이트로 밝아지는 차선 + 이동하는 차량 포함.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))

    total_frames = duration_sec * fps
    for i in range(total_frames):
        # 배경: 어두운 도로
        bg = np.full((h, w, 3), 45, dtype=np.uint8)

        # 차선 (고정 흰색 선) — 헤드라이트가 비추면 밝아지는 효과 시뮬레이션
        lane_brightness = 160 + int(60 * np.sin(i / fps * np.pi))  # 밝기 변동
        lane_color = (lane_brightness, lane_brightness, lane_brightness)
        cv2.line(bg, (w // 2 - 80, 0), (w // 2 - 80, h), lane_color, 4)
        cv2.line(bg, (w // 2 + 80, 0), (w // 2 + 80, h), lane_color, 4)
        # 점선 차선
        for seg in range(0, h, 60):
            cv2.line(bg, (w // 2, seg), (w // 2, seg + 30), lane_color, 3)

        # 움직이는 차량 1 (왼→오, 부피 있는 직사각형)
        x1 = int((i / total_frames) * (w + 160)) - 80
        cv2.rectangle(bg, (x1, 180), (x1 + 110, 280), (150, 90, 50), -1)   # 차체
        cv2.rectangle(bg, (x1 + 10, 155), (x1 + 100, 182), (120, 70, 40), -1)  # 지붕
        # 헤드라이트 (밝은 원)
        cv2.circle(bg, (x1 + 100, 210), 10, (240, 240, 200), -1)
        cv2.circle(bg, (x1 + 100, 250), 10, (240, 240, 200), -1)

        # 움직이는 차량 2 (오→왼)
        x2 = w - int((i / total_frames) * (w + 160)) + 40
        cv2.rectangle(bg, (x2, 320), (x2 + 100, 420), (50, 100, 160), -1)
        cv2.rectangle(bg, (x2 + 5, 295), (x2 + 95, 322), (40, 80, 130), -1)

        # 노이즈
        noise = np.random.randint(-6, 6, bg.shape, dtype=np.int16)
        frame = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        out.write(frame)

    out.release()
    print(f"[INFO] 테스트 영상 생성 완료: {path}")


def build_subtractor() -> cv2.BackgroundSubtractorMOG2:
    return cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_THRESHOLD,
        detectShadows=DETECT_SHADOWS,
    )


def suppress_headlight_lane(fg_mask: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """
    헤드라이트·차선으로 인한 오검출을 마스크 단계에서 억제합니다.

    전략 1 — 밝기 억제:
      그레이스케일 변환 후 BRIGHTNESS_THRESH 이상인 픽셀(=헤드라이트/밝은 차선)을
      전경 마스크에서 제거합니다.
      단, 헤드라이트를 달고 있는 차량 본체까지 지워지지 않도록
      팽창(dilate)으로 복원 여유를 둡니다.

    전략 2 — Canny 엣지 마스킹:
      차선은 날카로운 직선 엣지로 나타납니다.
      엣지 밀도가 높은 영역(=차선 후보)을 전경 마스크에서 빼줍니다.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ① 과노출(헤드라이트) 영역 마스크
    _, bright_mask = cv2.threshold(gray, BRIGHTNESS_THRESH, 255, cv2.THRESH_BINARY)
    # 밝은 영역 주변 차량 본체는 살리기 위해 팽창
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    bright_expanded = cv2.dilate(bright_mask, k_dilate, iterations=1)
    # 전경 마스크에서 과노출 영역만 제거 (팽창 영역은 유지)
    fg_mask = cv2.bitwise_and(fg_mask, cv2.bitwise_not(bright_mask))
    # 팽창 영역은 다시 OR로 복원 (차량 본체 보존)
    fg_mask = cv2.bitwise_or(fg_mask, cv2.bitwise_and(fg_mask, bright_expanded))

    # ② 직선 엣지(차선) 억제
    edges = cv2.Canny(gray, 50, 150)
    # 엣지를 수직 방향으로 팽창 → 차선 형태 강조
    k_lane = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    lane_mask = cv2.dilate(edges, k_lane, iterations=2)
    # 엣지 밀도 계산: 주변에 엣지가 몰려 있는 곳 = 차선
    edge_density = cv2.GaussianBlur(lane_mask.astype(np.float32), (21, 21), 0)
    _, lane_region = cv2.threshold(edge_density, 30, 255, cv2.THRESH_BINARY)
    lane_region = lane_region.astype(np.uint8)
    fg_mask = cv2.bitwise_and(fg_mask, cv2.bitwise_not(lane_region))

    return fg_mask


def clean_mask(fg_mask: np.ndarray) -> np.ndarray:
    """형태학적 연산으로 마스크 정제 (OPEN → CLOSE → DILATE)."""
    k_small  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  k_small)
    mask = cv2.morphologyEx(mask,    cv2.MORPH_CLOSE, k_medium)
    mask = cv2.dilate(mask, k_small, iterations=1)
    return mask


def is_vehicle_contour(cnt) -> bool:
    """
    윤곽선이 차량인지 차선/노이즈인지 판별합니다.

    차량의 특성:
      - 면적이 일정 크기 이상
      - 바운딩박스가 너무 가늘지 않음 (종횡비 범위 내)
      - 바운딩박스 높이가 충분히 있음 (차선은 매우 얇음)
      - 솔리디티(면적/볼록껍질 비율)가 어느 정도 있음

    차선의 특성:
      - 매우 좁고 긴 형태 → 종횡비 필터로 제거
      - 높이가 낮음 → MIN_BOX_HEIGHT 필터로 제거
      - 파편화된 윤곽 → 솔리디티 필터로 제거
    """
    area = cv2.contourArea(cnt)
    if area < MIN_AREA:
        return False

    x, y, w, h = cv2.boundingRect(cnt)

    # 높이 필터 (차선은 얇음)
    if h < MIN_BOX_HEIGHT:
        return False

    # 종횡비 필터 (차선은 w/h가 매우 작거나 큼)
    aspect = w / max(h, 1)
    if not (MIN_ASPECT_RATIO <= aspect <= MAX_ASPECT_RATIO):
        return False

    # 솔리디티 필터 (파편화된 윤곽 제거)
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    if hull_area == 0:
        return False
    solidity = area / hull_area
    if solidity < MIN_SOLIDITY:
        return False

    return True


def draw_detections(frame: np.ndarray, contours) -> tuple[np.ndarray, int]:
    """차량으로 판별된 윤곽선에만 빨간 바운딩 박스를 그립니다."""
    canvas = frame.copy()
    count = 0

    for cnt in contours:
        if not is_vehicle_contour(cnt):
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # 빨간색 바운딩 박스
        cv2.rectangle(canvas, (x, y), (x + w, y + h), BOX_COLOR, 2)

        # 레이블
        label = f"Vehicle {count + 1}  [{w}x{h}]"
        label_y = y + 18 if y < 20 else y - 6
        cv2.putText(canvas, label, (x + 4, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
        count += 1

    return canvas, count


def draw_overlay(canvas: np.ndarray, obj_count: int, frame_no: int) -> np.ndarray:
    """좌측 상단 반투명 상태 오버레이."""
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (270, 55), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
    cv2.putText(canvas, f"Frame  : {frame_no:05d}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Vehicles: {obj_count}", (8, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 80, 255) if obj_count else (180, 180, 180), 1, cv2.LINE_AA)
    return canvas


def run(video_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 영상을 열 수 없습니다: {video_path}")
        sys.exit(1)

    fps      = cap.get(cv2.CAP_PROP_FPS) or 25
    delay    = max(1, int(1000 / fps))
    fgbg     = build_subtractor()
    frame_no = 0

    print(f"[INFO] 재생 시작 — FPS: {fps:.1f}  |  ESC: 종료  |  r: 처음부터 재시작")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] 영상 종료 — 처음부터 다시 재생합니다.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            fgbg     = build_subtractor()
            frame_no = 0
            continue

        # ① 배경 분리
        fg_mask = fgbg.apply(frame)

        # ② 헤드라이트·차선 억제 (핵심 추가)
        fg_mask = suppress_headlight_lane(fg_mask, frame)

        # ③ 형태학적 정제
        fg_mask = clean_mask(fg_mask)

        # ④ 윤곽선 검출 → 차량만 바운딩 박스
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        canvas, obj_count = draw_detections(frame, contours)
        canvas = draw_overlay(canvas, obj_count, frame_no)

        # ⑤ 디버그 뷰: [결과 | 정제된 마스크]
        mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([canvas, mask_bgr])
        combined = cv2.resize(combined, (0, 0), fx=0.85, fy=0.85)

        cv2.imshow(WINDOW_NAME, combined)
        frame_no += 1

        key = cv2.waitKey(delay) & 0xFF
        if key == 27:
            print("[INFO] 사용자가 종료했습니다.")
            break
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            fgbg     = build_subtractor()
            frame_no = 0
            print("[INFO] 영상을 처음부터 재시작합니다.")

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
if __name__ == "__main__":
    video_path = VIDEO_PATH
    if len(sys.argv) > 1:
        video_path = sys.argv[1]

    if not os.path.exists(video_path):
        print(f"[WARNING] 영상 파일 없음: {video_path}")
        print("[INFO] 테스트용 합성 영상을 자동으로 생성합니다...")
        create_demo_video(video_path)

    run(video_path)
