"""
특정 색상 객체 추적 - Color Space & Masking
BGR → HSV 변환 및 cv2.inRange 마스크 기반 중심점 좌표 출력
"""

import platform
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 한글 폰트 (macOS) ───────────────────────
if platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
elif platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
else:
    plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False


# ──────────────────────────────────────────
# 1. 색상 프리셋 정의 (HSV 범위)
# ──────────────────────────────────────────
# OpenCV HSV: H(0-179)  S(0-255)  V(0-255)
COLOR_PRESETS = {
    "blue":   {"lower": (100, 100,  50), "upper": (130, 255, 255), "label": "파란색"},
    "red_lo": {"lower": (  0, 120,  80), "upper": ( 10, 255, 255), "label": "빨간색(하)"},
    "red_hi": {"lower": (170, 120,  80), "upper": (179, 255, 255), "label": "빨간색(상)"},
    "yellow": {"lower": ( 22, 120, 100), "upper": ( 38, 255, 255), "label": "노란색"},
    "green":  {"lower": ( 40,  80,  60), "upper": ( 80, 255, 255), "label": "초록색"},
}


# ──────────────────────────────────────────
# 2. 테스트용 합성 이미지 생성
# ──────────────────────────────────────────
def create_test_frame(img_size: tuple = (480, 640)) -> np.ndarray:
    """다양한 색상 공이 흩어진 BGR 합성 이미지를 생성합니다."""
    h, w = img_size
    frame = np.full((h, w, 3), (180, 180, 180), dtype=np.uint8)

    # 배경 질감
    noise = np.random.randint(-15, 15, (h, w, 3), dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    rng = np.random.default_rng(7)
    balls = [
        # 파란 공 (주 타겟)
        {"center": (200, 150), "r": 45, "bgr": (200, 80,  40)},
        {"center": (450, 300), "r": 30, "bgr": (220, 100, 60)},
        # 방해 색상
        {"center": (100, 320), "r": 35, "bgr": (40,  200, 40)},   # 초록
        {"center": (350, 80),  "r": 28, "bgr": (40,  40,  220)},  # 빨강
        {"center": (520, 380), "r": 40, "bgr": (30,  200, 220)},  # 노랑
        {"center": (280, 400), "r": 22, "bgr": (180, 50,  180)},  # 보라
    ]

    for b in balls:
        cx, cy = b["center"]
        r = b["r"]
        bgr = b["bgr"]

        # 공 본체
        cv2.circle(frame, (cx, cy), r, bgr, -1)
        # 하이라이트 효과
        hl = tuple(min(255, c + 70) for c in bgr)
        cv2.circle(frame, (cx - r//4, cy - r//4), r//3, hl, -1)
        # 테두리
        shadow = tuple(max(0, c - 60) for c in bgr)
        cv2.circle(frame, (cx, cy), r, shadow, 2)

    return frame


# ──────────────────────────────────────────
# 3. 단일 색상 마스킹 + 중심점 탐지
# ──────────────────────────────────────────
def detect_color_object(
    bgr_frame: np.ndarray,
    lower_hsv: tuple,
    upper_hsv: tuple,
    min_area: float = 500,
) -> tuple[np.ndarray, list[dict]]:
    """
    BGR 이미지에서 특정 HSV 범위의 색상 객체를 탐지하고 중심점을 반환합니다.

    Parameters
    ----------
    bgr_frame : BGR 입력 이미지
    lower_hsv : HSV 범위 하한 (H, S, V)
    upper_hsv : HSV 범위 상한 (H, S, V)
    min_area  : 잡음 제거 최소 면적 (px²)

    Returns
    -------
    mask      : 이진 마스크 이미지
    objects   : 탐지된 객체 정보 리스트 (cx, cy, area, contour)
    """
    # ① BGR → HSV 변환
    hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)

    # ② inRange 마스크 생성
    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)
    mask = cv2.inRange(hsv_frame, lower, upper)

    # ③ 모폴로지 연산 (작은 노이즈 제거 + 빈 구멍 메우기)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # ④ 윤곽선 검출
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    objects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # ⑤ 모멘트로 중심점 계산
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        objects.append({
            "cx": cx,
            "cy": cy,
            "area": area,
            "contour": cnt,
            "bbox": cv2.boundingRect(cnt),
        })

    # 면적 내림차순 정렬
    objects.sort(key=lambda o: o["area"], reverse=True)
    return mask, objects


# ──────────────────────────────────────────
# 4. 빨간색 처리 (Hue 0 경계 양쪽 분리)
# ──────────────────────────────────────────
def detect_red(bgr_frame: np.ndarray, min_area: float = 500):
    """빨간색은 HSV Hue 0 경계에 걸쳐 있어 두 구간으로 나눠 OR 합산합니다."""
    lo_preset = COLOR_PRESETS["red_lo"]
    hi_preset = COLOR_PRESETS["red_hi"]

    mask1, _ = detect_color_object(bgr_frame,
                                    lo_preset["lower"], lo_preset["upper"], 0)
    mask2, _ = detect_color_object(bgr_frame,
                                    hi_preset["lower"], hi_preset["upper"], 0)
    combined_mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(
        combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    objects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        objects.append({
            "cx": int(M["m10"] / M["m00"]),
            "cy": int(M["m01"] / M["m00"]),
            "area": area,
            "contour": cnt,
            "bbox": cv2.boundingRect(cnt),
        })
    objects.sort(key=lambda o: o["area"], reverse=True)
    return combined_mask, objects


# ──────────────────────────────────────────
# 5. 결과 시각화
# ──────────────────────────────────────────
def draw_detections(bgr_frame: np.ndarray, objects: list[dict]) -> np.ndarray:
    """탐지 결과를 이미지에 그립니다."""
    result = bgr_frame.copy()
    for i, obj in enumerate(objects, start=1):
        cx, cy = obj["cx"], obj["cy"]
        bx, by, bw, bh = obj["bbox"]

        # 바운딩 박스
        cv2.rectangle(result, (bx, by), (bx + bw, by + bh), (255, 150, 0), 2)

        # 중심점 표시
        cv2.circle(result, (cx, cy), 6, (0, 0, 255), -1)
        cv2.drawMarker(result, (cx, cy), (0, 0, 255),
                       cv2.MARKER_CROSS, 20, 2)

        # 좌표 텍스트
        label = f"#{i} cx={cx}, cy={cy}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (bx, by - th - 8), (bx + tw + 4, by), (255, 150, 0), -1)
        cv2.putText(result, label, (bx + 2, by - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return result


def visualize_pipeline(bgr_frame: np.ndarray,
                        mask: np.ndarray,
                        objects: list[dict],
                        color_label: str) -> None:
    """원본 / HSV / 마스크 / 결과를 2×2 레이아웃으로 출력합니다."""
    hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    h_channel = hsv_frame[:, :, 0]       # H 채널만 추출
    result = draw_detections(bgr_frame, objects)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"색상 추적 파이프라인 — {color_label}", fontsize=14, fontweight="bold")

    axes[0, 0].imshow(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("① 원본 BGR 이미지")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(h_channel, cmap="hsv")
    axes[0, 1].set_title("② HSV 변환 (H 채널)")
    axes[0, 1].axis("off")
    fig.colorbar(axes[0, 1].images[0], ax=axes[0, 1], fraction=0.046)

    axes[1, 0].imshow(mask, cmap="gray")
    axes[1, 0].set_title("③ cv2.inRange 마스크")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"④ 탐지 결과 ({len(objects)}개 객체)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig("color_track_result.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("결과 저장: color_track_result.png")


# ──────────────────────────────────────────
# 6. 콘솔 출력
# ──────────────────────────────────────────
def print_results(objects: list[dict], color_label: str) -> None:
    print("\n" + "=" * 50)
    print(f"  색상 추적 결과 — {color_label}")
    print("=" * 50)
    print(f"  탐지된 객체 수: {len(objects)}개")
    print("-" * 50)
    if objects:
        print(f"  {'#':>3}  {'중심 X':>8}  {'중심 Y':>8}  {'면적(px²)':>12}")
        print("-" * 50)
        for i, o in enumerate(objects, 1):
            print(f"  {i:>3}  {o['cx']:>8}  {o['cy']:>8}  {int(o['area']):>12,}")
    else:
        print("  탐지된 객체가 없습니다.")
    print("=" * 50)


# ──────────────────────────────────────────
# 7. 실시간 웹캠 추적 (선택)
# ──────────────────────────────────────────
def run_webcam_tracker(color: str = "blue") -> None:
    """
    웹캠 영상에서 실시간으로 색상 객체를 추적합니다.
    종료: q 키
    """
    preset = COLOR_PRESETS[color]
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    print(f"[{preset['label']}] 실시간 추적 시작 — q 키로 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask, objects = detect_color_object(
            frame,
            preset["lower"],
            preset["upper"],
        )
        result = draw_detections(frame, objects)

        cv2.imshow("원본", frame)
        cv2.imshow("마스크", mask)
        cv2.imshow("추적 결과", result)

        if objects:
            o = objects[0]
            print(f"  중심점: cx={o['cx']}, cy={o['cy']}  면적={int(o['area'])}px²")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ──────────────────────────────────────────
# 8. 메인
# ──────────────────────────────────────────
def main(image_path: str | None = None,
         color: str = "blue",
         use_webcam: bool = False) -> None:
    """
    Parameters
    ----------
    image_path : 실제 이미지 경로 (None이면 합성 이미지 사용)
    color      : 추적할 색상 키 ("blue" | "yellow" | "green" | "red")
    use_webcam : True이면 웹캠 실시간 추적 모드
    """
    if use_webcam:
        run_webcam_tracker(color)
        return

    # 이미지 로드
    if image_path:
        bgr_frame = cv2.imread(image_path)
        if bgr_frame is None:
            raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")
    else:
        print("합성 테스트 이미지를 생성합니다...")
        bgr_frame = create_test_frame()

    # 색상별 탐지
    if color == "red":
        mask, objects = detect_red(bgr_frame)
        label = "빨간색"
    else:
        preset = COLOR_PRESETS.get(color, COLOR_PRESETS["blue"])
        mask, objects = detect_color_object(
            bgr_frame, preset["lower"], preset["upper"]
        )
        label = preset["label"]

    print_results(objects, label)
    visualize_pipeline(bgr_frame, mask, objects, label)

if __name__ == "__main__":
    # ── 사용 예시 ───────────────────────────
    # 합성 이미지로 파란색 추적
    main(color="red_lo", use_webcam=True)

    # 실제 이미지 사용: main("tennis.jpg", color="blue")
    # 웹캠 실시간 추적: main(use_webcam=True, color="blue")
