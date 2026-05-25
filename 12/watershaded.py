import cv2
import numpy as np
import os
import sys
import random

# ─────────────────────────────────────────────
# 설정값
# ─────────────────────────────────────────────
IMAGE_PATH       = "./images/pills.jpg"   # 입력 이미지 경로
OUT_PATH         = "./temp/watershed_result.jpg"
DIST_THRESH      = 0.45   # 거리 변환 임계값 비율 (0.3~0.6: 낮을수록 더 많이 분리)
MORPH_ITER       = 2      # 팽창 반복 횟수 (sure_bg 확장 강도)
MIN_OBJECT_AREA  = 300    # 이 픽셀 면적보다 작은 객체는 무시 (노이즈 제거)
SHOW_STEPS       = True   # 중간 단계 창 출력 여부
# ─────────────────────────────────────────────


def make_test_image(path: str) -> None:
    """
    실제 이미지가 없을 때 서로 붙어있는 원형 객체(알약/세포) 합성 이미지를 생성합니다.
    일부러 겹치거나 붙어있도록 배치하여 Watershed 효과를 명확히 보여줍니다.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h, w = 480, 640
    img = np.ones((h, w, 3), dtype=np.uint8) * 30  # 어두운 배경

    random.seed(42)
    np.random.seed(42)

    # 알약/세포 정의: (cx, cy, radius, color_bgr)
    objects = [
        # 독립 객체
        (80,  80,  38, (180, 210, 240)),
        (580, 60,  42, (240, 190, 170)),
        (560, 400, 35, (170, 240, 200)),
        (80,  400, 40, (240, 220, 160)),
        # 두 개가 붙어있는 쌍
        (200, 150, 45, (200, 170, 240)),
        (275, 165, 45, (200, 170, 240)),
        # 세 개가 붙어있는 군집
        (380, 120, 40, (170, 220, 250)),
        (450, 100, 38, (170, 220, 250)),
        (420, 175, 42, (170, 220, 250)),
        # 중앙 밀집 군집
        (260, 300, 50, (230, 200, 180)),
        (340, 290, 48, (230, 200, 180)),
        (300, 370, 46, (230, 200, 180)),
        (210, 360, 44, (230, 200, 180)),
        (390, 355, 47, (230, 200, 180)),
        # 소형 객체
        (480, 280, 28, (200, 240, 210)),
        (510, 340, 30, (200, 240, 210)),
        (150, 250, 32, (240, 200, 220)),
    ]

    for cx, cy, r, color in objects:
        # 객체 본체
        cv2.circle(img, (cx, cy), r, color, -1)
        # 테두리 (질감 표현)
        darker = tuple(max(0, c - 40) for c in color)
        cv2.circle(img, (cx, cy), r, darker, 2)
        # 하이라이트 (입체감)
        highlight_x = cx - r // 4
        highlight_y = cy - r // 4
        cv2.circle(img, (highlight_x, highlight_y), r // 5,
                   (255, 255, 255), -1)
        cv2.circle(img, (highlight_x, highlight_y), r // 5,
                   (255, 255, 255), -1)

    # 약간의 가우시안 블러로 자연스럽게
    img = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.imwrite(path, img)
    print(f"[INFO] 테스트 이미지 생성 완료: {path}")


# ─────────────────────────────────────────────
# Watershed 파이프라인
# ─────────────────────────────────────────────

def step1_preprocess(img: np.ndarray) -> np.ndarray:
    """
    [Step 1] 전처리: 그레이스케일 → 가우시안 블러 → 이진화(Otsu)

    Otsu 이진화: 히스토그램을 분석해 최적의 임계값을 자동으로 결정합니다.
    객체(흰색)와 배경(검은색)을 분리하는 바이너리 마스크를 만듭니다.
    """
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def step2_sure_bg(binary: np.ndarray) -> np.ndarray:
    """
    [Step 2] 확실한 배경(Sure Background) 추출

    팽창(Dilate)으로 객체 영역을 살짝 부풀립니다.
    부풀린 영역 '밖'은 절대 객체가 아니므로 '확실한 배경'입니다.
    """
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sure_bg  = cv2.dilate(binary, kernel, iterations=MORPH_ITER)
    return sure_bg


def step3_sure_fg(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    [Step 3] 확실한 전경(Sure Foreground) 추출 — 거리 변환(Distance Transform)

    distanceTransform: 각 픽셀에서 가장 가까운 배경 픽셀까지의 거리를 계산합니다.
    → 객체 중심부일수록 값이 크고, 가장자리는 값이 작습니다.
    → 임계값으로 잘라내면 '절대 객체 중심' 영역(sure_fg)만 남습니다.

    DIST_THRESH 비율이 낮을수록: 중심 영역을 넓게 잡아 더 많이 분리
    DIST_THRESH 비율이 높을수록: 중심을 좁게 잡아 덜 분리됨
    """
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    # 최대 거리 값의 DIST_THRESH 비율 이상인 픽셀만 확실한 전경으로 선택
    _, sure_fg = cv2.threshold(dist_transform,
                                DIST_THRESH * dist_transform.max(),
                                255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    return sure_fg, dist_transform


def step4_unknown_and_markers(sure_bg: np.ndarray,
                               sure_fg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    [Step 4] 불확실 영역(Unknown) 계산 및 마커 생성

    unknown = sure_bg - sure_fg
    → 배경도 전경도 아닌 '경계 불명확' 영역입니다.
    → Watershed가 이 영역의 경계를 찾아 채워줍니다.

    마커(Marker):
    - 0      : unknown (Watershed가 판단할 영역)
    - 1      : 확실한 배경
    - 2 이상 : 각 객체의 고유 ID (connectedComponents로 라벨링)
    """
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 연결된 전경 컴포넌트에 고유 라벨 부여
    num_labels, markers = cv2.connectedComponents(sure_fg)

    # 배경 라벨(0)을 1로 올려 Watershed가 배경을 인식하게 함
    markers = markers + 1
    # unknown 영역은 0으로 설정 → Watershed가 경계를 탐색할 영역
    markers[unknown == 255] = 0

    return unknown, markers


def step5_watershed(img: np.ndarray,
                    markers: np.ndarray) -> tuple[np.ndarray, int]:
    """
    [Step 5] Watershed 알고리즘 실행

    원리: 마커를 '씨앗'으로 물을 채우듯 영역을 확장합니다.
    두 영역이 만나는 지점 = 경계(watershed line) → markers에서 -1로 표시됩니다.

    반환: (watershed 결과 markers, 검출된 객체 수)
    """
    markers = cv2.watershed(img, markers)
    num_objects = markers.max() - 1  # 라벨 1=배경이므로 제외
    return markers, num_objects


def step6_colorize(img: np.ndarray, markers: np.ndarray,
                   num_objects: int) -> np.ndarray:
    """
    [Step 6] 결과 시각화

    각 객체에 고유한 색상을 배정하고 경계선을 흰색으로 표시합니다.
    작은 노이즈 객체는 MIN_OBJECT_AREA 기준으로 제거합니다.
    """
    result = img.copy()

    # 객체마다 무작위 색상 생성 (라벨 1=배경 제외, 2부터 시작)
    np.random.seed(0)
    colors = {label: tuple(int(c) for c in np.random.randint(60, 230, 3))
              for label in range(2, num_objects + 2)}

    valid_count = 0
    for label, color in colors.items():
        mask = (markers == label).astype(np.uint8)
        area = cv2.countNonZero(mask)
        if area < MIN_OBJECT_AREA:   # 너무 작은 노이즈 제거
            continue

        # 객체 영역을 반투명 색상으로 채색
        colored = np.zeros_like(img)
        colored[mask == 1] = color
        result = cv2.addWeighted(result, 0.55, colored, 0.45, 0)

        # 객체 번호 텍스트 (무게중심 계산)
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(result, str(valid_count + 1), (cx - 8, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        valid_count += 1

    # 경계선(-1) 흰색으로 표시
    result[markers == -1] = [255, 255, 255]

    # 우측 상단 카운트 표시
    label_text = f"Objects: {valid_count}"
    cv2.putText(result, label_text, (result.shape[1] - 160, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2, cv2.LINE_AA)

    return result, valid_count


def show_pipeline(steps: list[tuple[str, np.ndarray]]) -> None:
    """
    중간 단계 이미지를 한 창에 격자로 출력합니다.
    각 단계에 제목 텍스트를 삽입합니다.
    """
    titled = []
    for title, img_step in steps:
        # 그레이스케일이면 3채널로 변환
        if len(img_step.shape) == 2:
            display = cv2.cvtColor(img_step, cv2.COLOR_GRAY2BGR)
        else:
            display = img_step.copy()

        display = cv2.resize(display, (320, 240))
        # 반투명 검은 배경 위에 제목
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (320, 28), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)
        cv2.putText(display, title, (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
        titled.append(display)

    # 2행 3열 격자
    row1 = np.hstack(titled[:3])
    row2 = np.hstack(titled[3:6])
    grid = np.vstack([row1, row2])
    cv2.imshow("Watershed Pipeline  [any key: 종료]", grid)


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def run(image_path: str) -> None:
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] 이미지 로드 실패: {image_path}")
        sys.exit(1)

    print(f"[INFO] 이미지 로드: {image_path}  ({img.shape[1]}x{img.shape[0]})")

    # ── 파이프라인 실행 ──────────────────────────
    binary         = step1_preprocess(img)
    sure_bg        = step2_sure_bg(binary)
    sure_fg, dist  = step3_sure_fg(binary)
    unknown, markers = step4_unknown_and_markers(sure_bg, sure_fg)
    markers, n_obj   = step5_watershed(img, markers)
    result, count    = step6_colorize(img, markers, n_obj)

    print(f"[INFO] 검출된 객체 수: {count}개")

    # 거리 변환 시각화 (정규화)
    dist_vis = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dist_colored = cv2.applyColorMap(dist_vis, cv2.COLORMAP_JET)

    # unknown 영역 시각화
    unknown_vis = cv2.cvtColor(unknown, cv2.COLOR_GRAY2BGR)
    unknown_vis[unknown == 255] = (0, 165, 255)  # 주황색으로 표시

    # ── 단계별 파이프라인 창 ──────────────────────
    if SHOW_STEPS:
        show_pipeline([
            ("Step1: Binary (Otsu)", binary),
            ("Step2: Sure BG (Dilate)", sure_bg),
            ("Step3: Distance Transform", dist_colored),
            ("Step4: Sure FG + Unknown", unknown_vis),
            ("Step5: Markers", sure_fg),          # 전경 씨앗 위치
            ("Step6: Watershed Result", result),
        ])

    # ── 최종 결과 저장 ───────────────────────────
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cv2.imwrite(OUT_PATH, result)
    print(f"[INFO] 결과 저장: {OUT_PATH}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH

    if not os.path.exists(image_path):
        print(f"[WARNING] 이미지 없음: {image_path} → 테스트 이미지 자동 생성")
        make_test_image(image_path)

    run(image_path)