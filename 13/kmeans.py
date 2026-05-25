import cv2
import numpy as np
import os
import sys

# ─────────────────────────────────────────────
# 설정값
# ─────────────────────────────────────────────
IMAGE_PATH  = "./images/fruits.jpg"
K           = 5          # 추출할 주요 색상 수
MAX_ITER    = 20         # K-Means 최대 반복 횟수
EPSILON     = 1.0        # K-Means 수렴 기준 (픽셀 단위)
ATTEMPTS    = 5          # 다른 초기값으로 시도할 횟수 (최적 결과 선택)
SWATCH_W    = 160        # 색상 팔레트 스와치 너비(px)
SWATCH_H    = 80         # 색상 팔레트 스와치 높이(px)
# ─────────────────────────────────────────────


def make_test_image(path: str) -> None:
    """테스트용 다채로운 합성 이미지 생성"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h, w = 480, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # 배경 하늘색
    img[:, :] = (180, 210, 240)

    # 색상 블록들 (다양한 주요 색상 구성)
    regions = [
        ((0,   0,   320, 240), (60,  120, 200)),   # 파란 하늘
        ((240, 0,   480, 640), (80,  160,  70)),   # 초록 풀밭
        ((0,   320, 240, 640), (220, 100,  50)),   # 주황빛 땅
        ((100, 100, 220, 300), (230, 210, 180)),   # 밝은 베이지 (건물)
        ((110, 420, 230, 580), (50,   50, 160)),   # 진한 파랑 (그림자)
    ]
    for (y1, x1, y2, x2), color in regions:
        img[y1:y2, x1:x2] = color

    # 노이즈 추가 (자연스러운 픽셀 분포)
    noise = np.random.randint(-18, 18, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite(path, img)
    print(f"[INFO] 테스트 이미지 생성: {path}")


# ─────────────────────────────────────────────
# K-Means 색상 추출 핵심 함수
# ─────────────────────────────────────────────

def extract_colors(img_bgr: np.ndarray, k: int) -> tuple[list, list]:
    """
    K-Means 클러스터링으로 이미지의 주요 색상을 추출합니다.

    처리 순서:
      1. BGR → RGB 변환 (출력 색상 코드를 RGB 기준으로 맞추기 위해)
      2. 픽셀 배열 재구성: (H, W, 3) → (H*W, 3), float32
         cv2.kmeans는 2D 배열(샘플 수 × 특징 수)을 요구합니다.
      3. cv2.kmeans 실행
         - data   : 픽셀 배열 (N × 3, float32)
         - K      : 클러스터 수
         - labels : 각 픽셀의 클러스터 번호 (N × 1)
         - centers: 각 클러스터 중심 색상 (K × 3)
      4. 각 클러스터의 픽셀 비율 계산 → 비율 내림차순 정렬
    """
    # 1. BGR → RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. (H, W, 3) → (N, 3) float32  ← 핵심 배열 재구성
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)

    # 3. K-Means 실행
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        MAX_ITER,
        EPSILON,
    )
    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,                       # bestLabels: None이면 내부에서 초기화
        criteria,
        ATTEMPTS,
        cv2.KMEANS_PP_CENTERS,      # K-Means++ 초기화 (수렴 안정성 향상)
    )

    # 4. 클러스터별 픽셀 수 → 비율 계산
    labels_flat = labels.flatten()                      # (N,)
    total_pixels = len(labels_flat)

    color_info = []
    for idx in range(k):
        count = int(np.sum(labels_flat == idx))
        ratio = count / total_pixels
        rgb   = tuple(int(c) for c in centers[idx])    # float → int
        color_info.append((ratio, rgb))

    # 비율 내림차순 정렬
    color_info.sort(key=lambda x: x[0], reverse=True)

    ratios = [ci[0] for ci in color_info]
    colors = [ci[1] for ci in color_info]
    return colors, ratios


# ─────────────────────────────────────────────
# 시각화 함수
# ─────────────────────────────────────────────

def luminance(r: int, g: int, b: int) -> float:
    """상대 휘도 계산 — 텍스트 색상(흰/검) 자동 선택에 사용"""
    def lin(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def build_palette(colors: list, ratios: list,
                  swatch_w: int = SWATCH_W,
                  swatch_h: int = SWATCH_H) -> np.ndarray:
    """
    추출된 색상으로 팔레트 이미지를 생성합니다.
    각 스와치에 HEX 코드, RGB, 비율(%)을 표시합니다.
    """
    k          = len(colors)
    padding    = 10
    text_area  = 60           # 텍스트 영역 높이
    total_h    = swatch_h + text_area + padding * 2
    total_w    = swatch_w * k + padding * (k + 1)
    palette    = np.full((total_h, total_w, 3), 245, dtype=np.uint8)  # 밝은 회색 배경

    for i, (rgb, ratio) in enumerate(zip(colors, ratios)):
        r, g, b = rgb
        bgr = (b, g, r)

        x1 = padding + i * (swatch_w + padding)
        y1 = padding

        # 색상 스와치
        cv2.rectangle(palette, (x1, y1), (x1 + swatch_w, y1 + swatch_h), bgr, -1)
        cv2.rectangle(palette, (x1, y1), (x1 + swatch_w, y1 + swatch_h), (180, 180, 180), 1)

        # 텍스트 색상 (밝기에 따라 흰/검 자동 선택)
        lum       = luminance(r, g, b)
        txt_color = (255, 255, 255) if lum < 0.35 else (30, 30, 30)

        # 스와치 위에 비율 표시
        pct_text = f"{ratio * 100:.1f}%"
        (tw, th), _ = cv2.getTextSize(pct_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        tx = x1 + (swatch_w - tw) // 2
        ty = y1 + swatch_h // 2 + th // 2
        cv2.putText(palette, pct_text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, txt_color, 1, cv2.LINE_AA)

        # 스와치 아래 텍스트 (어두운 색으로 고정)
        text_y_base = y1 + swatch_h + padding + 4
        dark = (40, 40, 40)

        # HEX 코드
        hex_code = f"#{r:02X}{g:02X}{b:02X}"
        (tw, _), _ = cv2.getTextSize(hex_code, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(palette, hex_code, (x1 + (swatch_w - tw) // 2, text_y_base + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, dark, 1, cv2.LINE_AA)

        # RGB 코드
        rgb_text = f"({r},{g},{b})"
        (tw, _), _ = cv2.getTextSize(rgb_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(palette, rgb_text, (x1 + (swatch_w - tw) // 2, text_y_base + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1, cv2.LINE_AA)

    return palette


def print_results(colors: list, ratios: list) -> None:
    """터미널에 결과 출력"""
    print("\n" + "=" * 50)
    print(f"  주요 색상 TOP {len(colors)} (K-Means, K={K})")
    print("=" * 50)
    print(f"  {'순위':<4} {'HEX':<10} {'RGB':<20} {'비율':>6}")
    print("-" * 50)
    for rank, (rgb, ratio) in enumerate(zip(colors, ratios), 1):
        r, g, b  = rgb
        hex_code = f"#{r:02X}{g:02X}{b:02X}"
        rgb_str  = f"({r:3d}, {g:3d}, {b:3d})"
        bar      = "█" * int(ratio * 30)
        print(f"  {rank:<4} {hex_code:<10} {rgb_str:<20} {ratio*100:5.1f}%  {bar}")
    print("=" * 50 + "\n")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def run(image_path: str) -> None:
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] 이미지 로드 실패: {image_path}")
        sys.exit(1)

    h, w = img.shape[:2]
    print(f"[INFO] 이미지 로드: {image_path}  ({w}x{h}, {w*h:,} 픽셀)")
    print(f"[INFO] K-Means 실행 중... (K={K}, attempts={ATTEMPTS})")

    # ── K-Means 색상 추출 ──────────────────────
    colors, ratios = extract_colors(img, K)

    # ── 터미널 출력 ───────────────────────────
    print_results(colors, ratios)

    # ── 팔레트 이미지 생성 ────────────────────
    palette = build_palette(colors, ratios)

    # ── 결과 창 표시 ─────────────────────────
    # 원본 이미지를 팔레트 너비에 맞게 리사이즈
    palette_w = palette.shape[1]
    scale     = palette_w / w
    img_resized = cv2.resize(img, (palette_w, int(h * scale)))

    # 원본 + 팔레트 세로로 합치기
    combined = np.vstack([img_resized, palette])

    cv2.imshow(f"Color Extraction (K={K})  [any key: 종료]", combined)

    # 결과 저장
    out_path = os.path.join(os.path.dirname(image_path), "color_palette.jpg")
    cv2.imwrite(out_path, combined)
    print(f"[INFO] 팔레트 저장: {out_path}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH

    if not os.path.exists(image_path):
        print(f"[WARNING] 이미지 없음: {image_path} → 테스트 이미지 자동 생성")
        make_test_image(image_path)

    run(image_path)