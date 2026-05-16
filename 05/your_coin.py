"""
동전 개수 카운팅 - Contour & Properties
cv2.findContours + cv2.contourArea 기반 객체 탐지 및 크기별 분류
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ──────────────────────────────────────────
# 1. 테스트용 합성 동전 이미지 생성
# ──────────────────────────────────────────
def create_coin_image(n_coins: int = 12, img_size: tuple = (480, 640)) -> np.ndarray:
    """랜덤한 크기의 동전들이 흩어진 그레이스케일 이미지를 생성합니다."""
    h, w = img_size
    img = np.full((h, w), 60, dtype=np.uint8)   # 어두운 배경

    # 노이즈 추가 (실제 이미지 질감 모사)
    noise = np.random.randint(0, 20, (h, w), dtype=np.uint8)
    img = cv2.add(img, noise)

    coins = []
    rng = np.random.default_rng(42)

    for _ in range(n_coins * 30):           # 겹침 없이 배치
        if len(coins) >= n_coins:
            break
        r = int(rng.integers(15, 50))       # 동전 반지름 15–50 px
        x = int(rng.integers(r + 5, w - r - 5))
        y = int(rng.integers(r + 5, h - r - 5))

        # 기존 동전과 겹침 검사
        overlap = any(
            np.hypot(x - cx, y - cy) < r + cr + 5
            for cx, cy, cr in coins
        )
        if overlap:
            continue

        coins.append((x, y, r))

        # 밝은 원 (동전 외관)
        cv2.circle(img, (x, y), r, 200, -1)
        cv2.circle(img, (x, y), r, 160, 2)        # 테두리
        # 하이라이트 효과
        cv2.circle(img, (x - r // 4, y - r // 4), r // 3, 230, -1)

    return img


# ──────────────────────────────────────────
# 2. 동전 탐지 및 분류 함수
# ──────────────────────────────────────────
SIZE_THRESHOLDS = {
    "S": (0, 1500),      # 소형: 면적 < 1500 px²
    "M": (1500, 4000),   # 중형: 1500 ≤ 면적 < 4000 px²
    "L": (4000, np.inf), # 대형: 면적 ≥ 4000 px²
}

SIZE_COLORS_BGR = {
    "S": (80, 200, 120),   # 초록
    "M": (220, 150, 50),   # 파랑
    "L": (50, 100, 210),   # 빨강 계열
}


def classify_by_area(area: float) -> str:
    for size, (lo, hi) in SIZE_THRESHOLDS.items():
        if lo <= area < hi:
            return size
    return "L"


def detect_coins(gray: np.ndarray, min_area: float = 500) -> list[dict]:
    """
    그레이스케일 이미지에서 동전을 탐지하고 크기별로 분류합니다.

    Parameters
    ----------
    gray     : 그레이스케일 입력 이미지
    min_area : 잡음 제거를 위한 최소 면적 임계값 (px²)

    Returns
    -------
    탐지된 동전 정보(bounding_rect, area, size)의 리스트
    """
    # ① 가우시안 블러로 고주파 노이즈 제거
    blurred = cv2.GaussianBlur(gray, (11, 11), 2)

    # ② Otsu 이진화 (임계값 자동 결정)
    _, binary = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ③ 모폴로지 연산으로 작은 구멍/돌출 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)

    # ④ 외곽선 검출
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    coins = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:          # 최소 면적 필터
            continue

        # ⑤ 면적 기반 크기 분류
        size = classify_by_area(area)

        # 바운딩 박스 (x, y, w, h)
        bx, by, bw, bh = cv2.boundingRect(cnt)

        # 원형도 계산 (circularity = 4π·면적 / 둘레²)
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area / (perimeter ** 2)
                       if perimeter > 0 else 0)

        coins.append({
            "contour":    cnt,
            "area":       area,
            "size":       size,
            "bbox":       (bx, by, bw, bh),
            "circularity": circularity,
        })

    # 면적 내림차순 정렬
    return sorted(coins, key=lambda c: c["area"], reverse=True)


# ──────────────────────────────────────────
# 3. 시각화 함수
# ──────────────────────────────────────────
def draw_results(original: np.ndarray, coins: list[dict]) -> np.ndarray:
    """탐지 결과를 바운딩 박스와 레이블로 시각화합니다."""
    result = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    for idx, coin in enumerate(coins, start=1):
        bx, by, bw, bh = coin["bbox"]
        color = SIZE_COLORS_BGR[coin["size"]]

        # 바운딩 박스
        cv2.rectangle(result, (bx, by), (bx + bw, by + bh), color, 2)

        # 레이블 배경 + 텍스트
        label = f"#{idx} {coin['size']} {int(coin['area'])}px2"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(result, (bx, by - th - 6), (bx + tw + 4, by), color, -1)
        cv2.putText(result, label, (bx + 2, by - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)

    return result


def visualize_pipeline(gray: np.ndarray, coins: list[dict],
                        binary: np.ndarray | None = None) -> None:
    """원본 / 이진화 / 결과 이미지를 matplotlib으로 표시합니다."""

    plt.rcParams["font.family"] = "AppleGothic"   # 맥OS 기본 한글 폰트
    plt.rcParams["axes.unicode_minus"] = False     # − 기호도 같이 깨지므로 필수

    result_bgr = draw_results(gray, coins)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("동전 카운팅 파이프라인", fontsize=14, fontweight="bold")

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("① 원본 그레이스케일")
    axes[0].axis("off")

    if binary is not None:
        axes[1].imshow(binary, cmap="gray")
        axes[1].set_title("② Otsu 이진화 + 모폴로지")
    axes[1].axis("off")

    axes[2].imshow(result_rgb)
    axes[2].set_title(f"③ 탐지 결과 (총 {len(coins)}개)")
    axes[2].axis("off")

    # 범례
    patches = [
        mpatches.Patch(color=tuple(c / 255 for c in v[::-1]), label=f"{k}형")
        for k, v in SIZE_COLORS_BGR.items()
    ]
    axes[2].legend(handles=patches, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig("coin_result.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("결과 이미지 저장: coin_result.png")


# ──────────────────────────────────────────
# 4. 통계 출력
# ──────────────────────────────────────────
def print_stats(coins: list[dict]) -> None:
    counts = {"S": 0, "M": 0, "L": 0}
    for c in coins:
        counts[c["size"]] += 1

    print("\n" + "=" * 45)
    print("  동전 탐지 결과 요약")
    print("=" * 45)
    print(f"  전체 탐지 개수  : {len(coins)}개")
    print(f"  소형 S          : {counts['S']}개")
    print(f"  중형 M          : {counts['M']}개")
    print(f"  대형 L          : {counts['L']}개")
    print("-" * 45)
    print(f"  {'#':>3}  {'크기':>4}  {'면적(px²)':>10}  {'원형도':>7}")
    print("-" * 45)
    for i, c in enumerate(coins, 1):
        print(f"  {i:>3}  {c['size']:>4}  "
              f"{int(c['area']):>10,}  {c['circularity']:>7.3f}")
    print("=" * 45)


# ──────────────────────────────────────────
# 5. 메인 실행
# ──────────────────────────────────────────
def main(image_path: str | None = None) -> None:
    """
    Parameters
    ----------
    image_path : 실제 동전 이미지 경로 (None이면 합성 이미지 사용)
    """
    if image_path:
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")
    else:
        print("합성 동전 이미지를 생성합니다...")
        gray = create_coin_image(n_coins=14)

    # 탐지 실행
    coins = detect_coins(gray, min_area=500)

    # 이진화 이미지 재생성 (시각화용)
    blurred = cv2.GaussianBlur(gray, (11, 11), 2)
    _, binary = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)

    print_stats(coins)
    visualize_pipeline(gray, coins, binary)


if __name__ == "__main__":
    # 실제 이미지 사용 시: main("your_coins.jpg")
    main()