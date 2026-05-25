import cv2
import numpy as np
import os, sys

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BG_PATH   = "./images/airplane.jpg"
LOGO_PATH = "./images/logo.png"
OUT_PATH  = "./images/result.jpg"
MARGIN    = 20          # 우측 하단 여백(px)
LOGO_MAX  = 200         # 로고 최대 너비/높이(px) — 너무 크면 축소
ALPHA     = 0.85        # 로고 불투명도 (0.0 투명 ~ 1.0 불투명)
# ─────────────────────────────────────────────


# ── 테스트용 이미지 자동 생성 ────────────────
def make_test_background(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h, w = 480, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # 하늘 그라디언트
    for y in range(h):
        t = y / h
        img[y, :] = [int(200 - 120 * t), int(180 - 60 * t), int(240 - 40 * t)]
    # 구름 모양 타원
    cv2.ellipse(img, (160, 120), (90, 35), 0, 0, 360, (230, 230, 240), -1)
    cv2.ellipse(img, (420, 90),  (70, 28), 0, 0, 360, (225, 228, 238), -1)
    # 비행기 실루엣
    pts_body = np.array([[260,230],[380,230],[400,250],[380,270],[260,270]], np.int32)
    cv2.fillPoly(img, [pts_body], (80, 80, 90))
    cv2.ellipse(img, (390, 250), (50, 20), 0, 0, 360, (75, 75, 85), -1)
    cv2.rectangle(img, (280, 215), (350, 230), (70, 70, 80), -1)  # 날개
    cv2.imwrite(path, img)
    print(f"[INFO] 배경 이미지 생성: {path}")


def make_test_logo(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    size = 120
    # BGRA (4채널) — Alpha 포함
    logo = np.zeros((size, size, 4), dtype=np.uint8)

    # 원형 배경 (반투명 파란색)
    cv2.circle(logo, (size//2, size//2), size//2 - 4, (200, 100, 30, 220), -1)
    cv2.circle(logo, (size//2, size//2), size//2 - 4, (255, 160, 60, 255), 3)

    # 별 모양 (흰색)
    cx, cy, r_out, r_in, n = size//2, size//2, 38, 18, 5
    pts = []
    for i in range(n * 2):
        angle = np.pi / n * i - np.pi / 2
        r = r_out if i % 2 == 0 else r_in
        pts.append([int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))])
    cv2.fillPoly(logo, [np.array(pts, np.int32)], (255, 255, 255, 255))

    # 텍스트
    cv2.putText(logo, "LOGO", (22, size - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(path, logo)
    print(f"[INFO] 로고 이미지 생성: {path}")


# ── 핵심 합성 함수 ────────────────────────────
def overlay_logo(bg: np.ndarray, logo_bgra: np.ndarray,
                 margin: int = MARGIN, alpha: float = ALPHA) -> np.ndarray:
    """
    배경 이미지(bg) 우측 하단에 로고(logo_bgra, BGRA)를 합성합니다.

    처리 순서:
      1. 로고 크기가 LOGO_MAX를 초과하면 축소
      2. 우측 하단 ROI 좌표 계산
      3. Alpha 채널로 마스크/반전마스크 생성
      4. bitwise_and 로 각 영역 분리
      5. cv2.add 로 합산 → 불투명도(alpha) 적용
    """
    result = bg.copy()
    H, W = bg.shape[:2]

    # ── 1. 로고 크기 조정 ──
    lh, lw = logo_bgra.shape[:2]
    scale = min(LOGO_MAX / lw, LOGO_MAX / lh, 1.0)  # 최대 크기 초과 시만 축소
    if scale < 1.0:
        lw = int(lw * scale)
        lh = int(lh * scale)
        logo_bgra = cv2.resize(logo_bgra, (lw, lh), interpolation=cv2.INTER_AREA)

    # ── 2. 우측 하단 ROI 좌표 (핵심 수정) ──
    x1 = W - lw - margin
    y1 = H - lh - margin
    x2, y2 = x1 + lw, y1 + lh

    if x1 < 0 or y1 < 0:
        print("[WARNING] 로고가 배경보다 큽니다. margin을 줄이거나 로고를 축소하세요.")
        x1, y1 = max(x1, 0), max(y1, 0)

    roi = result[y1:y2, x1:x2]

    # ── 3. 마스크 생성 ──
    logo_bgr = logo_bgra[:, :, :3]   # BGR 채널만 분리

    if logo_bgra.shape[2] == 4:
        # PNG Alpha 채널을 마스크로 직접 사용 (투명도 정확하게 반영)
        alpha_ch = logo_bgra[:, :, 3]
    else:
        # Alpha 없는 경우: 밝기 기반 임계값으로 마스크 생성
        gray = cv2.cvtColor(logo_bgr, cv2.COLOR_BGR2GRAY)
        _, alpha_ch = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # 불투명도(ALPHA) 적용: 마스크를 스케일링
    mask     = cv2.multiply(alpha_ch, alpha)           # 로고 영역
    mask     = np.clip(mask, 0, 255).astype(np.uint8)
    mask_inv = cv2.bitwise_not(mask)                   # 배경 영역

    # ── 4. 비트 연산 (핵심 수정: 마스크 역할 교정) ──
    #   img_bg  : ROI에서 로고가 들어갈 자리(=로고 영역)를 0으로 → mask_inv 사용
    #   logo_fg : 로고에서 배경 부분(=투명 영역)을 0으로   → mask 사용
    img_bg  = cv2.bitwise_and(roi,      roi,      mask=mask_inv)  # ← 원본 코드 버그 수정
    logo_fg = cv2.bitwise_and(logo_bgr, logo_bgr, mask=mask)      # ← 원본 코드 버그 수정

    # ── 5. 합산 후 ROI에 적용 ──
    dst = cv2.add(img_bg, logo_fg)
    result[y1:y2, x1:x2] = dst

    return result


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
if __name__ == "__main__":
    bg_path   = sys.argv[1] if len(sys.argv) > 1 else BG_PATH
    logo_path = sys.argv[2] if len(sys.argv) > 2 else LOGO_PATH

    # 파일 없으면 자동 생성
    if not os.path.exists(bg_path):
        print(f"[WARNING] 배경 이미지 없음: {bg_path}")
        make_test_background(bg_path)

    if not os.path.exists(logo_path):
        print(f"[WARNING] 로고 이미지 없음: {logo_path}")
        make_test_logo(logo_path)

    # 이미지 로드
    bg        = cv2.imread(bg_path)
    # IMREAD_UNCHANGED: Alpha 채널까지 포함해서 로드 (핵심 수정)
    logo_bgra = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)

    if bg is None:
        print(f"[ERROR] 배경 이미지 로드 실패: {bg_path}")
        sys.exit(1)
    if logo_bgra is None:
        print(f"[ERROR] 로고 이미지 로드 실패: {logo_path}")
        sys.exit(1)

    # 합성 실행
    result = overlay_logo(bg, logo_bgra)

    # 결과 저장
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cv2.imwrite(OUT_PATH, result)
    print(f"[INFO] 결과 저장 완료: {OUT_PATH}")

    # 나란히 비교 출력: [원본 | 합성 결과]
    compare = np.hstack([bg, result])
    cv2.imshow("Before (left)  |  After with Logo (right)  [any key: 종료]", compare)
    cv2.waitKey(0)
    cv2.destroyAllWindows()