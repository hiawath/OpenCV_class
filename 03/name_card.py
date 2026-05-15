"""
명함 스캐너 - Perspective Transform (Homography Warping)
=======================================================
기울어지게 촬영된 명함의 네 모서리를 찾아 정면 뷰로 평면화(Warping)합니다.

알고리즘 흐름:
  1. 전처리 (그레이스케일 → 블러 → Canny 엣지)
  2. 외곽선(Contour) 탐색 → 가장 큰 사각형 꼭짓점 4개 추출
  3. 꼭짓점 순서 정렬 (좌상·우상·우하·좌하)
  4. 목적지(dst) 좌표 계산 (유클리드 거리 기반 너비·높이)
  5. cv2.getPerspectiveTransform() → 3×3 단응사상 행렬 H 생성
  6. cv2.warpPerspective() → 최종 정면 뷰 출력
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────
# 1. 꼭짓점 정렬 유틸리티
# ─────────────────────────────────────────────
def sort_corners(pts: np.ndarray) -> np.ndarray:
    """
    4개의 꼭짓점을 [좌상, 우상, 우하, 좌하] 순서로 정렬합니다.

    Args:
        pts: shape (4, 2) — (x, y) 좌표 배열

    Returns:
        정렬된 shape (4, 2) 배열
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    # 합(x+y)이 가장 작은 점 → 좌상단
    # 합(x+y)이 가장 큰 점  → 우하단
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # 좌상 (top-left)
    rect[2] = pts[np.argmax(s)]   # 우하 (bottom-right)

    # 차(y-x)가 가장 작은 점 → 우상단
    # 차(y-x)가 가장 큰 점  → 좌하단
    diff = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(diff)]  # 우상 (top-right)
    rect[3] = pts[np.argmax(diff)]  # 좌하 (bottom-left)

    return rect


# ─────────────────────────────────────────────
# 2. 목적지 너비/높이 계산
# ─────────────────────────────────────────────
def compute_dst_size(rect: np.ndarray) -> tuple[int, int]:
    """
    정렬된 꼭짓점 4개의 유클리드 거리를 기반으로
    출력 이미지의 너비(W)와 높이(H)를 계산합니다.

    Args:
        rect: sort_corners()의 반환값 (4, 2)

    Returns:
        (width, height) 정수 튜플
    """
    tl, tr, br, bl = rect

    # 위쪽 변 길이와 아래쪽 변 길이 중 최댓값 → 출력 너비
    width_top    = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    W = int(max(width_top, width_bottom))

    # 왼쪽 변 길이와 오른쪽 변 길이 중 최댓값 → 출력 높이
    height_left  = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    H = int(max(height_left, height_right))

    return W, H


# ─────────────────────────────────────────────
# 3. 자동 꼭짓점 탐색 (Contour 기반)
# ─────────────────────────────────────────────
def find_card_corners_auto(image: np.ndarray) -> np.ndarray | None:
    """
    입력 이미지에서 명함(사각형)의 꼭짓점 4개를 자동으로 탐색합니다.

    전처리 파이프라인:
        grayscale → GaussianBlur → Canny → dilate → findContours → approxPolyDP

    Args:
        image: BGR 또는 그레이스케일 이미지

    Returns:
        꼭짓점 배열 shape (4, 2) 또는 None (탐지 실패 시)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # 노이즈 제거 (커널 크기 5×5)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny 엣지 검출 (자동 임계값: Otsu 기반)
    otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(blurred, otsu_thresh * 0.5, otsu_thresh)

    # 엣지 팽창으로 끊어진 선 연결
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # 외곽선 탐색 (RETR_EXTERNAL: 최외곽만)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 면적 기준 내림차순 정렬 → 상위 5개만 검사
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for cnt in contours:
        # 호의 길이의 2% 오차 허용으로 다각형 근사
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # 꼭짓점이 정확히 4개인 경우 → 명함 후보
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    return None  # 탐지 실패


# ─────────────────────────────────────────────
# 4. 핵심: Perspective Transform 적용
# ─────────────────────────────────────────────
def warp_card(image: np.ndarray, src_pts: np.ndarray) -> np.ndarray:
    """
    명함의 꼭짓점 4개를 기반으로 Perspective Transform을 적용합니다.

    수식 (단응사상 행렬 H):
        dst_pt = H @ src_pt     (동차 좌표계)
        H = cv2.getPerspectiveTransform(src, dst)   # 3×3

    Args:
        image  : 입력 BGR 이미지
        src_pts: 꼭짓점 4개 배열 — 임의 순서 가능 (내부에서 정렬)

    Returns:
        정면 뷰(warped) BGR 이미지
    """
    # ① 꼭짓점 정렬: 좌상 → 우상 → 우하 → 좌하
    rect = sort_corners(src_pts)
    tl, tr, br, bl = rect

    # ② 출력 크기 계산
    W, H = compute_dst_size(rect)

    # ③ 목적지(destination) 좌표 — 직사각형
    dst_pts = np.array([
        [0,     0    ],   # 좌상
        [W - 1, 0    ],   # 우상
        [W - 1, H - 1],   # 우하
        [0,     H - 1],   # 좌하
    ], dtype=np.float32)

    # ④ getPerspectiveTransform → 3×3 단응사상 행렬 H
    M = cv2.getPerspectiveTransform(rect, dst_pts)
    print(f"[변환 행렬 H]\n{np.round(M, 4)}\n")

    # ⑤ warpPerspective → 역방향 매핑(inverse mapping)으로 픽셀 보간
    warped = cv2.warpPerspective(image, M, (W, H),
                                  flags=cv2.INTER_LINEAR)
    return warped


# ─────────────────────────────────────────────
# 5. 전체 파이프라인
# ─────────────────────────────────────────────
def scan_business_card(image_path: str,
                        output_path: str = "warped_card.jpg",
                        manual_pts: list[tuple] | None = None) -> np.ndarray:
    """
    명함 이미지를 읽어 Perspective Transform을 적용한 뒤 저장합니다.

    Args:
        image_path : 입력 이미지 경로
        output_path: 출력 이미지 저장 경로 (기본값: warped_card.jpg)
        manual_pts : 수동으로 지정할 꼭짓점 4개 [(x1,y1),...] — None이면 자동 탐지

    Returns:
        정면 뷰(warped) 이미지 ndarray
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")

    h, w = image.shape[:2]
    print(f"입력 이미지: {w}×{h}px")

    # 꼭짓점 결정
    if manual_pts is not None:
        src_pts = np.array(manual_pts, dtype=np.float32)
        print(f"수동 꼭짓점 사용: {src_pts.tolist()}")
    else:
        src_pts = find_card_corners_auto(image)
        if src_pts is None:
            raise RuntimeError("명함 꼭짓점을 자동으로 탐지할 수 없습니다. "
                               "manual_pts를 직접 지정해 주세요.")
        print(f"자동 탐지 꼭짓점: {src_pts.tolist()}")

    # Perspective Transform 적용
    warped = warp_card(image, src_pts)
    wh, ww = warped.shape[:2]
    print(f"출력 이미지: {ww}×{wh}px")

    cv2.imwrite(output_path, warped)
    print(f"저장 완료: {output_path}")

    return warped


# ─────────────────────────────────────────────
# 6. 마우스 클릭으로 꼭짓점 수동 선택 (대화형)
# ─────────────────────────────────────────────
def interactive_scan(image_path: str, output_path: str = "warped_card.jpg") -> np.ndarray:
    """
    이미지를 화면에 띄우고 마우스로 꼭짓점 4개를 클릭해 변환합니다.
    클릭 순서: 좌상단 → 우상단 → 우하단 → 좌하단

    Args:
        image_path : 입력 이미지 경로
        output_path: 출력 저장 경로

    Returns:
        정면 뷰 이미지 ndarray
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")

    clicked_pts: list[tuple[int, int]] = []
    clone = image.copy()
    labels = ["TL (좌상)", "TR (우상)", "BR (우하)", "BL (좌하)"]
    colors = [(0, 200, 120), (55, 138, 221), (216, 90, 48), (186, 117, 23)]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_pts) < 4:
            clicked_pts.append((x, y))
            idx = len(clicked_pts) - 1
            cv2.circle(clone, (x, y), 6, colors[idx], -1)
            cv2.circle(clone, (x, y), 8, (255, 255, 255), 1)
            cv2.putText(clone, labels[idx], (x + 10, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 1)
            if len(clicked_pts) > 1:
                cv2.line(clone, clicked_pts[-2], clicked_pts[-1], (100, 200, 100), 1)
            if len(clicked_pts) == 4:
                cv2.line(clone, clicked_pts[-1], clicked_pts[0], (100, 200, 100), 1)
            cv2.imshow("명함 스캐너 — 꼭짓점 선택", clone)

    cv2.namedWindow("명함 스캐너 — 꼭짓점 선택")
    cv2.setMouseCallback("명함 스캐너 — 꼭짓점 선택", on_mouse)
    cv2.imshow("명함 스캐너 — 꼭짓점 선택", clone)

    print("좌상 → 우상 → 우하 → 좌하 순서로 꼭짓점 4개를 클릭하세요.")
    print("4개 클릭 후 아무 키나 누르면 변환합니다.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key != 255 and len(clicked_pts) == 4:
            break
        if key == 27:   # ESC → 취소
            cv2.destroyAllWindows()
            return image

    cv2.destroyAllWindows()
    return scan_business_card(image_path, output_path, manual_pts=clicked_pts)


# ─────────────────────────────────────────────
# 사용 예시
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    IMAGE_PATH = './images/name_card.png' # 입력 동영상 파일 경로


    # ── 예시 1: 자동 탐지 ──────────────────────
    #warped = scan_business_card("./images/name_card.png")

    # ── 예시 2: 수동 꼭짓점 지정 ────────────────
    # (픽셀 좌표는 실제 이미지에 맞게 수정하세요)
    manual = [
        (120, 80),   # 좌상단 (TL)
        (480, 60),   # 우상단 (TR)
        (510, 280),  # 우하단 (BR)
        (90,  300),  # 좌하단 (BL)
    ]
    # warped = scan_business_card("card_skewed.jpg", manual_pts=manual)

    # ── 예시 3: 대화형 마우스 선택 ──────────────
    if len(sys.argv) > 1:
        result = interactive_scan(sys.argv[1], "warped_output.jpg")
        cv2.imshow("결과 — 정면 뷰", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("사용법: python business_card_scanner.py <이미지 경로>")
        print("\n알고리즘 요약:")
        print("  1. cv2.Canny()          — 엣지 검출")
        print("  2. cv2.findContours()   — 외곽선 탐색")
        print("  3. cv2.approxPolyDP()   — 사각형 꼭짓점 근사")
        print("  4. sort_corners()       — TL/TR/BR/BL 정렬")
        print("  5. getPerspectiveTransform(src, dst)  — 3×3 행렬 H")
        print("  6. warpPerspective(img, H, (W, H))    — 정면 뷰 획득")

        result = interactive_scan(IMAGE_PATH, "./temp/warped_output.jpg")
        cv2.imshow("결과 — 정면 뷰", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()