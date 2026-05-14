import cv2
import numpy as np

# -----------------------------------------------------------------------
# 변수 설정
# -----------------------------------------------------------------------
VIDEO_PATH = './temp/drive_yy.mp4' # 입력 동영상 파일 경로
OUTPUT_PATH = './output_dashed_lanes.avi' # 출력 동영상 파일 경로
FRAME_SKIP = 2 # 테스트를 위해 2프레임마다 처리할 수 있도록 설정 (성능 최적화)

# -----------------------------------------------------------------------
# 1. 동영상 캡처 및 쓰기 설정
# -----------------------------------------------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"오류: 비디오 파일을 열 수 없습니다. 경로를 확인하세요: {VIDEO_PATH}")
    exit()

# 비디오 속성을 가져옵니다.
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# VideoWriter를 설정하여 결과를 저장할 비디오 객체 생성
# fourcc = cv2.VideoWriter_fourcc(*'XVID') # 비디오 코덱 지정
# out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

# print(f"--- 동영상 처리 시작 ---")
# print(f"입력: {VIDEO_PATH} | 출력: {OUTPUT_PATH} | FPS: {fps:.2f}")


# -----------------------------------------------------------------------
# 2. 메인 처리 루프
# -----------------------------------------------------------------------
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break # 더 이상 프레임이 없으면 루프 종료

    # 성능 최적화를 위해 일정 간격 프레임만 처리 (필요에 따라 주석 해제)
    if frame_count % FRAME_SKIP != 0:
        frame_count += 1
        continue

    # 원본 프레임 복사본 (결과를 그릴 용도)
    copy_img = frame.copy()

    # 2. 전처리: 그레이스케일 변환 및 가우시안 블러 (노이즈 제거)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Canny Edge 검출
    edges = cv2.Canny(blur, 50, 150)

    # 4. ROI (관심 영역) 설정 - 도로 영역 마스킹
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # --- ***중요***: 이 좌표는 이미지 해상도에 따라 조정되어야 합니다. ---
    # 원본 코드의 좌표를 유지하되, 실제 동영상 해상도(frame_width, frame_height)를 반영해야 합니다.
    # 예시 좌표: (좌상_x, 좌상_y), (우상_x, 우상_y), (우하_x, 우하_y), (좌하_x, 좌하_y)
    # 만약 해상도가 1280x720 이라면, 아래 좌표들을 수동으로 조정해야 합니다.
    polygon = np.array([[(0, height), (int(width * 0.1), int(height * 0.4)), (int(width * 0.9), int(height * 0.4)), (width, 0)]], np.int32)

    # 만약 위의 임의 좌표가 너무 심하게 벗어난다면, 원본 코드의 비율을 유지하며 임시로 좌표를 조정합니다.
    # 이 부분은 실제 영상을 보며 튜닝이 필요합니다.

    cv2.fillPoly(mask, polygon, 255)

    # 엣지 이미지와 마스크를 비트 AND 연산하여 도로 쪽 엣지만 남김
    roi_edges = cv2.bitwise_and(edges, mask)

    # 5. 확률적 허프 변환 (HoughLinesP) 적용
    lines = cv2.HoughLinesP(
        roi_edges,
        rho=1,                # 거리 해상도 (픽셀 단위)
        theta=np.pi/180,      # 각도 해상도 (라디안 단위, 1도)
        threshold=30,         # 직선으로 판단할 최소 엣지 점의 수
        minLineLength=40,     # 선으로 인정할 최소 길이
        maxLineGap=20         # 끊어진 선 사이의 최대 허용 간격
    )

    # 6. 검출된 직선을 원본 이미지에 빨간색으로 그리기
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 빨간색 (BGR 포맷: Blue=0, Green=0, Red=255)
            cv2.line(copy_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)

    # 결과를 임시 이미지에 복사 (디버깅 및 확인용)
    copy_img = cv2.cvtColor(copy_img, cv2.COLOR_BGR2RGB)

    # (Optional: 현재 프레임을 출력하거나 저장하려면 여기서 코드를 추가합니다.)
    # img를 imshow로 보여주기
    cv2.imshow('Lane Detection', copy_img)

    # 프레임 건너뛰기 (테스트 용도)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# 작업 완료 후 리소스 해제
cv2.destroyAllWindows()