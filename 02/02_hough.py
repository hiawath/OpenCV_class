import cv2
import numpy as np

# 1. 이미지 로드
img = cv2.imread('./temp/dashcam_sample.jpg')
copy_img = img.copy()

# 2. 전처리: 그레이스케일 변환 및 가우시안 블러 (노이즈 제거)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 3. Canny Edge 검출
# 픽셀 값의 변화가 급격한 부분(차선과 도로의 경계)을 얇은 선으로 추출
edges = cv2.Canny(blur, 50, 150)

# 4. ROI (관심 영역) 설정 - 블랙박스 영상의 핵심
# 하늘이나 주변 건물에서 검출된 불필요한 엣지를 제거하기 위해 도로 영역만 마스킹
height, width = edges.shape
mask = np.zeros_like(edges)
# 도로 부분에 해당하는 사다리꼴 다각형 좌표
polygon = np.array([[(0, height), (350, 350), (450, 350), (width, height)]], np.int32)
cv2.fillPoly(mask, polygon, 255)
# 엣지 이미지와 마스크를 비트 AND 연산하여 도로 쪽 엣지만 남김
roi_edges = cv2.bitwise_and(edges, mask)

# 5. 확률적 허프 변환 (HoughLinesP) 적용
# 엣지 픽셀들을 조합하여 직선 성분을 찾아냄
lines = cv2.HoughLinesP(
    roi_edges, 
    rho=1,                # 거리 해상도 (픽셀 단위)
    theta=np.pi/180,      # 각도 해상도 (라디안 단위, 1도)
    threshold=30,         # 직선으로 판단할 최소 엣지 점의 수(교차점)
    minLineLength=40,     # 선으로 인정할 최소 길이 (점선을 선으로 인식하기 위함)
    maxLineGap=20         # 끊어진 선 사이의 최대 허용 간격 (점선을 이어주기 위함)
)

# 6. 검출된 직선을 원본 이미지에 빨간색으로 그리기
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(copy_img, (x1, y1), (x2, y2), (0, 0, 255), 3) # 빨간색(BGR), 두께 3

# 7. 결과 시각화
cv2.imshow('Original', img)
cv2.imshow('Canny Edges', edges)
cv2.imshow('ROI Edges', roi_edges)
cv2.imshow('Lane Detection', copy_img)

cv2.waitKey(0)
cv2.destroyAllWindows()