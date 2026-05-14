import cv2
import numpy as np

# 1. 예제 블랙박스 주행 영상(프레임) 생성
width, height = 800, 600
img = np.ones((height, width, 3), dtype=np.uint8) * 180  # 배경 (하늘 등)

# 도로 그리기 (사다리꼴, 원근감 표현)
road_pts = np.array([[350, 350], [450, 350], [800, 600], [0, 600]], np.int32)
cv2.fillPoly(img, [road_pts], (80, 80, 80)) # 어두운 회색 도로

# 차선 그리기
# 왼쪽: 흰색 실선
cv2.line(img, (330, 350), (100, 600), (255, 255, 255), 8)
# 오른쪽: 흰색 점선
for y in range(350, 600, 40):
    x1 = int(470 + (y - 350) * 1.3)
    x2 = int(470 + (y + 20 - 350) * 1.3)
    if y + 20 <= 600:
        cv2.line(img, (x1, y), (x2, y + 20), (255, 255, 255), 8)

# 노이즈 추가 (가로수, 건물 등 차선이 아닌 직선 성분)
cv2.rectangle(img, (50, 200), (120, 450), (50, 100, 50), -1) # 좌측 가로수
cv2.rectangle(img, (650, 150), (750, 400), (100, 100, 120), -1) # 우측 건물

# 가우시안 블러 및 약간의 노이즈 추가로 현실감 부여
img = cv2.GaussianBlur(img, (5, 5), 0)
noise = np.random.normal(0, 5, (height, width, 3))
final_img = np.clip(img + noise, 0, 255).astype(np.uint8)

# 파일 저장
img_path = './temp/dashcam_sample.jpg'
cv2.imwrite(img_path, final_img)
print(f"Generated: {img_path}")