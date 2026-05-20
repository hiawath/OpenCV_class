import cv2
import numpy as np

# 1. 파노라마용 가상 배경(전체 씬) 생성 (1000x500)
canvas = np.ones((500, 1000, 3), dtype=np.uint8) * 200

# 하늘과 땅
cv2.rectangle(canvas, (0, 0), (1000, 250), (255, 200, 150), -1) # 하늘
cv2.rectangle(canvas, (0, 250), (1000, 500), (100, 200, 100), -1) # 땅

# 특징점(Keypoints)이 잘 잡히도록 디테일한 요소(건물, 창문) 추가
cv2.circle(canvas, (500, 100), 50, (0, 100, 255), -1) # 태양 (중앙)

for x in range(100, 900, 150):
    cv2.rectangle(canvas, (x, 150), (x+100, 400), (50, 50, 50), -1) # 건물
    for wx in range(x+10, x+90, 30):
        for wy in range(170, 380, 40):
            cv2.rectangle(canvas, (wx, wy), (wx+20, wy+20), (0, 255, 255), -1) # 창문

# 땅 영역에 가우시안 노이즈를 추가하여 텍스처(특징점) 생성
noise = np.random.randint(0, 50, (250, 1000, 3), dtype=np.uint8)
canvas[250:500, :] = cv2.add(canvas[250:500, :], noise)

# 2. 겹치는 영역을 두어 두 장의 사진으로 분할 (크롭)
# 왼쪽 사진 (가로 0 ~ 600)
img_left = canvas[:, 0:600].copy()
# 오른쪽 사진 (가로 400 ~ 1000, 즉 200픽셀 겹침)
img_right = canvas[:, 400:1000].copy()

cv2.imwrite('./temp/pano_left.jpg', img_left)
cv2.imwrite('./temp/pano_right.jpg', img_right)
print("Generated: pano_left.jpg, pano_right.jpg")