import cv2
import numpy as np

# 1. 원본 바이너리 이미지 생성 (검은 배경, 흰색 사각형)
img = np.zeros((400, 400), dtype=np.uint8)
cv2.rectangle(img, (100, 100), (300, 300), 255, -1)

# 2. Salt & Pepper 노이즈 추가
noisy_img = img.copy()
noise = np.random.rand(400, 400)

# Pepper 노이즈 (객체 내부의 검은 점)
noisy_img[noise < 0.05] = 0
# Salt 노이즈 (배경의 흰 점)
noisy_img[noise > 0.95] = 255

# 파일 저장
cv2.imwrite('morphology_test.jpg', noisy_img)
print("Generated: morphology_test.jpg")