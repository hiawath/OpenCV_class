# 예제 이미지 생성
import cv2
import numpy as np

# 1. 예제 이미지 생성 (불균일한 조명 문서)
width, height = 800, 500
img = np.ones((height, width), dtype=np.uint8) * 255

# 텍스트 삽입
texts = [
    "OpenCV Computer Vision",
    "Adaptive Thresholding Test",
    "Problem: Severe Uneven Lighting",
    "Goal: Extract Text Clearly"
]
for i, text in enumerate(texts):
    cv2.putText(img, text, (40, 100 + i * 110), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 4)

# 불균일한 조명 맵 생성 (좌측 상단 밝음, 우측 하단 어두움)
X, Y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
illumination = 1.0 - (X * 0.9 + Y * 0.1)
illumination = np.clip(illumination, 0.1, 1.0)

# 조명 적용 및 가우시안 노이즈 추가
uneven_img = (img * illumination).astype(np.uint8)
noise = np.random.normal(0, 5, (height, width))
final_img = np.clip(uneven_img + noise, 0, 255).astype(np.uint8)

# 파일 저장
img_path = './temp/uneven_document.jpg'
cv2.imwrite(img_path, final_img)
print(f"Generated: {img_path}")