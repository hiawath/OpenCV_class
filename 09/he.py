import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 저조도 이미지 로드 (그레이스케일)
img = cv2.imread('./temp/night_drive.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("이미지 로드 실패")
    exit()

# 2. 전역 히스토그램 평활화 (Global Histogram Equalization)
# 이미지 전체의 명암 분포를 강제로 0~255로 쫙 펼쳐줍니다.
global_equalized = cv2.equalizeHist(img)

# 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
# 이미지를 격자(Grid)로 쪼개어 지역적으로 평활화를 하되, 노이즈가 증폭되는 것을 제한합니다.
# clipLimit: 대비 한계치 (값이 클수록 대비가 강해지나 노이즈도 증가), tileGridSize: 쪼갤 영역 크기
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
clahe_equalized = clahe.apply(img)

# 4. 결과 이미지 시각화 (OpenCV 창)
cv2.imshow('1. Original Night Video', img)
cv2.imshow('2. Global Equalization (Noisy)', global_equalized)
cv2.imshow('3. CLAHE (Restored)', clahe_equalized)

# 5. [평가 핵심] 전후 히스토그램 분포 분석 및 시각화 (Matplotlib)
plt.figure(figsize=(15, 5))

# 원본 히스토그램 (어두운 영역에 극단적으로 몰려있음)
plt.subplot(1, 3, 1)
plt.title("Original Histogram")
plt.hist(img.ravel(), 256, [0, 256], color='gray')
plt.xlim([0, 256])

# 전역 평활화 히스토그램 (그래프가 듬성듬성 찢어지며 전체로 펼쳐짐)
plt.subplot(1, 3, 2)
plt.title("Global Equalized")
plt.hist(global_equalized.ravel(), 256, [0, 256], color='blue')
plt.xlim([0, 256])

# CLAHE 히스토그램 (지역별 밸런스가 조절되어 고르게 분포됨)
plt.subplot(1, 3, 3)
plt.title("CLAHE Equalized")
plt.hist(clahe_equalized.ravel(), 256, [0, 256], color='green')
plt.xlim([0, 256])

plt.tight_layout()
plt.show(block=True) 


cv2.waitKey(0)
cv2.destroyAllWindows()
