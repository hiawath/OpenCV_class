import cv2
import numpy as np

# 1. 노이즈가 포함된 이진 이미지 로드 (항상 그레이스케일로 읽기)
# 실습 시 위에서 생성한 'morphology_test.jpg'를 사용합니다.
img = cv2.imread('./images/morphology_test.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("이미지 로드 실패")
    exit()

# 2. 커널(구조 요소) 생성
# 5x5 크기의 사각형 커널 사용 (노이즈 크기에 따라 크기 조절 필요)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) # (3,3)이 더 명확함
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)) # (3,3)이 더 명확함
# 3. 열림 연산 (Opening): 침식(Erosion) -> 팽창(Dilation)
# 효과: 배경(검은색)에 있는 자잘한 흰색 점(Salt 노이즈)을 제거합니다.
opened_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# 4. 닫힘 연산 (Closing): 팽창(Dilation) -> 침식(Erosion)
# 효과: 객체(흰색) 내부에 있는 검은색 구멍(Pepper 노이즈)을 메웁니다.
# 주의: 열림 연산이 끝난 이미지(opened_img)에 이어서 적용해야 합니다.
final_img = cv2.morphologyEx(opened_img, cv2.MORPH_CLOSE, kernel)

# 5. 결과 시각화 비교
cv2.imshow('1. Noisy Original', img)
cv2.imshow('2. After Opening (Salt Removed)', opened_img)
cv2.imshow('3. After Closing (Pepper Removed)', final_img)

cv2.waitKey(0)
cv2.destroyAllWindows()