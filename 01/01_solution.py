import cv2

# 1. 이미지 로드 (반드시 그레이스케일로 읽어야 함)
img = cv2.imread('./temp/uneven_document.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("이미지 로드 실패. 경로를 확인하세요.")
    exit()

# 2. 전역 이진화 (비교용: 조명 문제로 인해 실패하는 사례 시연)
_, global_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 3. 적응형 이진화 (해결책)
# 파라미터 튜닝: 블록 크기 51, 평균에서 뺄 상수 15 적용
adaptive_thresh = cv2.adaptiveThreshold(
    img,                                # 입력 이미지
    255,                                # 최댓값 (흰색)
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,     # 적응형 이진화 방식 (가우시안 가중치 평균)
    cv2.THRESH_BINARY,                  # 이진화 타입
    51,                                 # blockSize (홀수)
    15                                  # C (차감 상수)
)

# 4. 결과 비교 시각화
cv2.imshow('Original (Uneven Lighting)', img)
cv2.imshow('Global Thresholding (Failed)', global_thresh)
cv2.imshow('Adaptive Thresholding (Success)', adaptive_thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()