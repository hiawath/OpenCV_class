import cv2
import numpy as np

# 1. 두 이미지 로드
img_left = cv2.imread('./temp/pano_left.jpg')
img_right = cv2.imread('pano_right.jpg')

# 그레이스케일 변환 (특징점 추출은 흑백 이미지에서 수행하는 것이 효율적)
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# 2. ORB 특징점 검출기 초기화
orb = cv2.ORB_create()

# 특징점(Keypoints)과 디스크립터(Descriptors) 계산
kp1, des1 = orb.detectAndCompute(gray_right, None) # 기준 이미지를 오른쪽으로 설정
kp2, des2 = orb.detectAndCompute(gray_left, None)  # 붙일 이미지를 왼쪽으로 설정

# 3. BFMatcher (Brute-Force Matcher) 생성
# ORB는 이진 디스크립터를 사용하므로 거리 측정 방식으로 NORM_HAMMING을 사용해야 함
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 디스크립터 매칭 및 거리에 따라 정렬 (거리가 짧을수록 유사도가 높음)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# 매칭 결과 시각화 (상위 50개만)
match_img = cv2.drawMatches(img_right, kp1, img_left, kp2, matches[:50], None, flags=2)
cv2.imshow('Feature Matching', match_img)

# 4. 호모그래피(Homography) 계산을 위한 좌표 추출
# 매칭된 점들이 4개 이상이어야 원근 변환 행렬을 구할 수 있음 (상위 50개 사용)
if len(matches) > 4:
    # 기준(오른쪽) 이미지의 특징점 좌표
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
    # 변환(왼쪽) 이미지의 특징점 좌표
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)

    # 5. RANSAC 알고리즘을 이용해 호모그래피 행렬(H) 도출
    # H: 오른쪽 이미지를 왼쪽 이미지의 시점으로 변환하는 3x3 행렬
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 6. 이미지 합성 (Warp Perspective)
    # 변환될 캔버스의 너비는 두 이미지 너비의 합으로 충분히 크게 설정
    width = img_right.shape[1] + img_left.shape[1]
    height = img_right.shape[0]

    # 오른쪽 이미지를 왼쪽 이미지 시점으로 원근 변환하여 캔버스에 붙임
    panorama = cv2.warpPerspective(img_right, H, (width, height))

    # 캔버스의 왼쪽 빈 공간에 원본 왼쪽 이미지를 덮어씌움
    panorama[0:img_left.shape[0], 0:img_left.shape[1]] = img_left

    cv2.imshow('Panorama Result', panorama)
else:
    print("특징점이 충분하지 않아 파노라마를 생성할 수 없습니다.")

cv2.waitKey(0)
cv2.destroyAllWindows()