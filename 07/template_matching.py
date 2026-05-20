import cv2
import numpy as np

# 1. 이미지 로드 (연산 속도와 정확도를 위해 보통 그레이스케일로 변환하여 진행)
main_img = cv2.imread('./temp/web_screenshot.jpg')
main_gray = cv2.cvtColor(main_img, cv2.COLOR_BGR2GRAY)

template = cv2.imread('./temp/button_template.jpg')
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# 템플릿 이미지의 가로, 세로 길이 추출
h, w = template_gray.shape

# 2. 템플릿 매칭 연산 수행
# cv2.TM_CCOEFF_NORMED: 정규화된 상관계수 매칭 방법. (가장 신뢰도가 높음, 1.0에 가까울수록 일치)
res = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED)

# 3. 유사도 맵(res)에서 가장 값이 높은(유사한) 위치 찾기
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# 4. 검출 로직 (오탐 방지를 위한 임계값 설정)
threshold = 0.8  # 유사도가 80% 이상일 때만 찾은 것으로 간주

if max_val >= threshold:
    print(f"버튼 발견! (유사도: {max_val:.2f})")

    # max_loc은 매칭된 영역의 좌측 상단 (x, y) 좌표를 의미함
    top_left = max_loc
    # 우측 하단 좌표 계산 (좌측 상단 + 템플릿의 너비/높이)
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # 원본 이미지에 매칭 결과 사각형 그리기 (빨간색, 두께 3)
    cv2.rectangle(main_img, top_left, bottom_right, (0, 0, 255), 3)

    # 템플릿 위에 이름표 붙이기
    cv2.putText(main_img, f"Found: {max_val:.2f}", (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
else:
    print("화면에서 해당 버튼을 찾을 수 없습니다.")

# 5. 결과 시각화
cv2.imshow('Template Matching Result', main_img)
cv2.waitKey(0)
cv2.destroyAllWindows()