import cv2
import numpy as np

# 1. 웹 화면 스크린샷(원본 이미지) 생성
width, height = 800, 600
img = np.ones((height, width, 3), dtype=np.uint8) * 240 # 밝은 회색 배경

# 헤더(Header) 영역
cv2.rectangle(img, (0, 0), (width, 80), (200, 200, 200), -1)
cv2.putText(img, "MyWebSite", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 3)

# 사이드바(Sidebar) 영역
cv2.rectangle(img, (0, 80), (200, height), (220, 220, 220), -1)
for i in range(120, 500, 50):
    cv2.rectangle(img, (20, i), (180, i+30), (180, 180, 180), -1) # 메뉴 항목들

# 본문 더미 콘텐츠
for i in range(150, 550, 60):
    cv2.line(img, (250, i), (750, i), (190, 190, 190), 15)
    cv2.line(img, (250, i+25), (600, i+25), (200, 200, 200), 10)

# 가짜 버튼 (오답 유도용)
cv2.rectangle(img, (500, 20), (620, 60), (150, 150, 150), -1)
cv2.putText(img, "SIGN UP", (515, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# 타겟 버튼 (찾아야 할 로고/버튼)
button_x, button_y = 650, 20
button_w, button_h = 120, 40
cv2.rectangle(img, (button_x, button_y), (button_x + button_w, button_y + button_h), (50, 150, 255), -1) # 주황색 버튼
cv2.putText(img, "LOGIN", (button_x + 25, button_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# 메인 이미지 저장
cv2.imwrite('./temp/web_screenshot.jpg', img)

# 2. 템플릿 이미지 (찾을 대상) 잘라내어 저장
template = img[button_y:button_y+button_h, button_x:button_x+button_w]
cv2.imwrite('./temp/button_template.jpg', template)

print("Generated: web_screenshot.jpg, button_template.jpg")