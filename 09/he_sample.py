import cv2
import numpy as np

# 1. 야간/저조도 시뮬레이션 예제 이미지 생성
width, height = 800, 600
img = np.zeros((height, width), dtype=np.uint8)

# 도로 배경 (아주 어둡게 처리)
cv2.rectangle(img, (0, 0), (width, height), 20, -1)

# 도로 차선 (어두운 회색)
cv2.line(img, (100, 600), (350, 350), 60, 5)
cv2.line(img, (700, 600), (450, 350), 60, 5)

# 어둠 속에 숨겨진 객체들 (사람, 차량 실루엣, 표지판)
# 보행자 형태 (어두운 회색으로 묻히게 설정)
cv2.circle(img, (250, 420), 15, 45, -1) # 머리
cv2.rectangle(img, (235, 435), (265, 500), 40, -1) # 몸통

# 도로 표지판 모양
cv2.rectangle(img, (550, 300), (650, 380), 50, -1)
cv2.line(img, (600, 380), (600, 500), 35, 8)
cv2.putText(img, "STOP", (565, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 30, 2)

# 가우시안 노이즈 추가로 야간 카메라 특성 반영
noise = np.random.normal(0, 3, (height, width))
final_img = np.clip(img + noise, 0, 255).astype(np.uint8)

# 파일 저장
cv2.imwrite('./temp/night_drive.jpg', final_img)
print("Generated: night_drive.jpg")