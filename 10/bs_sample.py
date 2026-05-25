import cv2
import numpy as np

# 1. 고정된 CCTV 실시간 상황 시뮬레이션 영상 생성 (배경 + 움직이는 물체)
# 30프레임짜리 비디오를 생성하기 위한 이미지 시퀀스 만들기
width, height = 800, 600
frames = []

# 기본 고정 배경 (벽, 바닥, 고정된 화분)
bg = np.ones((height, width, 3), dtype=np.uint8) * 150 # 회색 바닥
cv2.rectangle(bg, (0, 0), (width, 200), (90, 90, 90), -1) # 어두운 벽
cv2.rectangle(bg, (50, 150), (100, 250), (50, 100, 50), -1) # 고정된 화분

# 30프레임 동안 움직이는 객체(침입자 사각형)가 좌에서 우로 이동
for f in range(30):
    frame = bg.copy()

    # 침입자 (프레임마다 X 좌표가 이동함)
    obj_x = 50 + f * 22
    obj_y = 300
    cv2.rectangle(frame, (obj_x, obj_y), (obj_x + 60, obj_y + 120), (30, 30, 30), -1) # 침입자 실루엣

    # 미세한 카메라 센서 노이즈 추가
    noise = np.random.normal(0, 2, (height, width, 3))
    frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
    frames.append(frame)

# 연속 실행을 확인하기 위해 비디오 파일로 저장 시도 (동영상 파일 대신 이미지 시퀀스로 처리할 수 있게 분할 보존)
# 학생 실습의 편의를 위해 30장의 이미지를 넘파이 배열 하나로 묶어서 동영상처럼 연출하도록 유도
np.save('./temp/cctv_simulation.npy', np.array(frames))
print("Generated: cctv_simulation.npy (30 frames of video data)")