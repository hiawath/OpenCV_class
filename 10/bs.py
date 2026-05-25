import cv2
import numpy as np

# 1. 시뮬레이션 비디오 데이터 로드 (30프레임 이미지 시퀀스)
# 실습의 편의를 위해 넘파이 파일로 제공되는 영상을 순회합니다.
video_frames = np.load('./temp/cctv_simulation.npy')

# 2. MOG2 배경 분리 패키지 생성
# history: 배경 학습에 반영할 이전 프레임 개수 (기본 500)
# varThreshold: 가우시안 혼합 모델에서 마하노비스 거리를 이용해 전경/배경을 판단할 임계값 (기본 16)
# detectShadows: 그림자 감지 여부 (True로 설정 시 그림자를 회색(127)으로 표시하여 분리해줌)
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=True)

print("CCTV 모니터링을 시작합니다... (아무 키나 누르면 다음 프레임으로 이동)")

for idx, frame in enumerate(video_frames):
    # 3. 현재 프레임에서 전경 마스크(Foreground Mask) 추출
    # fgbg.apply()는 자동으로 내부 배경 모델을 누적 학습하며 흑백 마스크를 반환함
    fg_mask = fgbg.apply(frame)

    # 4. 모폴로지 연산(열림/닫힘)을 통한 마스크 노이즈 제거
    # 카메라 센서 노이즈로 인해 자잘하게 튄 흰색 점들을 지우고, 객체 내부 구멍을 메움
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # 5. 전경 마스크에서 객체의 외곽선(Contours) 검출
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 원본 이미지 복사본에 시각화 진행
    display_frame = frame.copy()

    for contour in contours:
        # 노이즈 성격의 너무 작은 움직임(예: 면적 400 이하)은 무시하고 패스
        if cv2.contourArea(contour) < 400:
            continue

        # 외곽선을 감싸는 최소 크기의 사각형(Bounding Box) 좌표 구하기
        x, y, w, h = cv2.boundingRect(contour)

        # 침입자 발견 시 빨간색 사각형과 'Warning' 텍스트 표시
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(display_frame, "INTRUDER DETECTED", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 6. 모니터링 화면 출력 (원본, 전경 마스크)
    cv2.imshow('CCTV Live Monitor', display_frame)
    cv2.imshow('Foreground Mask (MOG2)', fg_mask)

    # 실습 관찰을 위해 150ms 대기 (0으로 두면 키를 누를 때마다 한 프레임씩 이동)
    if cv2.waitKey(150) & 0xFF == 27: # ESC 누르면 종료
        break

cv2.destroyAllWindows()