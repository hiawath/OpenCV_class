import cv2
import numpy as np
import sys

# 1. 실제 MP4 동영상 파일 열기
video_path = './temp/drive_yy.mp4'  # 실습용 mp4 파일 경로를 지정하세요
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"오류: 동영상 파일({video_path})을 열 수 없습니다.")
    sys.exit()

# 동영상의 FPS(초당 프레임) 정보 가져오기 및 재생 속도(delay) 계산
fps = cap.get(cv2.CAP_PROP_FPS)
delay = max(1, int(1000 / fps))

# 루카스-카나데 및 특징점 검출 파라미터 셋업
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# 2. 첫 번째 프레임 읽기 및 초기 추적점(p0) 세팅
ret, old_frame = cap.read()
if not ret:
    print("동영상의 첫 프레임을 읽을 수 없습니다.")
    sys.exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# 최초 특징점(코너) 검출
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# 궤적(선)을 누적해서 그릴 투명한 마스크 캔버스 생성
mask = np.zeros_like(old_frame)

print("광학 흐름 추적을 시작합니다... (ESC 키를 누르면 종료)")

# 3. 무한 루프를 돌며 실시간 영상 처리
while True:
    ret, frame = cap.read()
    
    # 영상이 끝나면 루프 탈출
    if not ret:
        print("동영상 재생이 완료되었습니다.")
        break
        
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # [핵심] 추적 중인 점(p0)이 화면에 남아 있을 때만 연산 수행
    if p0 is not None and len(p0) > 0:
        
        # 루카스-카나데 알고리즘으로 다음 프레임에서의 위치(p1) 계산
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        # 추적에 성공한(상태값 st==1) 점들만 골라냄
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        
        # 이동 궤적 시각화
        for j, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            
            # 이전 점과 현재 점을 이어주는 궤적 그리기 (마스크 캔버스에 누적)
            mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
            # 현재 위치에 동그라미 그리기 (현재 프레임 캔버스에만)
            frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)
            
        # 다음 프레임 연산을 위해 과거 데이터 업데이트
        old_gray = frame_gray.copy()
        # [주의] reshape(-1, 1, 2)를 통해 OpenCV가 요구하는 다차원 배열 형태로 맞춤
        p0 = good_new.reshape(-1, 1, 2)
        
    # 원본 프레임과 궤적이 누적된 마스크를 합성
    img = cv2.add(frame, mask)
    
    cv2.imshow('Optical Flow (MP4 Video)', img)
    
    # 원본 동영상 프레임 레이트(FPS)에 맞춘 딜레이 처리
    if cv2.waitKey(delay) & 0xFF == 27:
        break

# 4. 자원 해제
cap.release()
cv2.destroyAllWindows()