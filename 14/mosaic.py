import cv2
import numpy as np

def apply_full_face_mosaic(image_path):
    # 1. 두 가지 Cascade 모델 로드 (정면, 측면)
    frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

    img = cv2.imread(image_path)
    if img is None:
        print("이미지 로드 실패")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    all_faces = [] # 검출된 모든 얼굴의 좌표(x, y, w, h)를 담을 리스트

    # --- [탐지 1] 정면 얼굴 검출 ---
    frontal_faces = frontal_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    for face in frontal_faces:
        all_faces.append(face)

    # --- [탐지 2] 왼쪽 옆모습 검출 ---
    left_profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    for face in left_profiles:
        all_faces.append(face)

    # --- [탐지 3] 오른쪽 옆모습 검출 (Flip Trick) ---
    # 이미지를 좌우 반전(flipCode=1)하여 오른쪽 옆모습이 왼쪽을 향하게 만듦
    flipped_gray = cv2.flip(gray, 1)
    right_profiles = profile_cascade.detectMultiScale(flipped_gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    
    for (x, y, w, h) in right_profiles:
        # 반전된 이미지에서 찾은 X 좌표를 원본 이미지의 X 좌표로 복구
        original_x = width - (x + w)
        all_faces.append((original_x, y, w, h))

    print(f"총 {len(all_faces)}개의 얼굴(정면/측면)이 검출되었습니다.")

    # 2. 수집된 모든 얼굴 좌표에 모자이크(블러) 적용
    for (x, y, w, h) in all_faces:
        # 화면 밖으로 좌표가 넘어가는 것을 방지 (안전 장치)
        x, y = max(0, x), max(0, y)
        
        face_roi = img[y:y+h, x:x+w]
        
        # 가우시안 블러 적용 (30, 30)
        blurred_face = cv2.blur(face_roi, (30, 30))
        img[y:y+h, x:x+w] = blurred_face
        
        # 검출 박스 표시 (선택)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2) # 노란색 테두리

    # 3. 결과 출력
    cv2.imshow('Full Face Mosaic', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "./images/crowd_image.png" # 테스트할 이미지 경로
    apply_full_face_mosaic(image_path)