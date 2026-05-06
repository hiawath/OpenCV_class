import cv2
import matplotlib
matplotlib.use('TkAgg') # 대화형 백엔드 설정
import matplotlib.pyplot as plt

print(cv2.__version__)

#이미지 불러오기
lena_img=cv2.imread("images/lena.jpg")
if lena_img is None:
    # Recommendation: Always use the Guard Clause
    raise FileNotFoundError("Could not load image. Check if the path 'lena.jpg' is correct.")

# 3. Now Pylance knows 'lena_img' is definitely an ndarray and not None
print(lena_img.shape)

type(lena_img)

fname = 'images/lena.jpg'
color= cv2.imread(fname, cv2.IMREAD_COLOR)
gray=cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
unchanged =cv2.imread(fname, cv2.IMREAD_UNCHANGED)


if (color, gray, unchanged) is None:
    raise 
print(gray.shape)

#이미지 출력하기
#cv2.imshow('Color', color)
#cv2.imshow('Grayscale', gray)
#cv2.imshow('Unchanged', unchanged)


cv2.waitKey()
cv2.destroyAllWindows()

#이미지 저장하기
img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
cv2.imwrite('./temp/lenagray.png',img)


#matplotlib으로 그리기
img = cv2.imread('images/lena.jpg', cv2.IMREAD_COLOR)

b, g, r = cv2.split(img)
img2 = cv2.merge([r, g, b])

#plt.imshow(img2)
#plt.show()

#컬러 공간 변환
flags = [flag for flag in dir(cv2) if flag.startswith('COLOR_')]
print(flags)

B,G, R=lena_img[0][0]
print("color pixel", lena_img[0][0])

print("YPrPb", 0.114*B + 0.587*G + 0.299*R)
print("YCrCb", 0.0722*B + 0.7152*G + 0.2116*R)
lena_gray = cv2.imread("images/lena.jpg", cv2.IMREAD_GRAYSCALE)
print("그레이스케일 화소", lena_gray[0][0])

#동영상 읽기
cap = cv2.VideoCapture("images/Puppies-HD.mp4")

if cap.isOpened():
    print(cap.get(cv2.CAP_PROP_FPS))
    delay = int(1000 / cap.get(cv2.CAP_PROP_FPS))
    while True:
        ret, img = cap.read()
        if ret:
            cv2.imshow("Movie", img)
            if cv2.waitKey(delay) & 0xFF == 27 : # ESC키
                print("ESC Key pressed")
                break
        else:
            print("No Frame")
            print(ret, img)
            break
        
else:
    print("File not opened")

cap.release()
cv2.destroyAllWindows()

#카메라 영상 읽기
import cv2

cap = cv2.VideoCapture(0)

if cap.isOpened():
    print(cap.get(cv2.CAP_PROP_FPS))
    delay = int(1000 / cap.get(cv2.CAP_PROP_FPS))
    while True:
        ret, img = cap.read()
        if ret:
            # 영상처리 코드
            cv2.imshow("Movie", cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            if cv2.waitKey(delay) & 0xFF == 27 : # ESC키
                print("ESC Key pressed")
                break
        else:
            print("No Frame")
            print(ret, img)
            break
        
else:
    print("File not opened")

cap.release()
cv2.destroyAllWindows()

#동영상 속성
import cv2

cap = cv2.VideoCapture(0)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if cap.isOpened():
    print(cap.get(cv2.CAP_PROP_FPS))
    delay = int(1000 / cap.get(cv2.CAP_PROP_FPS))
    while True:
        ret, img = cap.read()
        if ret:
            cv2.imshow("Movie", img)
            if cv2.waitKey(delay) & 0xFF == 27 : # ESC키
                print("ESC Key pressed")
                break
        else:
            print("No Frame")
            print(ret, img)
            break
        
else:
    print("File not opened")

cap.release()
cv2.destroyAllWindows()

#동영상 프레임 저장하기
import cv2
import numpy as np
from datetime import datetime

def mouseHandler(event, x, y, flags, param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print(event, x, y)
        print(datetime.today())
        filename="./temp/"+str(datetime.today().microsecond)+".jpg"
        cv2.imwrite(filename, img)

cv2.namedWindow('Movie')
cv2.setMouseCallback("Movie", mouseHandler)

cap = cv2.VideoCapture("images/Puppies-HD.mp4")

img = None
if cap.isOpened():

    print(cap.get(cv2.CAP_PROP_FPS))
    delay = int(1000 / cap.get(cv2.CAP_PROP_FPS))
    while True:
        ret, img = cap.read()
        if ret:
            cv2.imshow("Movie", img)
            if cv2.waitKey(delay) & 0xFF == 27 : # ESC키
                print("ESC Key pressed")
                break
        else:
            print("No Frame")
            print(ret, img)
            break
        
else:
    print("File not opened")

cap.release()
cv2.destroyAllWindows()

#비디오 저장하기
import cv2

cap = cv2.VideoCapture("images/Puppies-HD.mp4")

if cap.isOpened():
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') # DIVX, XVID, FMP4, X264, MJPG
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (int(width), int(height))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter("./temp/video.avi", fourcc, fps, size)

    delay = int(1000/cap.get(cv2.CAP_PROP_FPS))
    print(width, height, size, fps, delay)

    while True:
        ret, img = cap.read()
        if ret:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            cv2.imshow("Movie", gray)
            out.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
            if cv2.waitKey(delay) & 0xFF == 27:
                print("윈도우 종료")
                break
        else:
            print(ret, img)
            break
else:
    print("비디오 안열림")

out.release()
cap.release()
cv2.destroyAllWindows()

#마우스 이벤트

