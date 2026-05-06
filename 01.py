import cv2
from matplotlib import pyplot as plt

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

plt.imshow(img2)
plt.imshow(img)
plt.show()