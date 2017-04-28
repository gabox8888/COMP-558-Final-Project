import cv2
from skimage import measure
import matplotlib.pyplot as plt


def main():
    image = cv2.imread('../data/imagenet/im2.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(image,(5,25),0)
    edges = cv2.Canny(blur,100,200)
    ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    image_grey,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    tam = 0

    for contorno in contours:
        cv2.drawContours(image,contorno.astype('int'),-1,(0,255,0),2)
    cv2.imshow('My image',image)

    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__': main()