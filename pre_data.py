import numpy as np
import cv2

def main():
    X = np.load('../../data/tiny/tinyX.npy')
    Y = np.load('../../data/tiny/tinyY.npy')
    for i,im in enumerate(X):
        name = '../../data/new_data/im{}.png'.format(i)
        b = np.zeros([64,64,3])
        b[:,:,0] = im[0]
        b[:,:,1] = im[1]
        b[:,:,2] = im[2]
        cv2.imwrite(name,b)

if __name__ == '__main__':main()