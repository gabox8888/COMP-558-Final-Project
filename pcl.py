import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import path
import cv2
import csv

class pcl:
    def __init__(self,rgb_path,depth_path,intrinsic_path,extrinsic_path):
        self.rgb_image = cv2.imread(rgb_path)
        self.depth_image = cv2.imread(depth_path)
        self.intrinsic = self.read_matrix(intrinsic_path)
        self.extrinsic = self.read_matrix(extrinsic_path)
        self.P = np.dot(self.intrinsic,self.extrinsic)
        self.P = self.P[0:3,0:3]
        self.Pinv = np.linalg.inv(self.P)
        self.point_cloud = self.generate_pcl()

    def read_matrix(self,path):
        matrix = []
        with open(path) as f:
            content = f.readlines()
        for line in content:
            arr = list(map(lambda x: float(x), line.split(' ')))
            matrix.append(arr)
        return np.array(matrix)

    def generate_pcl(self):
        point_cloud = []
        for i,x in enumerate(self.rgb_image):
            for k,y in enumerate(x):
                temp = [float(self.depth_image[i,k,0]) * i,float(self.depth_image[i,k,0]) * k,float(self.depth_image[i,k,0])]
                temp = np.array(temp)
                temp = np.dot(self.Pinv,temp)
                y = list(map( lambda x : float(x)/255.0,y))
                point_cloud.append([tuple(temp),tuple(y)])
        return point_cloud
    
    def save_csv(self):
        with open('../data/im1/point_cloud.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in self.point_cloud:
                row = list(i[0]) + list(i[1])
                writer.writerow(row)

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in self.point_cloud:
            xyz = i[0]
            rgb = i[1]
            ax.scatter(xyz[0],xyz[1],xyz[2],c=rgb)
        plt.show()