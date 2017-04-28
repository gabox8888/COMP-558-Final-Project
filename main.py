import pcl

def main():
    rgb = '../data/im1/rgbImage_1.jpg'
    depth = '../data/im1/depthImage_1.png'
    intr = '../data/im1/intrinsics_1.txt'
    extr = '../data/im1/extrinsic_1.txt'

    pcl_obj = pcl.pcl(rgb,depth,intr,extr)
    # pcl_obj.plot()
    pcl_obj.save_csv()
if __name__ == '__main__':main()