import numpy as np
import numpy.linalg as ln

def main():
    matrix = np.load('../../data/new_data/matrices/adjacency/a_18077.npy')
    weights = np.load('../../data/new_data/matrices/features/f_18077.npy')

    for i,x in enumerate(matrix):
        for j,y in enumerate(x):
            matrix[i,j] *= weights[j]

    eigen = ln.eig(matrix)

    print(matrix[0])
    print(eigen[1][0])
    print(eigen[0])


if __name__ == '__main__':main()