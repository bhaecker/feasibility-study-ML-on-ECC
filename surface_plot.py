import sys
import math
import pandas as pd
import pickle
import numpy as np
import collections

from dataset_creation import curve_points, arrays

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

"""
Plot the surface of y=x*k, where x and y are the id´s of the points 
"""

#set curve parameters
a,b,p = 2,3,7

x_points,k_points,y_points =  arrays(a,b,p)
X, K = np.meshgrid(x_points, k_points)

def getY_slow(X,K,x_points,k_points,y_points):
    """
    get y values for 3d plot
    """
    Y = np.full_like(X, -1)
    for i in range(np.shape(Y)[0]):
        for j in range(np.shape(Y)[1]):
            for k in range(len(x_points)):
                if X[i,j] == x_points[k] and K[i,j] == k_points[k]:
                    Y[i,j] = y_points[k]
    return(Y)

def getY_fast(X,K,x_points,k_points,y_points):
    #todo: doesnt work yet, figure out how to use the indicies to get the corresponding entry in Y
    Y = np.full_like(X, -1)
    for row in range(int((np.shape(Y)[0]))):
        for column in range(int((np.shape(Y)[1]))):
            print(row,column,((math.floor(row/3)*3))+(column%3))
            Y[row,column] = y_points[((math.floor(row/3)*3))+(column%3)]
    return('not usuable atm')


#plot the function in 3d
Y = getY_slow(X,K,x_points,k_points,y_points)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, K, Y, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('k')
ax.set_zlabel('Y')
ax.set_title('3D contour')
plt.show()

###not needed:

sys.exit()

#sort by k
sorter = k_points.argsort()

fig = plt.figure()

ax = plt.axes(projection="3d")

ax.set_xlabel('x')
ax.set_ylabel('k')
ax.set_zlabel('y')

ax.scatter3D(x_points, k_points, y_points,s=1.3, c=k_points)
plt.show()

