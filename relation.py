import sys
import math
import numpy as np
import pandas as pd
import pickle
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import seaborn as sn

from dataset_creation import curve_points
from curve import add

"""
Visualise search space
"""

#curve parameters
a,b,p = 2,3,17

ds = curve_points(a,b,p)

#helper column
ds['p'] = ds["x"].map(str) + "," + ds["y"].map(str)
n=ds.shape[0]

#create an adjecent matrix, where jxi entry means j*k=i, (i,j'th points)
matrix = np.full([n,n],-10)
for index, row in ds.iterrows():
    matrix[index,index] = 1
    x_fix, y_fix = row[0], row[1]
    x_old, y_old = x_fix, y_fix
    x_new, y_new = -1, -1
    k = 2
    while x_new != x_fix or y_new != y_fix:
        x_new, y_new = add(x_old, y_old, x_fix, y_fix, a, p)
        point_id = ds[ds['p'].values == str(x_new)+','+str(y_new)].iloc[0,2]
        matrix[index, point_id] = k
        x_old, y_old = x_new, y_new
        k += 1

#plot heat colored matrix
plt.imshow(matrix)
plt.colorbar().set_label('k')
plt.xlabel('y')
plt.ylabel('x')
plt.show()

#plot correlation matrix
sn.heatmap(pd.DataFrame(matrix).corr(), annot=False)
plt.show()

#plot matrix in 3d
m = np.count_nonzero(matrix)
x,y,k = np.empty(m),np.empty(m),np.empty(m)
j = 0
for index, value in np.ndenumerate(matrix):
    if value != 0:
        x[j],y[j],k[j]= index[0],index[1],value
        j +=1

ax = plt.axes(projection="3d")

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('k')

ax.scatter3D(x,y,k,s=7.3)
plt.show()

