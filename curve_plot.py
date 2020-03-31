import sys
import math
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits import mplot3d

from dataset_creation import curve_points
from curve import add

"""
Plot the path of adding a point to itself in different spaces
"""

#set curve parameters
a,b,p = 2,3,97

ds = curve_points(a,b,p)

#plot (id of) y on k with fixed x
#fix x
x_id = 3
#introduce helper column
ds['p'] = ds["x"].map(str) + "," + ds["y"].map(str)

y = np.array(range(p))#y values to be filled
y_1 = np.array(range(p))#y values to be filled
y_2 = np.array(range(p))#y values to be filled

#fill in first two entries of y
y[0] = ds['id'].iloc[-1]
y[1] = x_id
y_1[0],y_2[0] = p,p
y_1[1],y_2[1] = ds['x'].iloc[0],ds['y'].iloc[0]

#initialise loop
x_fix,y_fix = ds.at[x_id, 'x'], ds.at[x_id, 'y']
x_old,y_old = x_fix,y_fix
x_new,y_new = -1,-1

for i in range(2,p):
    x_new,y_new = add(x_old,y_old,x_fix,y_fix,a,p)
    point_id = ds[ds['p'].values == str(x_new) + ',' + str(y_new)].iloc[0, 2]
    y[i] = point_id
    if x_new == None and y_new == None: #for INF point use (p,p)
        y_1[i] = p
        y_2[i] = p
    else:
        y_1[i] =  x_new
        y_2[i] =  y_new
    x_old, y_old = x_new,y_new

k = np.array(range(p))

df = pd.DataFrame({'y':y, 'k':k})

fig = plt.figure()
plt.plot(df['k'],df['y'],'-o', markersize=2)
plt.xlabel('k')
plt.ylabel('y')
plt.show()


#plot the trail with coordinates of actual points, where z axis is k
ax = plt.axes(projection="3d")
ax.set_xlabel('y_1')
ax.set_ylabel('y_2')
ax.set_zlabel('k')

ax.plot3D(y_1, y_2, k)
plt.show()

#plot the path of a point being added to itself on the plane
plt.scatter(ds['x'], ds['y'], s= 1.2)
plt.xlabel('x')
plt.ylabel('y')

#create colour map for arrows
cmap = plt.cm.jet
cNorm  = colors.Normalize(vmin=np.min(ds['id']), vmax=np.max(ds['id']))
scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=cmap)

#list the id´s of points we want to draw the path for
list= [7,66] #or all points: range(ds.shape[0])

for point_id in list:
    colorVal = scalarMap.to_rgba(point_id)
    x_fix,y_fix = ds.at[point_id, 'x'], ds.at[point_id, 'y']
    x_old,y_old = x_fix,y_fix
    x_new,y_new = -1,-1
    while x_new != x_fix and y_new != y_fix:
        x_new,y_new = add(x_old,y_old,x_fix,y_fix,a,p)
        if x_new == None or y_new == None:
            print('reached INF')
            break
        plt.arrow(x_old,y_old,x_new-x_old,y_new-y_old,head_width=0.8,color=colorVal)
        x_old,y_old = x_new,y_new

plt.show()
