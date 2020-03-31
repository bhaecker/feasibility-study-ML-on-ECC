import pandas as pd
import numpy as np
import pickle
import time
import sys

from curve import power
from curve import add

"""
Data set creation of curve points and training sets
"""

def curve_points(a,b,p):
    """
    gives all points on a curve y^2=x^3+ax+b mod p
    and an identifier
    """
    Y = pd.DataFrame({'Left': [y for y in range(p)], 'Right': [power(y,2,p) for y in range(p)]})
    X = pd.DataFrame({ 'Left': [x for x in range(p)], 'Right': [ (power(x,3,p) + a * x+ b) % p for x in range(p)]})
    Z = pd.merge(X, Y, on='Right')[['Left_x','Left_y']] #look for common values on the right side and report only the corresponding x and y values
    Z.columns = ['x','y'] #rename columns

    Inf = pd.DataFrame([[None, None]],columns=['x','y']) #create point at infinity
    Z = Z.append(Inf,ignore_index=True) #add point at infinity

    id_col = pd.DataFrame({id: [j for j in range(0, len(Z.index))]})  # add Id for each point
    Z['id'] = id_col  # add column for Id´s
    return(Z)


def arrays(a,b,p):
    """
    gives three arrays corresponding to x*k =y on a curve with input parameters
    with enumerated curve points (use of id's instead of actual two dimensional points)
    """
    ds = curve_points(a,b,p)
    ds['p'] = ds["x"].map(str) + "," + ds["y"].map(str)
    n = len(ds)
    x = np.empty(n * n, dtype=int)
    k = np.empty(n * n, dtype=int)
    y = np.empty(n*n, dtype = int)
    for i,row1 in ds.iterrows():
        if i == n-1:
            xp, yp = None, None
            xq, yq = None, None
        else:
            xp, yp = tuple(map(int, row1['p'].split(',')))
            xq, yq = tuple(map(int, row1['p'].split(',')))
        for j in range(n):
            x[i * n + j] = i
            k[i * n + j] = j
            if j == 0:    # x*0 = INF
                y[i * n + j] = n-1 #INF is last entry in ds
            elif j == 1:  # x*1 = x
                y[i * n + j] = i
            else:
                point = add(xp,yp,xq,yq,a,p)
                point_id = ds[ds['p'].values == str(point[0]) + ',' + str(point[1])].iloc[0, 2]
                y[i*n+j] = point_id
                xq, yq = point[0], point[1]
    return(x,k,y)



def mult_arrays(a,b,p):
    """
    gives back three arrays corresponding to x*k =y on a curve with input parameters
    """
    ds = curve_points(a,b,p).drop('id', axis = 1)
    ds['point'] = ds["x"].map(str) + "," + ds["y"].map(str)
    n = len(ds)
    X = pd.DataFrame(index=range(n*n), columns=['x_X','y_X','point_X'])
    Y = pd.DataFrame(index=range(n*n), columns=['x_Y','y_Y','point_Y'])
    k = np.empty(n*n, dtype=int)
    for i,row1 in ds.iterrows():
        if i == n-1:
            xp, yp = None, None
            xq, yq = None, None
        else:
            xp, yp = tuple(map(int, row1['point'].split(',')))
            xq, yq = tuple(map(int, row1['point'].split(',')))
        for j in range(n):
            X.at[i * n + j,'x_X'],X.at[i * n + j,'y_X']  = xp,yp
            k[i * n + j] = j
            if j == 0:    # x*0 = INF
                Y.at[i * n + j,'x_Y'],Y.at[i * n + j,'y_Y'] = None, None #INF is last entry in ds
            elif j == 1:  # x*1 = x = y
                Y.at[i * n + j, 'x_Y'],Y.at[i * n + j, 'y_Y'] = X.at[i * n + j, 'x_X'], X.at[i * n + j, 'y_X']
            else:
                point = add(xp,yp,xq,yq,a,p)
                Y.at[i * n + j, 'x_Y'], Y.at[i * n + j, 'y_Y'] = point[0], point[1]
                xq, yq = point[0], point[1]
    return(X,k,Y)

