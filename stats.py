import sys
import math
import numpy as np
import pandas as pd
import pickle
from collections import Counter
import seaborn as sn
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter

from curve import add
from dataset_creation import curve_points, arrays, mult_arrays

"""
Get some stats about the curve
"""

#check how often we have to divide the search space
p = 2**256 - 2**32 - 977
for i in range(250):
    p /= 2
    print('size of search space:',p,', if divided',i,'times')


#plot bars how often a point circulates until it reaches itself again
a,b,p = 2,3,463

ds = curve_points(a,b,p)

counters = np.empty(ds.shape[0])
for point_id in range(ds.shape[0]):
    x_fix,y_fix = ds.at[point_id, 'x'], ds.at[point_id, 'y']
    x_old,y_old = x_fix,y_fix
    x_new,y_new = -1,-1
    counter = 0
    while x_new != x_fix and y_new != y_fix:
        x_new,y_new = add(x_old,y_old,x_fix,y_fix,a,p)
        x_old,y_old = x_new,y_new
        counter += 1

    counters[point_id] = counter

plt.bar(range(ds.shape[0]),counters)
plt.show()

#test how many samples we have with INF points
#curve parameters
a,b = 9,9
for num in range(500):
   #loop through all prime numbers
   if num > 1:
       for i in range(2, num):
           if (num % i) == 0:
               break
       else:
           p = num
           x_points, k_points, y_points = mult_arrays(a, b, p)

           df = pd.concat([x_points.drop('point_X',axis = 1), y_points.drop('point_Y',axis = 1)], axis=1, sort=False)
           df.fillna(value=-99, inplace=True)
           df = pd.DataFrame(pd.np.column_stack([df, k_points])).rename(columns={0: "x_X", 1: "y_X",2: "x_Y", 3: "y_Y",4:"k"})
           #before deleting INF points:
           k = df.iloc[:,-1]
           num_classes_old = len(Counter(k).keys())

           #after deleting INF points:
           df = df[df.x_X != -99]
           df = df[df.x_Y != -99]
           k = df.iloc[:, -1]
           num_classes_new = len(Counter(k).keys())
           print('prime',p,':',num_classes_new,num_classes_old,'=>',num_classes_new/num_classes_old,'%')




