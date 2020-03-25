# feasibility-study-ML-on-ECC
## Problem
Many cryptography protocols utilize functions, which are easy to compute in one direction, but are hard to compute in the other direction. Such functions are called *One-way functions*. Despite the wide use of one-way functions in cryptography it is an open problem, if they really exist or if the algorithm to compute the the *hard* direction, is just not found yet. 
## Idea
We would like to use ...
## ECC
An *elliptic curve* in this context is a plane curve of the form *``y^2 = x^3 + ax + b``* over a finite field generated by a prime number *p*. A point *P* on this curve is a tuple of *x* and *y*, which satisfy the equation. By defining an operation on all points of the curve we obtain a group. For the identity element we include the point at infinity. The number of all points, which satisfy the equation is called the order of the group and is finite. The operation we define, is the addition of two points *P* and *Q*, which yields a third point of the curve. 
Here we see all points of the curve with parameters ``a=2``, ``b=3`` and ``p=97``.

![curve](https://github.com/bhaecker/feasibility-study-ML-on-ECC/blob/master/images/curve97.png)


### Multiplication
We are interested in multipliying points on a curve with an integer, where multiplication is defined from addition as usual. Multiplication of a point on the curve with an integer *k* yields another point on the curve. So let *P* be a point on a curve with parameters *a,b,p* and *k* some integer, then *``Q = P  k``* is easy to compute.  
#### Path 
We can visualise adding a point to itself by a path. For example if we start with the point ``P1=(17,10)`` on the curve before, we obtain ``P2 = 2 P1 = (32,7)`` and ``P3 = 3 P1 = (1,54)`` and so on until we end up with ``P1`` again. 
![curve](https://github.com/bhaecker/feasibility-study-ML-on-ECC/blob/master/images/curve97_arrows.png)

Note that our path ends at ``P8 = 8 P1 = (0,87)``, one step before we end up in ``P1`` again.
[...]
### Dimension reduction
We want to see the relation between ``k`` and two points ``P``and ``Q``. For that we enumerate all the points. Instead of ``5`` dimensions (two per point plus one for ``k``) we end up with ``3`` dimensions. We refer to an enumerated ``P`` as ``x`` and to ``Q`` as ``y``. 
#### 3D Plot
For ``a=2``, ``b=3`` and ``p=17`` the surface plot of all points ``y``, which are obtained multiplying all ``x`` by all ``k``'s looks like the following:

![curve](https://github.com/bhaecker/feasibility-study-ML-on-ECC/blob/master/images/3d_p17.png)

Note that we are working with discrete points, but connect to neighbouring points with a line. 
#### Search Space Visualisation
Since we are not interested in calculating ``y`` as a product from ``x`` and an integer ``k`` (this is the easy direction), but to find ``k`` if given ``x`` and ``y`` for which ``x k = y`` holds, we take a look at our search space. Fixing ``a=2``, ``b=3`` and ``p=97``, we produce an adjacency matrix, where the rows corespond to ``x`` and the columns to ``y``. An entry denotes ``k`` for which ``x k = y`` holds. We heat colour the matrix:

![curve](https://github.com/bhaecker/feasibility-study-ML-on-ECC/blob/master/images/3d_p17.png)

Since there might be more ``k``'s for which the realtion holds, we take the highest of these. 
#### Correlation
### Supervised Learning
#### Classification with NN
### Unsupervised Learning
For unsupervised algorithms we need a metric of how good our current solution/classification is. Unfourtunally we do not have such a measurment. PLOT y on k with x fix.
### Divide and Conquer approach
