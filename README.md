# feasibility-study-ML-on-ECC
## Problem
Many cryptography protocols utilize functions, which are easy to compute in one direction, but are hard to compute in the other direction. Such functions are called *One-way functions*.
## Idea
We would like to use ...
## ECC
An *elliptic curve* in this context is a plane curve of the form *``y^2 = x^3 + ax + b``* over a finite field generated by a prime number *p*. A point *P* on this curve is a tuple of *x* and *y*, which satisfy the equation. By defining an operation on all points of the curve we obtain a group. For the identity element we include the point at infinity. The number of all points, which satisfy the eqution is called the order of the group and is finite.
The operation we define is the addition of two points *P* and *Q*, which yields a third point of the curve. Multiplication is defined as usual. 

### Multiplication
Multiplication of a point on the curve with an integer *k* yields another point on the curve. So let *P* be a point on a curve with parameters *a,b,p* and *k* some integer, then *``Q = P  k``* is easy to compute.  

#### Path 
### Dimension reduction
#### 3D Plot
#### Search Space Visualisation
#### Correlation
### Supervised Learning
#### Classification with NN
### Unsupervised Learning
### Divide and Conquer approach
