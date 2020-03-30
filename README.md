# feasibility-study-ML-on-ECC
## Problem
Many cryptography protocols utilize functions, which are easy to compute in one direction, but are hard to compute in the other direction. Such functions are called *One-way functions*. Despite the wide use of one-way functions in cryptography it is an open problem, if they really exist or if the algorithm to compute the *hard* direction, is just not found yet. 
## Idea
We would like to use ...
## ECC
An *elliptic curve* in this context is a plane curve of the form *``y^2 = x^3 + ax + b``* over a finite field generated by a prime number *p*. A point *P* on this curve is a tuple of *x* and *y*, which satisfy the equation. By defining an operation on all points of the curve we obtain a group. For the identity element we include the point at infinity. The number of all points, which satisfy the equation is called the order of the group and is finite. The operation we define, is the addition of two points *P* and *Q*, which yields a third point of the curve. 
Here we see all points of the curve with parameters ``a=2``, ``b=3`` and ``p=97``.

![curve](https://github.com/bhaecker/feasibility-study-ML-on-ECC/blob/master/images/curve97.png)

For being cryptographically relevant, the curve has to be chosen with a very large prime. For the curve ``secp256k1`` used in Bitcoin for example, ``p`` is equal to ``2^256 - 2^32 - 977``. For the sake of understandability, we analyse curves with much lower prime numbers in the following and hope that we can transfer the insides to curves over higher primes.   

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

![curve](https://github.com/bhaecker/feasibility-study-ML-on-ECC/blob/master/images/matrix_p17.png)


A ``k`` of ``-10`` (the dark blue regions) denotes there is no ``k``, which satisfies the relation.
On the other hand, there might be more ``k``'s for which the realtion holds, so we take the highest of these. 

## Classification
After getting comfortable with the curve and the problem we'd like to solve, we are ready to takle it, by reformulating it as a classification problem. 
The idea is straightforward [...] 
### Supervised Learning
Supervised learning uses labeled instances. In order to classify an unseen instance, a supervised algorithm should have seen samples of the class it belongs to, during the training process. For a curve with ``p = 2^256 - 2^32 - 977`` this is not possible, since there are just too many possible ``k``'s, respectively classes.  
Even though we can not takle our problem with supervised learning, we want to see how a neural network performes on a curve, where its possible to see all classes, during the training process. For that we fix a curve with parameters ``a=2``, ``b=3`` and ``p=257``. This gives us ``239`` points. The resulting data set has ``57600`` samples. 

#### Correlation
Before applying algorithms on the data set, we are interested, if there are any correlations between our features. For that, we calculate the pearson coefficient of both data sets.

![curve](https://github.com/bhaecker/feasibility-study-ML-on-ECC/blob/master/images/corrID257.png)

On the left we see the correlation matrix for the data set with two features (two curve points enumerated by an id) and on the right the correlation matrix for the data set with four features (two 2 dimensional points). 
#### Classification with NN
Since we have very few classes, we use a shallow fully connected neural network with two layers and 128 neurons per layer. We train on ``80%`` of the samples for ``100 epochs`` and test on ``20%``. In the following we see the accuracy (left) and the loss (right) on both sets for the data set with four features. 

![curve](https://github.com/bhaecker/feasibility-study-ML-on-ECC/blob/master/images/p257ep100.png)

The data set with two features shows similar characteristics. 
The overall accuracy stays around ``~1.5%``. Different network architectures and changes in other hyper-parameters, do not improve the performance.
### Unsupervised Learning
For unsupervised algorithms we need a metric of how good our current solution/classification is. In most cases, a form of derivation is used, to point the algorithm in the right direction. Since our function is non continous, we can not rely on that neat performance measure.
To see how y depends on k, we fix an point x on the curve and plot ``y=x k`` in depending on ``k``.
Here we fix the point ``P=(17,87)`` with ``id=3`` on the curve with parameters ``a=2``, ``b=3`` and ``p=97`` and analyse how ``y`` changes, when we in- or decrease ``k``.
 
![curve](https://github.com/bhaecker/feasibility-study-ML-on-ECC/blob/master/images/xfix.png)

Unfourtunally, we do not find a measurment of how good our current solution is. 
### Divide and Conquer approach
It turns out, that working with parameter ``k`` as label is not fruitful. In the next approach, we want to use an algorithm to reduce our search space. We proceed in a divide and conquer matter, which means that for two points ``P`` and ``Q`` we decide if ``k`` lies n the left half or right half of the interval of all possible ``k``'s. We do this, until we reduced the interval of possible ``k``'s so far, that we can brute-force ``k`` times ``P``. Even though the number of points on a curve is not equal the generating prime number, they are of the same order of magnintute. The exact number of points can be derived with Schoof's algorithm. For an estimate on how often we have to devide the search space, we use ``p`` instead of the real number of points. It turns out that after ``~235`` times, we end up with a search space small enough, to brute force it, when working with ``p = 2^256 - 2^32 - 977``.

As a decision algorithm, we use a neural net, which gets as inputs two points ``P`` and ``Q`` for which ``Q = P k`` holds. It is trained on two classes: ``0`` if ``k<p/2`` and ``1`` if ``k>=p/2``. The training samples are generated randomly, by choosing a random ``P`` and multiplying it with a random ``k``. 

Training and testing with different network architectures and hyper parameter configurations, leads to an average accuracy of ``50%``, which is as good as flipping a coin in a decision problem. If we take into account, that we have to decide right in ``~235`` cases, this approach is not very promising either. 

## Conclusion
