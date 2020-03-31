from __future__ import division
import numpy as np
import pandas as pd

"""
Disclaimer: some functions are taken from https://github.com/andreacorbellini/ecc
Functions to carry out operations on a discrete elliptic curve modulo some prime
"""

#todo: double and add algo!

def power(x, y, p):
    """
    effective calculation of x to the power of y mod p
    """
    res = 1  # Initialize result
    x = x % p # Update x if it is more than or equal to p
    while (y > 0):
        # If y is odd, multiply x with result
        if ((y & 1) == 1):
            res = (res * x) % p
        y = y >> 1  # y must be even now, so y = y/2
        x = (x * x) % p
    return res

def extended_euclidean_algorithm(a, b):
    """
    Returns a three-tuple (gcd, x, y) such that
    a * x + b * y == gcd, where gcd is the greatest
    common divisor of a and b.

    This function implements the extended Euclidean
    algorithm and runs in O(log b) in the worst case.
    """
    s, old_s = 0, 1
    t, old_t = 1, 0
    r, old_r = b, a

    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t

    return old_r, old_s, old_t


def inverse_of(n, p):
    """
    Returns the multiplicative inverse of
    n modulo p.

    This function returns an integer m such that
    (n * m) % p == 1.
    """
    gcd, x, y = extended_euclidean_algorithm(n, p)
    assert (n * x + p * y) % p == gcd % p

    if gcd != 1:
        # Either n is 0, or p is not a prime number.
        return('no inverse')
    else:
        return x % p

def inverse_mod(k, p):
    """
    Returns the inverse of k modulo p.
    This function returns the only integer x such that (x * k) % p == 1.
    k must be non-zero and p must be a prime.
    """
    if k == 0:
        raise ZeroDivisionError('division by zero')

    if k < 0:
        # k ** -1 = p - (-k) ** -1  (mod p)
        return p - inverse_mod(-k, p)

    # Extended Euclidean algorithm.
    s, old_s = 0, 1
    t, old_t = 1, 0
    r, old_r = p, k

    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t

    gcd, x, y = old_r, old_s, old_t

    assert gcd == 1
    assert (k * x) % p == 1

    return x % p


def add(xp,yp,xq,yq,a,p):
    """
    adds two points on a curve y^2=x^3+ax+b mod p
    """
    if xp is None and yp is None:
        # 0 + point2 = point2
        return(xq,yq)

    if xq is None and yq is None:
        # point1 + 0 = point1
        return(xp,yp)


    if xp == xq and yp != yq:
        # point1 + (-point1) = 0
        return(None,None)

    if (xp == xq):
        temp = inverse_of(2*yp,p)
        if temp == 'no inverse':
            return(None,None)
        else:
            m = (3*xp**2+a)*temp % p
    else:
        temp = inverse_of(xp-xq,p)
        if temp == 'no inverse':
            return (None, None)
        else:
            m = (((yp-yq))*temp) % p
    xr = (m**2 - xp - xq) % p
    yr = (yp + m * ( xr - xp )) % p
    return(xr, (p - yr) % p)


def mult(n,xp,yp,a,p):
    """
    multiply n by (xp,xp) on a curve y^2=x^3+ax+b mod p
    """
    if n == 0 or (xp is None and yp is None):
        return(None,None)
    #if xp is None and yp is None:
     #   return(None,None)
    x,y = xp,yp
    for i in range(n-1):
        x,y = add(xp,yp,x,y,a,p)
    return(x,y)

###not needed:

def curve(a,b,p): # gives all points on a curve y^2=x^3+ax+b mod p
    Y = pd.DataFrame({'Left': [y for y in range(p)], 'Right': [power(y,2,p) for y in range(p)]})
    X = pd.DataFrame({ 'Left': [x for x in range(p)], 'Right': [ (power(x,3,p) + a * x+ b) % p for x in range(p)]})
    Z = pd.merge(X, Y, on='Right')[['Left_x','Left_y']] #look for common values on the right side and report only the corresponding x and y values
    Z.columns = ['x','y'] #rename columns

    Inf = pd.DataFrame([[None, None]],columns=['x','y']) #create point at infinity
    Z = Z.append(Inf,ignore_index=True) #add point at infinity
    return(Z)


def make_ds(a,b,p): #returns a dataframe by multiplying all points of the curve with 1..p
    X = curve(a,b,p)
    Y = pd.DataFrame({ i: [ mult(i,X.iat[j,0],X.iat[j,1],a,p) for j in range(0,len(X.index))] for i in range(1,p+1)})
    return(Y)

