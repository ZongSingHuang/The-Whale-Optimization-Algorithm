# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:01:28 2021

@author: zongsing.huang
"""

# =============================================================================
# main ref
# A new mutation operator for real coded genetic algorithms, 2007, https://doi.org/10.1016/j.amc.2007.03.046
# A Literature Survey of Benchmark Functions For Global Optimization Problems, 2013, https://arxiv.org/pdf/1308.4008.pdf
# An Efficient Real-coded Genetic Algorithm for Real- Parameter Optimization, 2010, https://doi.org/10.1109/ICNC.2010.5584209
# https://www.al-roomi.org/benchmarks
# =============================================================================

import numpy as np

def Ackley(X):
    # X in [-32, 32]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    left = -20 * np.exp( -0.2 * ( 1/D * np.sum(X**2, axis=1) )**0.5 )
    right = np.exp( 1/D * np.sum( np.cos(2*np.pi*X), axis=1 ) )
    
    F = left - right + 20 + np.e
    
    return F

def Cosine_Mixture(X):
    # X in [-1, 1]
    # X* = [0, 0, ..., 0]
    # F* = -0.1*D
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    F = np.sum(X**2, axis=1) - 0.1* np.sum(np.cos(5*np.pi*X), axis=1)
    
    return F

def Exponential(X):
    # X in [-1, 1]
    # X* = [0, 0, ..., 0]
    # F* = -1
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    F = -1 * np.exp( -0.5 * np.sum(X**2, axis=1) )
    
    return F

def Griewank(X):
    # X in [-600, 600]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    F = 1 + 1/4000 * np.sum(X**2, axis=1) - np.prod( np.cos(X/(np.arange(D)+1)), axis=1 )
    
    return F

def Levy_and_Montalvo_1(X):
    # X in [-10, 10]
    # X* = [-1, -1, ..., -1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    y = 1 + 1/4 * (X+1)
    first = 10 * np.sin( np.pi*y[:, 0] )**2
    second = np.sum( (y[:, :-1]-1)**2 * ( 1 + 10*np.sin(np.pi*y[:, 1:])**2 ), axis=1 )
    third = ( y[:, -1] - 1 )**2
    
    F = np.pi/D * (first+second+third)
    
    return F

def Levy_and_Montalvo_2(X):
    # X in [-5, 5]
    # X* = [1, 1, ..., 1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    first = np.sin(3*np.pi*X[:, 0])**2
    second = np.sum( (X[:, :-1]-1)**2 * ( 1+np.sin(3*np.pi*X[:, 1:])**2 ), axis=1 )
    third = ( X[:, -1] - 1 )**2 * (1 + np.sin(2*np.pi*X[:, -1])**2)
    
    F = 0.1 * (first+second+third)
    
    return F

def Paviani(X):
    # X in [2, 10], D fixed 10
    # X* = [9.351, 9.351, ..., 9.351]
    # F* = -45.77848
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    F = np.sum( np.log( X-2 )**2 + np.log( 10-X )**2, axis=1 ) - np.prod(X, axis=1)**0.2
    
    return F

def Rastrigin(X):
    # X in [-5.12, 5.12]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    F = 10*D + np.sum( X**2 - 10*np.cos(2*np.pi*X), axis=1 )
    
    return F

def Rosenbrock(X):
    # De_Jong2
    # X in [-30, 30]
    # X* = [1, 1, ..., 1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    F = np.sum( 100*(X[:, 1:]-X[:, :-1]**2)**2 + (X[:, :-1]-1)**2, axis=1 )
    
    return F

def Schwefel(X):
    # X in [-500, 500]
    # X* = [420.9687, 420.9687, ..., 420.9687]
    # F* = -418.9829*D
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    F = np.sum( -1 * X * np.sin( np.abs(X)**0.5 ), axis=1 )
    
    return F

def Sinusoidal(X):
    # X in [0, pi]
    # X* = (2/3) * [pi, pi, ..., pi]
    # F* = -3.5
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    F = -1 * ( 2.5*np.prod(np.sin(X-np.pi/6), axis=1) + np.prod(np.sin(5*(X-np.pi/6)), axis=1) )
    
    return F

def Zakharov(X):
    # X in [-5.12, 5.12]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    F = np.sum(X**2, axis=1) + np.sum(np.arange(D)/2*X, axis=1)**2 + np.sum(np.arange(D)/2*X, axis=1)**4
    
    return F

def Sphere(X):
    # De_Jong1
    # X in [-5.12, 5.12]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    F = np.sum(X**2, axis=1)
    
    return F

def Axis_Parallel_Hyper_Ellipsoid(X):
    # X in [-5.12, 5.12]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    F = np.sum(np.arange(D)*X**2, axis=1)
    
    return F

def Schewefel3(X):
    # X in [-10, 10]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    F = np.sum(np.abs(X), axis=1) + np.prod(np.abs(X), axis=1)
    
    return F

def Schewefel4(X):
    # X in [-100, 100]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    F = np.max(np.abs(X), axis=1)
    
    return F

def Quartic(X, with_noise=False):
    # modified De_Jong4
    # X in [-1.28, 1.28]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    D = X.shape[1]
    
    if with_noise==True:
        r = np.random.uniform(size=[P])
    else:
        r = 0
        
    F = np.sum( np.arange(D)*X**4, axis=1 ) + r
    
    return F

def Ellipsoidal(X):
    # X in [-D, D]
    # X* = [0, 1, ..., D-1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    F = np.sum( (X-np.arange(D))**2, axis=1 )
    
    return F

def Ellipsoid(X):
    # X in [-5.12, 5.12]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    F = np.sum(( X*1000**(np.arange(D)/(D-1)) )**2, axis=1)
    
    return F

def k_tablet(X):
    # X in [-5.12, 5.12]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    F = np.sum(X**2, axis=1) + np.sum( (100*X)**2, axis=1)
    
    return F

def Bohachevsky(X):
    # X in [-5.12, 5.12]
    # X* = [0, 0, ..., 0]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    F = np.sum(X[:, :-1]**2 + 2*X[:, 1:]**2 - 0.3*np.cos(3*np.pi*X[:, :-1]) - 0.4*np.cos(4*np.pi*X[:, 1:]) + 0.7, axis=1)
    
    return F

def Step1(X):
    # X in [-100, 100]
    # X* = -1<X<1
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    F = np.sum(np.floor(np.abs(X)), axis=1)
    
    return F

def Step2(X):
    # X in [-100, 100]
    # X* = -0.5<=X<0.5
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    F = np.sum(np.floor(np.abs(X+0.5))**2, axis=1)
    
    return F

def Step3(X):
    # X in [-100, 100]
    # X* = -1<X<1
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    F = np.sum(np.floor(X**2), axis=1)
    
    return F

def Stepint(X):
    # X in [-5.12, 5.12]
    # X* = -5.12<X<-5
    # F* = 25-6*D
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    F = 25 + np.sum(np.floor(X), axis=1)
    
    return F

def Six_Hump_Camel_Back(X):
    # X in [-5, 5], D fixed 2
    # X* = (±0.08984201368301331,±0.7126564032704135)
    # F* = −1.031628453489877
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    F = 4*X1**2 - 2.1*X1**4 + 1/3*X1**6 + X1*X2 - 4*X2**2 + 4*X2**4
    
    return F

def Generalized_Penalized1(X):
    # X in [-50, 50]
    # X* = [-1, -1, ..., -1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    y = 1 + 1/4 * (X+1)
    first = 10 * np.sin( np.pi*y[:, 0] )**2
    second = np.sum( (y[:, :-1]-1)**2 * ( 1 + 10*np.sin(np.pi*y[:, 1:])**2 ), axis=1 )
    third = ( y[:, -1] - 1 )**2
    
    F = np.pi/D * (first+second+third) + u(X, 10, 100, 4)
    
    return F

def Generalized_Penalized2(X):
    # X in [-50, 50]
    # X* = [1, 1, ..., 1]
    # F* = 0
    if X.ndim==1:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    first = np.sin(3*np.pi*X[:, 0])**2
    second = np.sum( (X[:, :-1]-1)**2 * ( 1+np.sin(3*np.pi*X[:, 1:])**2 ), axis=1 )
    third = ( X[:, -1] - 1 )**2 * (1 + np.sin(2*np.pi*X[:, -1])**2)
    
    F = 0.1 * (first+second+third) + u(X, 5, 100, 4)
    
    return F

def u(X, a, k, m):
    if X.ndim==1:
        X = X.reshape(1, -1)
    P = X.shape[0]
    D = X.shape[1]
    
    F = np.zeros([P, D])
    mask1 = X>a
    mask3 = X<-a
    mask2 = ~(mask1+mask3)
    
    F[mask1] = k*(X[mask1]-a)**m
    F[mask3] = k*(-X[mask3]-a)**m
    F[mask2] = 0
    
    return F.sum(axis=1)

X = np.zeros([5, 10])
F = Six_Hump_Camel_Back(X)