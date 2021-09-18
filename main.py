# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 21:43:03 2020

@author: ZongSing_NB
"""

import time
import functools

import numpy as np
import pandas as pd

from WOA import WOA
import benchmark

D = 30
G = 500
P = 30
run_times = 50
table = pd.DataFrame(np.zeros([5, 36]), index=['avg', 'std', 'worst', 'best', 'time'])
loss_curves = np.zeros([G, 36])
F_table = np.zeros([run_times, 36])
for t in range(run_times):
    item = 0
    ub = 100*np.ones(D)
    lb = -100*np.ones(D)
    optimizer = WOA(fitness=benchmark.Sphere,
                    D=D, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve

    
    item = item + 1
    ub = 5.12*np.ones(D)
    lb = -5.12*np.ones(D)
    optimizer = WOA(fitness=benchmark.Rastrigin,
                    D=D, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve


    item = item + 1
    ub = 32*np.ones(D)
    lb = -32*np.ones(D)
    optimizer = WOA(fitness=benchmark.Ackley,
                    D=D, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve


    item = item + 1
    ub = 600*np.ones(D)
    lb = -600*np.ones(D)
    optimizer = WOA(fitness=benchmark.Griewank,
                    D=D, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    

    item = item + 1
    ub = 10*np.pi*np.ones(D)
    lb = -10*np.pi*np.ones(D)
    optimizer = WOA(fitness=benchmark.Schwefel_P222,
                    D=D, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    

    item = item + 1
    ub = 30*np.ones(D)
    lb = -30*np.ones(D)
    optimizer = WOA(fitness=benchmark.Rosenbrock,
                    D=D, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve


    item = item + 1
    ub = 100*np.ones(D)
    lb = -100*np.ones(D)
    optimizer = WOA(fitness=benchmark.Sehwwefel_P221,
                    D=D, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve


    item = item + 1
    ub = 1.28*np.ones(D)
    lb = -1.28*np.ones(D)
    optimizer = WOA(fitness=benchmark.Quartic,
                    D=D, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve


    item = item + 1
    ub = 100*np.ones(D)
    lb = -100*np.ones(D)
    optimizer = WOA(fitness=benchmark.Schwefel_P12,
                    D=D, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve


    item = item + 1
    ub = 50*np.ones(D)
    lb = -50*np.ones(D)
    optimizer = WOA(fitness=benchmark.Penalized1,
                    D=D, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    

    item = item + 1
    ub = 50*np.ones(D)
    lb = -50*np.ones(D)
    optimizer = WOA(fitness=benchmark.Penalized2,
                    D=D, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = 500*np.ones(D)
    lb = -500*np.ones(D)
    optimizer = WOA(fitness=benchmark.Schwefel_226,
                    D=D, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = 100*np.ones(D)
    lb = -100*np.ones(D)
    optimizer = WOA(fitness=benchmark.Step,
                    D=D, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = 5*np.ones(4)
    lb = -5*np.ones(4)
    optimizer = WOA(fitness=benchmark.Kowalik,
                    D=4, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = 65.536*np.ones(2)
    lb = -65.536*np.ones(2)
    optimizer = WOA(fitness=benchmark.ShekelFoxholes,
                    D=2, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = 2*np.ones(2)
    lb = -2*np.ones(2)
    optimizer = WOA(fitness=benchmark.GoldsteinPrice,
                    D=2, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    Shekel5 = functools.partial(benchmark.Shekel, m=5)
    ub = 10*np.ones(4)
    lb = 0*np.ones(4)
    optimizer = WOA(fitness=Shekel5,
                    D=4, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = [10, 15]*np.ones(2)
    lb = [-5, 0]*np.ones(2)
    optimizer = WOA(fitness=benchmark.Branin,
                    D=2, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = 1*np.ones(3)
    lb = 0*np.ones(3)
    optimizer = WOA(fitness=benchmark.Hartmann3,
                    D=3, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    Shekel7 = functools.partial(benchmark.Shekel, m=7)
    ub = 10*np.ones(4)
    lb = 0*np.ones(4)
    optimizer = WOA(fitness=Shekel7,
                    D=4, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    Shekel10 = functools.partial(benchmark.Shekel, m=10)
    ub = 10*np.ones(4)
    lb = 0*np.ones(4)
    optimizer = WOA(fitness=benchmark.Shekel,
                    D=4, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = 5*np.ones(2)
    lb = -5*np.ones(2)
    optimizer = WOA(fitness=benchmark.SixHumpCamelBack,
                    D=2, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = 1*np.ones(6)
    lb = 0*np.ones(6)
    optimizer = WOA(fitness=benchmark.Hartmann6,
                    D=6, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = 10*np.ones(4)
    lb = -5*np.ones(4)
    optimizer = WOA(fitness=benchmark.Zakharov,
                    D=4, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = 10*np.ones(4)
    lb = -10*np.ones(4)
    optimizer = WOA(fitness=benchmark.SumSquares,
                    D=4, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = 10*np.ones(4)
    lb = -10*np.ones(4)
    optimizer = WOA(fitness=benchmark.Alpine,
                    D=4, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    

    item = item + 1
    ub = np.pi*np.ones(2)
    lb = 0*np.ones(2)
    optimizer = WOA(fitness=benchmark.Michalewicz,
                    D=2, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    

    item = item + 1
    ub = 1*np.ones(D)
    lb = -1*np.ones(D)
    optimizer = WOA(fitness=benchmark.Exponential,
                    D=D, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = 100*np.ones(2)
    lb = -100*np.ones(2)
    optimizer = WOA(fitness=benchmark.Schaffer,
                    D=2, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = 100*np.ones(D)
    lb = -100*np.ones(D)
    optimizer = WOA(fitness=benchmark.BentCigar,
                    D=D, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = 50*np.ones(2)
    lb = -50*np.ones(2)
    optimizer = WOA(fitness=benchmark.Bohachevsky1,
                    D=2, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = 100*np.ones(D)
    lb = -100*np.ones(D)
    optimizer = WOA(fitness=benchmark.Elliptic,
                    D=D, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = 5.12*np.ones(2)
    lb = -5.12*np.ones(2)
    optimizer = WOA(fitness=benchmark.DropWave,
                    D=2, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = 1*np.ones(D)
    lb = -1*np.ones(D)
    optimizer = WOA(fitness=benchmark.CosineMixture,
                    D=D, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = D*np.ones(D)
    lb = -D*np.ones(D)
    optimizer = WOA(fitness=benchmark.Ellipsoidal,
                    D=D, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = 10*np.ones(2)
    lb = -10*np.ones(2)
    optimizer = WOA(fitness=benchmark.LevyandMontalvo1,
                    D=2, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve

    
    print(t+1)

loss_curves = loss_curves / run_times
table.loc[['avg', 'time']] = table.loc[['avg', 'time']] / run_times
table.loc['worst'] = F_table.max(axis=0)
table.loc['best'] = F_table.min(axis=0)
table.loc['std'] = F_table.std(axis=0)

table.columns = ['Sphere', 'Rastrigin', 'Ackley', 'Griewank', 'Schwefel P2.22',
                 'Rosenbrock', 'Sehwwefel P2.21', 'Quartic', 'Schwefel P1.2', 'Penalized 1',
                 'Penalized 2', 'Schwefel P2.26', 'Step', 'Kowalik', 'Shekel Foxholes',
                 'Goldstein-Price', 'Shekel 5', 'Branin', 'Hartmann 3', 'Shekel 7',
                 'Shekel 10', 'Six-Hump Camel-Back', 'Hartmann 6', 'Zakharov', 'Sum Squares',
                 'Alpine', 'Michalewicz', 'Exponential', 'Schaffer', 'Bent Cigar',
                 'Bohachevsky 1', 'Elliptic', 'Drop Wave', 'Cosine Mixture', 'Ellipsoidal',
                 'Levy and Montalvo 1']