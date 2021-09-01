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
run_times = 5
table = pd.DataFrame(np.zeros([5, 23]), index=['avg', 'std', 'worst', 'best', 'time'])
loss_curves = np.zeros([G, 23])
F_table = np.zeros([run_times, 23])
for t in range(run_times):
    item = 0
    ub = 5.12*np.ones(D)
    lb = -5.12*np.ones(D)
    optimizer = WOA(fit_func=benchmark.Sphere,
                    num_dim=D, num_particle=P, max_iter=G, x_max=ub, x_min=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve

    
    item = item + 1
    ub = 10*np.ones(D)
    lb = -10*np.ones(D)
    optimizer = WOA(fit_func=benchmark.Schewefel3,
                    num_dim=D, num_particle=P, max_iter=G, x_max=ub, x_min=lb)
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
    optimizer = WOA(fit_func=benchmark.Quartic,
                    num_dim=D, num_particle=P, max_iter=G, x_max=ub, x_min=lb)
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
    optimizer = WOA(fit_func=benchmark.Schewefel2,
                    num_dim=D, num_particle=P, max_iter=G, x_max=ub, x_min=lb)
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
    optimizer = WOA(fit_func=benchmark.Rosenbrock,
                    num_dim=D, num_particle=P, max_iter=G, x_max=ub, x_min=lb)
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
    optimizer = WOA(fit_func=benchmark.Step1,
                    num_dim=D, num_particle=P, max_iter=G, x_max=ub, x_min=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve


    item = item + 1
    Quadric_Noise = functools.partial(benchmark.Quartic, with_noise=True)
    ub = 1.28*np.ones(D)
    lb = -1.28*np.ones(D)
    optimizer = WOA(fit_func=Quadric_Noise,
                    num_dim=D, num_particle=P, max_iter=G, x_max=ub, x_min=lb)
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
    optimizer = WOA(fit_func=benchmark.Schwefel4,
                    num_dim=D, num_particle=P, max_iter=G, x_max=ub, x_min=lb)
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
    optimizer = WOA(fit_func=benchmark.Rastrigin,
                    num_dim=D, num_particle=P, max_iter=G, x_max=ub, x_min=lb)
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
    optimizer = WOA(fit_func=benchmark.Ackley1,
                    num_dim=D, num_particle=P, max_iter=G, x_max=ub, x_min=lb)
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
    optimizer = WOA(fit_func=benchmark.Griewank,
                    num_dim=D, num_particle=P, max_iter=G, x_max=ub, x_min=lb)
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
    optimizer = WOA(fit_func=benchmark.Generalized_Penalized1,
                    num_dim=D, num_particle=P, max_iter=G, x_max=ub, x_min=lb)
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
    optimizer = WOA(fit_func=benchmark.Generalized_Penalized2,
                    num_dim=D, num_particle=P, max_iter=G, x_max=ub, x_min=lb)
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
    optimizer = WOA(fit_func=benchmark.De_Jong5,
                    num_dim=2, num_particle=P, max_iter=G, x_max=ub, x_min=lb)
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
    optimizer = WOA(fit_func=benchmark.Kowalik,
                    num_dim=4, num_particle=P, max_iter=G, x_max=ub, x_min=lb)
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
    optimizer = WOA(fit_func=benchmark.Six_Hump_Camel_Back,
                    num_dim=2, num_particle=P, max_iter=G, x_max=ub, x_min=lb)
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
    optimizer = WOA(fit_func=benchmark.Branin,
                    num_dim=2, num_particle=P, max_iter=G, x_max=ub, x_min=lb)
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
    optimizer = WOA(fit_func=benchmark.Goldstein_Price,
                    num_dim=2, num_particle=P, max_iter=G, x_max=ub, x_min=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = 3*np.ones(3)
    lb = 1*np.ones(3)
    optimizer = WOA(fit_func=benchmark.Hartmann3,
                    num_dim=3, num_particle=P, max_iter=G, x_max=ub, x_min=lb)
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
    optimizer = WOA(fit_func=benchmark.Hartmann6,
                    num_dim=6, num_particle=P, max_iter=G, x_max=ub, x_min=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    Shekel_m5 = functools.partial(benchmark.Shekel, m=7)
    ub = 10*np.ones(4)
    lb = 0*np.ones(4)
    optimizer = WOA(fit_func=Shekel_m5,
                    num_dim=4, num_particle=P, max_iter=G, x_max=ub, x_min=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    
    item = item + 1
    Shekel_m7 = functools.partial(benchmark.Shekel, m=7)
    ub = 10*np.ones(4)
    lb = 0*np.ones(4)
    optimizer = WOA(fit_func=Shekel_m7,
                    num_dim=4, num_particle=P, max_iter=G, x_max=ub, x_min=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve
    
    item = item + 1
    Shekel_m10 = functools.partial(benchmark.Shekel, m=10)
    ub = 10*np.ones(4)
    lb = 0*np.ones(4)
    optimizer = WOA(fit_func=Shekel_m10,
                    num_dim=4, num_particle=P, max_iter=G, x_max=ub, x_min=lb)
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

table.columns = ['Sphere', 'Schewefel3', 'Quartic', 'Schewefel2', 'Rosenbrock',
                 'Step1', 'Quartic', 'Schwefel4', 'Rastrigin', 'Ackley1',
                 'Griewank', 'Generalized_Penalized1', 'Generalized_Penalized2', 'De_Jong5', 'Kowalik',
                 'Six_Hump_Camel_Back', 'Branin', 'Goldstein_Price', 'Hartmann3', 'Hartmann6',
                 'Shekel_m5', 'Shekel_m7', 'Shekel_m10']