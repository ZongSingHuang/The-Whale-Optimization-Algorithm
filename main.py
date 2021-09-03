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
table = pd.DataFrame(np.zeros([5, 72]), index=['avg', 'std', 'worst', 'best', 'time'])
loss_curves = np.zeros([G, 72])
F_table = np.zeros([run_times, 72])
for t in range(run_times):
    item = 0
    ub = 32*np.ones(D)
    lb = -32*np.ones(D)
    optimizer = WOA(fitness=benchmark.Ackley1,
                    D=D, P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    loss_curves[:, item] += optimizer.loss_curve

    
    # item = item + 1
    # ub = 10*np.ones(D)
    # lb = -10*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Alpine1,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve


    # item = item + 1
    # ub = 500*np.ones(D)
    # lb = -500*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Bartels_Conn,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve


    # item = item + 1
    # ub = 4.5*np.ones(D)
    # lb = -4.5*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Beale,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    

    # item = item + 1
    # ub = 2*np.pi*np.ones(D)
    # lb = -2*np.pi*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Bird,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    

    # item = item + 1
    # ub = 100*np.ones(D)
    # lb = -100*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Bohachevsky1,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve


    # item = item + 1
    # ub = 100*np.ones(D)
    # lb = -100*np.ones(D)
    # optimizer = WOA(fitness=Bohachevsky2,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve


    # item = item + 1
    # ub = 50*np.ones(D)
    # lb = -50*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Bohachevsky3,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve


    # item = item + 1
    # ub = 10*np.ones(D)
    # lb = -10*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Booth,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve


    # item = item + 1
    # ub = 5*np.ones(D)
    # lb = -5*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Branin,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    

    # item = item + 1
    # ub = 4*np.ones(D)
    # lb = -1*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Brown,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 100*np.ones(D)
    # lb = -100*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Chung_Reynolds,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 100*np.ones(D)
    # lb = -100*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Cigar,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 10*np.ones(2)
    # lb = -10*np.ones(2)
    # optimizer = WOA(fitness=benchmark.Colville,
    #                 D=2, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 1*np.ones(4)
    # lb = -1*np.ones(4)
    # optimizer = WOA(fitness=benchmark.Cosine_Mixture,
    #                 D=4, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 1*np.ones(2)
    # lb = -1*np.ones(2)
    # optimizer = WOA(fitness=benchmark.Csendes,
    #                 D=2, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 65*np.ones(2)
    # lb = -65*np.ones(2)
    # optimizer = WOA(fitness=benchmark.De_Jong5,
    #                 D=2, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 5.12*np.ones(2)
    # lb = -5.12*np.ones(2)
    # optimizer = WOA(fitness=benchmark.Drop_wave,
    #                 D=2, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 100*np.ones(3)
    # lb = -100*np.ones(3)
    # optimizer = WOA(fitness=benchmark.Easom,
    #                 D=3, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 5.12*np.ones(6)
    # lb = -5.12*np.ones(6)
    # optimizer = WOA(fitness=benchmark.Ellipsoid,
    #                 D=6, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = D*np.ones(4)
    # lb = -D*np.ones(4)
    # optimizer = WOA(fitness=Ellipsoidal,
    #                 D=4, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 100*np.ones(4)
    # lb = -100*np.ones(4)
    # optimizer = WOA(fitness=Elliptic,
    #                 D=4, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 1*np.ones(4)
    # lb = -1*np.ones(4)
    # optimizer = WOA(fitness=Exponential,
    #                 D=4, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    

    # item = item + 1
    # ub = 2*np.ones(D)
    # lb = -2*np.ones(D)
    # optimizer = WOA(fitness=benchmark.GoldStein_Price,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    

    # item = item + 1
    # ub = 600*np.ones(D)
    # lb = -600*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Griewank,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 3*np.ones(D)
    # lb = 1*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Hartmann3,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 1*np.ones(D)
    # lb = 0*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Hartmann6,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 1*np.ones(10)
    # lb = -1*np.ones(10)
    # optimizer = WOA(fitness=benchmark.Inverted_Cosine_Mixture,
    #                 D=10, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 5*np.ones(D)
    # lb = -5*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Kowalik,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 5.12*np.ones(D)
    # lb = -5.12*np.ones(D)
    # optimizer = WOA(fitness=benchmark.k_tablet,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 1.2*np.ones(D)
    # lb = -1.2*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Leon,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = *np.ones(D)
    # lb = *np.ones(D)
    # optimizer = WOA(fitness=benchmark.Levy,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 10*np.ones(2)
    # lb = -10*np.ones(2)
    # optimizer = WOA(fitness=benchmark.Levy_and_Montalvo1,
    #                 D=2, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 5*np.ones(D)
    # lb = -5*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Levy_and_Montalvo2,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 10*np.ones(D)
    # lb = -10*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Matyas,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = np.pi*np.ones(2)
    # lb = 0*np.ones(2)
    # optimizer = WOA(fitness=benchmark.Michalewicz,
    #                 D=2, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    

    # item = item + 1
    # ub = 5.12*np.ones(D)
    # lb = -5.12*np.ones(D)
    # optimizer = WOA(fitness=Noncontinuous_Rastrigin,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 1*np.ones(D)
    # lb = -1*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Pathological,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 10*np.ones(D)
    # lb = 2*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Paviani,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 1*np.ones(D)
    # lb = -1*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Inverted_Cosine_Mixture,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 50*np.ones(D)
    # lb = -50*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Penalized1,
    #                 D=2, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 50*np.ones(D)
    # lb = -50*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Penalized2,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 1*np.ones(2)
    # lb = -1*np.ones(2)
    # optimizer = WOA(fitness=benchmark.Powell,
    #                 D=2, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 1*np.ones(D)
    # lb = -1*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Powell_sum,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = *np.ones(2)
    # lb = *np.ones(2)
    # optimizer = WOA(fitness=benchmark.Quadric,
    #                 D=2, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = *np.ones(D)
    # lb = *np.ones(D)
    # optimizer = WOA(fitness=benchmark.Quadric_Noise,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 1.28*np.ones(2)
    # lb = -1.28*np.ones(2)
    # optimizer = WOA(fitness=benchmark.Quartic,
    #                 D=2, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 5.12*np.ones(2)
    # lb = -5.12*np.ones(2)
    # optimizer = WOA(fitness=benchmark.Rastrigin,
    #                 D=2, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 30*np.ones(2)
    # lb = -30*np.ones(2)
    # optimizer = WOA(fitness=benchmark.Rosenbrock,
    #                 D=2, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 100*np.ones(2)
    # lb = -100*np.ones(2)
    # optimizer = WOA(fitness=benchmark.Salomon,
    #                 D=2, D=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = *np.ones(D)
    # lb = *np.ones(D)
    # optimizer = WOA(fitness=benchmark.Schaffer?,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 100*np.ones(2)
    # lb = -100*np.ones(2)
    # optimizer = WOA(fitness=benchmark.Schaffer6,
    #                 D=2, D=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 100*np.ones(D)
    # lb = -100*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Schwefel12,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 100*np.ones(2)
    # lb = -100*np.ones(2)
    # optimizer = WOA(fitness=benchmark.Schwefel221,
    #                 D=2, D=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 10*np.ones(D)
    # lb = -10*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Schwefel222,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve


    # item = item + 1
    # ub = 500*np.ones(2)
    # lb = -500*np.ones(2)
    # optimizer = WOA(fitness=benchmark.Schwefel226,
    #                 D=2, D=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 10*np.ones(D)
    # lb = 0*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Shekel5,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 10*np.ones(2)
    # lb = 0*np.ones(2)
    # optimizer = WOA(fitness=benchmark.'Shekel7,
    #                 D=2, D=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 10*np.ones(D)
    # lb = 0*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Shekel10,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = *np.ones(2)
    # lb = *np.ones(2)
    # optimizer = WOA(fitness=benchmark.Shekel_Foxholes,
    #                 D=2, D=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 10*np.ones(D)
    # lb = -10*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Shubert,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = np.pi*np.ones(2)
    # lb = 0*np.ones(2)
    # optimizer = WOA(fitness=benchmark.Sinusoidal,
    #                 D=2, D=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 5*np.ones(D)
    # lb = -5*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Six_Hump_Camel_Back,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 100*np.ones(2)
    # lb = -100*np.ones(2)
    # optimizer = WOA(fitness=benchmark.Sphere,
    #                 D=2, D=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 100*np.ones(D)
    # lb = -100*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Step1,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 1*np.ones(2)
    # lb = -1*np.ones(2)
    # optimizer = WOA(fitness=benchmark.Sum_of_different_power,
    #                 D=2, D=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = *np.ones(D)
    # lb = *np.ones(D)
    # optimizer = WOA(fitness=benchmark.Sum_Power,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 10*np.ones(2)
    # lb = -10*np.ones(2)
    # optimizer = WOA(fitness=benchmark.Sum_Squares,
    #                 D=2, D=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 1*np.ones(D)
    # lb = -1*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Tablet,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 100*np.ones(2)
    # lb = -100*np.ones(2)
    # optimizer = WOA(fitness=benchmark.Xin_She_Yang3,
    #                 D=2, D=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 10*np.ones(D)
    # lb = -10*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Xin_She_Yang4,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 10*np.ones(2)
    # lb = -5*np.ones(2)
    # optimizer = WOA(fitness=benchmark.Zakharov,
    #                 D=2, D=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    
    # item = item + 1
    # ub = 5*np.ones(D)
    # lb = -5*np.ones(D)
    # optimizer = WOA(fitness=benchmark.Zettl,
    #                 D=D, P=P, G=G, x_max=ub, x_min=lb)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # F_table[t, item] = optimizer.gbest_F
    # table[item]['avg'] += optimizer.gbest_F
    # table[item]['time'] += ed - st
    # loss_curves[:, item] += optimizer.loss_curve
    
    print(t+1)

loss_curves = loss_curves / run_times
table.loc[['avg', 'time']] = table.loc[['avg', 'time']] / run_times
table.loc['worst'] = F_table.max(axis=0)
table.loc['best'] = F_table.min(axis=0)
table.loc['std'] = F_table.std(axis=0)

table.columns = ['Ackley1', 'Alpine1', 'Bartels_Conn', 'Beale', 'Bird',
                 'Bohachevsky1', 'Bohachevsky2', 'Bohachevsky3', 'Booth', 'Branin',
                 'Brown', 'Chung Reynolds', 'Cigar', 'Colville', 'Cosine Mixture',
                 'Csendes', 'De Jong5', 'Drop wave', 'Easom', 'Ellipsoid',
                 'Ellipsoidal', 'Elliptic', 'Exponential', 'Gold Stein & Price', 'Griewank',
                 'Hartmann3', 'Hartmann6', 'Inverted Cosine Mixture', 'Kowalik', 'k-tablet',
                 'Leon', 'Levy', 'Levy and Montalvo1', 'Levy and Montalvo2', 'Matyas',
                 'Michalewicz', 'Noncontinuous Rastrigin', 'Pathological', 'Paviani', 'Penalized 1',
                 'Penalized 2', 'Powell', 'Powell sum', 'Quadric', 'Quadric Noise',
                 'Quartic', 'Rastrigin', 'Rosenbrock', 'Salomon', 'Schaffer?',
                 'Schaffer6', 'Schwefel P1.2', 'Schwefel P2.21', 'Schwefel P2.22', 'Schwefel P2.26',
                 'Shekel 5', 'Shekel 7', 'Shekel 10', "Shekel's Foxholes", 'Shubert',
                 'Sinusoidal', 'Six Hump Camel Back', 'Sphere', 'Step', 'Sum of different power',
                 'Sum Power', 'Sum Squares', 'Tablet', 'Xin She Yang3', 'Xin She Yang4',
                 'Zakharov', 'Zettl']