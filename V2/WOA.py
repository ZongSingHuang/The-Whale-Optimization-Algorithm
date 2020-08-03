# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 21:59:58 2020

@author: ZongSing_NB

Main reference:http://www.alimirjalili.com/WOA.html
"""

import numpy as np
import matplotlib.pyplot as plt

class WOA():
    def __init__(self, fit_func, num_dim=30, num_particle=20, max_iter=500, update_when_iter=False,
                 b=1, x_max=1, x_min=0, a_max=2, a_min=0, l_max=1, l_min=-1, a2_max=-1, a2_min=-2):
        self.fit_func = fit_func        
        self.num_dim = num_dim
        self.num_particle = num_particle
        self.max_iter = max_iter     
        self.x_max = x_max
        self.x_min = x_min
        self.a_max = a_max
        self.a_min = a_min
        self.a2_max = a2_max
        self.a2_min = a2_min
        self.l_max = l_max
        self.l_min = l_min
        self.b = b
        self.update_when_iter = update_when_iter

        self._iter = 1
        self.gBest_X = None
        self.gBest_score = np.inf
        self.gBest_curve = np.zeros(self.max_iter)
        self.X = np.random.uniform(size=[self.num_particle, self.num_dim])*(self.x_max-self.x_min) + self.x_min
        
        score = self.fit_func(self.X)
        self.gBest_score = score.min().copy()
        self.gBest_X = self.X[score.argmin()].copy()
        self.gBest_curve[0] = self.gBest_score.copy()
        
    def opt(self):
        while(self._iter<self.max_iter):
            a = self.a_max - (self.a_max-self.a_min)*(self._iter/self.max_iter)
            
            for i in range(self.num_particle):
                p = np.random.uniform()
                R1 = np.random.uniform()
                R2 = np.random.uniform()
                A = 2*a*R1 - a
                C = 2*R2
                l = np.random.uniform()*(self.l_max-self.l_min) + self.l_min

                if p<0.5:
                    if np.abs(A)<1:
                        D = np.abs(C*self.gBest_X - self.X[i, :])
                        self.X[i, :] = self.gBest_X - A*D
                    else:
                        X_rand = self.X[np.random.randint(low=0, high=self.num_particle, size=self.num_dim), :]
                        X_rand = np.diag(X_rand).copy()
                        D = np.abs(C*X_rand - self.X[i, :])
                        self.X[i, :] = self.gBest_X - A*D                
                else:
                    D = np.abs(self.gBest_X - self.X[i, :])
                    self.X[i, :] = D*np.exp(self.b*l)*np.cos(2*np.pi*l)+self.gBest_X                    

                        
                if self.update_when_iter==False:      
                    # case2-2. 每跑完一條就更新一次gBest
                    self.X[i, self.x_max < self.X[i, :]] = self.x_max[self.x_max < self.X[i, :]]
                    self.X[i, self.x_min > self.X[i, :]] = self.x_min[self.x_min > self.X[i, :]]
                    score = self.fit_func(self.X[i])
                    if score < self.gBest_score:
                        self.gBest_X = self.X[i, :].copy()
                        self.gBest_score = score.copy()
            
            if self.update_when_iter==True:
                # case2-1. 每跑完一代才更新gBest
                bound_max = np.dot(np.ones(self.num_particle)[:, np.newaxis], self.x_max[np.newaxis, :])
                bound_min = np.dot(np.ones(self.num_particle)[:, np.newaxis], self.x_min[np.newaxis, :])
                self.X[bound_max < self.X] = bound_max[bound_max < self.X]
                self.X[bound_min > self.X] = bound_min[bound_min > self.X]
                score = self.fit_func(self.X)
                if np.min(score) < self.gBest_score:
                    self.gBest_X = self.X[score.argmin()].copy()
                    self.gBest_score = score.min().copy()
                
            self.gBest_curve[self._iter] = self.gBest_score.copy()    
            self._iter = self._iter + 1
        
    def plot_curve(self):
        plt.figure()
        plt.title('loss curve ['+str(round(self.gBest_curve[-1], 3))+']')
        plt.plot(self.gBest_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()        
            