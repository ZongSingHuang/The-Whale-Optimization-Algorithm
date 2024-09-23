import time
from typing import Callable

import numpy as np
import plotly.graph_objects as go


class woa:
    def __init__(
        self,
        max_iter: int,
        pop_size: int,
        b: float,
        a_max: float,
        a_min: float,
        a2_max: float,
        a2_min: float,
        l_max: float,
        l_min: float,
        lb: list,
        ub: list,
        benchmark: Callable,
        name: str = "",
    ):
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.a_max = a_max
        self.a_min = a_min
        self.a2_max = a2_max
        self.a2_min = a2_min
        self.l_max = l_max
        self.l_min = l_min
        self.b = b
        self.lb = lb
        self.ub = ub
        self.dim = len(lb)
        self.name = name
        self.benchmark = benchmark

        self.cost = 0
        self.curve = list()
        self.gbest_f = np.inf
        self.gbest_x = np.zeros(self.dim)
        self.pbest_F = np.full(self.pop_size, np.inf)
        self.pbest_X = np.zeros([self.pop_size, self.dim])

    def run(self):
        st = time.time()

        # 初始化
        X = np.random.uniform(low=self.lb, high=self.ub, size=[self.pop_size, self.dim])

        # 迭代
        for _iter in range(self.max_iter):
            # 適應值計算
            F = self.benchmark(X)

            # 更新最佳解
            mask = F < self.pbest_F
            self.pbest_X[mask] = X[mask].copy()
            self.pbest_F[mask] = F[mask].copy()

            if self.pbest_F.min() < self.gbest_f:
                idx = self.pbest_F.argmin()
                self.gbest_x = self.pbest_X[idx].copy()
                self.gbest_f = self.pbest_F.min()

            # 收斂曲線
            self.curve.append(self.gbest_f)

            # 更新
            a = self.a_max - (self.a_max - self.a_min) * (_iter / self.max_iter)
            a2 = self.a2_max - (self.a2_max - self.a2_min) * (_iter / self.max_iter)

            for i in range(self.pop_size):
                p = np.random.uniform()
                r1 = np.random.uniform()
                r2 = np.random.uniform()
                r3 = np.random.uniform()
                A = 2 * a * r1 - a  # (2.3)
                C = 2 * r2  # (2.4)
                l = (a2 - 1) * r3 + 1  # (???)

                if p > 0.5:
                    D = np.abs(self.gbest_x - X[i, :])
                    X[i, :] = (
                        D * np.exp(self.b * l) * np.cos(2 * np.pi * l) + self.gbest_x
                    )  # (2.5)
                else:
                    if np.abs(A) < 1:
                        D = np.abs(C * self.gbest_x - X[i, :])  # (2.1)
                        X[i, :] = self.gbest_x - A * D  # (2.2)
                    else:
                        idx = np.random.randint(low=0, high=self.pop_size)
                        X_rand = X[idx]
                        D = np.abs(C * X_rand - X[i, :])  # (2.7)
                        X[i, :] = X_rand - A * D  # (2.8)

            # 邊界處理
            X = np.clip(X, self.lb, self.ub)  # 邊界處理

        # 總計算時間
        ed = time.time()
        self.cost = round(ed - st, 2)

    def plot(self):
        if self.curve:
            # 創建 Figure
            fig = go.Figure()

            # 添加數據列到 Figure
            fig.add_trace(go.Scatter(y=self.curve, mode="lines+markers"))

            # 設置圖表標題及坐標軸標籤
            fig.update_layout(
                title=f"{self.name}(best fitness: {min(self.curve):.2f}, cost: {self.cost} sec)",
                xaxis_title="Iteration",
                yaxis_title="Fitness",
            )

            # 顯示圖表
            fig.show()
