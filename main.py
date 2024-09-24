import numpy as np
import pandas as pd
import plotly.graph_objects as go
from addict import Dict

from benchmark import Benchmark
from WOA import woa

# 參數設定
times = 30  # 每一 benchmark 要反覆跑幾次

# 執行
grades = Dict()
for time_ in range(times):
    for model_name in [
        "Sphere",
        "Schwefel_P222",
        "Schwefel_P221",
        "Rosenbrock",
        "Step",
        "Quartic",
        "Schwefel_226",
        "Rastrigin",
        "Ackley",
        "Griewank",
        "Penalized1",
        "Penalized2",
        "ShekelFoxholes",
        "Kowalik",
        "SixHumpCamelBack",
        "Branin",
        "GoldsteinPrice",
        "Hartmann3",
        "Hartmann6",
        "Shekel5",
        "Shekel7",
        "Shekel10",
    ]:
        # 取得 benchmark
        model = Benchmark(model_name=model_name)

        # 初始化求解器
        solver = woa(
            max_iter=500,
            pop_size=30,
            b=1,
            a_max=2,
            a_min=0,
            a2_max=-1,
            a2_min=-2,
            l_max=1,
            l_min=-1,
            lb=model.model.lb,
            ub=model.model.ub,
            name=model_name,
            benchmark=model.model.evaluate,
        )

        # 求解
        solver.run()

        # 儲存結果
        if not grades[model_name]:
            grades[model_name] = {
                "curves": [solver.curve],
                "costs": [solver.cost],
                "gbest_fs": [solver.gbest_f],
                "opt_f": model.model.opt_f,
            }
        else:
            grades[model_name]["curves"].append(solver.curve)
            grades[model_name]["costs"].append(solver.cost)
            grades[model_name]["gbest_fs"].append(solver.gbest_f)
        print(f"{time_ + 1}/{times} {model_name}")

# 模擬結果
table = Dict()
for model_name in [
    "Sphere",
    "Schwefel_P222",
    "Schwefel_P221",
    "Rosenbrock",
    "Step",
    "Quartic",
    "Schwefel_226",
    "Rastrigin",
    "Ackley",
    "Griewank",
    "Penalized1",
    "Penalized2",
    "ShekelFoxholes",
    "Kowalik",
    "SixHumpCamelBack",
    "Branin",
    "GoldsteinPrice",
    "Hartmann3",
    "Hartmann6",
    "Shekel5",
    "Shekel7",
    "Shekel10",
]:
    table[model_name] = {
        "opt": round(grades[model_name]["opt_f"], 5),
        "best": round(min(grades[model_name]["gbest_fs"]), 5),
        "avg": round(np.mean(grades[model_name]["gbest_fs"]), 5),
        "worst": round(max(grades[model_name]["gbest_fs"]), 5),
        "std": round(np.std(grades[model_name]["gbest_fs"]), 5),
        "cost(sec)": round(np.mean(grades[model_name]["costs"]), 5),
    }
table = pd.DataFrame(table).T
table.to_csv("WOA.csv")

# 收斂曲線
fig = go.Figure()
for curve in grades["Sphere"]["curves"]:
    fig.add_trace(go.Scatter(y=curve, mode="lines+markers"))
fig.update_layout(
    title=f"{'Sphere'}(f_opt: {grades['Sphere']['opt_f']:.2f}, f_best: {min(grades['Sphere']['gbest_fs']):.2f}, f_avg: {np.mean(grades['Sphere']['gbest_fs']):.2f}, f_worst: {max(grades['Sphere']['gbest_fs']):.2f}, std: {np.std(grades['Sphere']['gbest_fs']):.2f}, f_worst: {max(grades['Sphere']['gbest_fs']):.2f},  avg cost: {np.mean(grades['Sphere']['costs']):.2f} sec)",
    xaxis_title="Iteration",
    yaxis_title="Fitness",
)
fig.show()
