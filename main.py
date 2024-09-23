import numpy as np
import plotly.graph_objects as go
from addict import Dict

from benchmark import Benchmark
from WOA import woa

times = 30
grades = Dict()
for _ in range(times):
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
        model = Benchmark(model_name=model_name)
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
        solver.run()
        if not grades[model_name]:
            grades[model_name] = {
                "curves": [solver.curve],
                "costs": [solver.cost],
                "gbest_fs": [solver.gbest_f],
                "opt_f": [model.model.opt_f],
            }
        else:
            grades[model_name]["curves"].append(solver.curve)
            grades[model_name]["costs"].append(solver.cost)
            grades[model_name]["gbest_fs"].append(solver.gbest_f)
        print(model_name)

# 創建 Figure
fig = go.Figure()

# 添加數據列到 Figure
for curve in grades["Sphere"]["curves"]:
    fig.add_trace(go.Scatter(y=curve, mode="lines+markers"))

# 設置圖表標題及坐標軸標籤
fig.update_layout(
    title=f"{"Sphere"}(f_opt: {min(grades["Sphere"]["opt_f"]):.2f}, f_best: {min(grades["Sphere"]["gbest_fs"]):.2f}, f_avg: {np.mean(grades["Sphere"]["gbest_fs"]):.2f}, f_worst: {max(grades["Sphere"]["gbest_fs"]):.2f},  avg cost: {np.mean(grades["Sphere"]["costs"]):.2f} sec)",
    xaxis_title="Iteration",
    yaxis_title="Fitness",
)

# 顯示圖表
fig.show()
