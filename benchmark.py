import numpy as np


class Benchmark:
    def __init__(self, model_name: str):
        match model_name:
            case "Sphere":
                self.model = self.Sphere()
            case "Schwefel_P222":
                self.model = self.Schwefel_P222()
            case "Schwefel_P221":
                self.model = self.Schwefel_P221()
            case "Rosenbrock":
                self.model = self.Rosenbrock()
            case "Step":
                self.model = self.Step()
            case "Quartic":
                self.model = self.Quartic()
            case "Schwefel_226":
                self.model = self.Schwefel_226()
            case "Rastrigin":
                self.model = self.Rastrigin()
            case "Ackley":
                self.model = self.Ackley()
            case "Griewank":
                self.model = self.Griewank()
            case "Penalized1":
                self.model = self.Penalized1()
            case "Penalized2":
                self.model = self.Penalized2()
            case "ShekelFoxholes":
                self.model = self.ShekelFoxholes()
            case "Kowalik":
                self.model = self.Kowalik()
            case "SixHumpCamelBack":
                self.model = self.SixHumpCamelBack()
            case "Branin":
                self.model = self.Branin()
            case "GoldsteinPrice":
                self.model = self.GoldsteinPrice()
            case "Hartmann3":
                self.model = self.Hartmann3()
            case "Hartmann6":
                self.model = self.Hartmann6()
            case "Shekel5":
                self.model = self.Shekel5()
            case "Shekel7":
                self.model = self.Shekel7()
            case "Shekel10":
                self.model = self.Shekel10()
            case _:
                return None

    class Sphere:
        def __init__(self, dim: int = 30, lb: float = -100.0, ub: float = 100.0):
            self.name = "Sphere"
            self.lb = np.full(dim, lb)
            self.ub = np.full(dim, ub)
            self.opt_f = 0
            self.opt_x = np.full(dim, 0)

        def evaluate(self, X: np.array) -> np.array:
            return np.sum(X**2, axis=1)

    class Schwefel_P222:
        def __init__(self, dim: int = 30, lb: float = -10.0, ub: float = 10.0):
            self.name = "Schwefel_P222"
            self.lb = np.full(dim, lb)
            self.ub = np.full(dim, ub)
            self.opt_f = 0
            self.opt_x = np.full(dim, 0)

        def evaluate(self, X: np.array) -> np.array:
            return np.sum(np.abs(X), axis=1) + np.prod(np.abs(X), axis=1)

    class Schwefel_P221:
        def __init__(self, dim: int = 30, lb: float = -100.0, ub: float = 100.0):
            self.name = "Schwefel_P221"
            self.lb = np.full(dim, lb)
            self.ub = np.full(dim, ub)
            self.opt_f = 0
            self.opt_x = np.full(dim, 0)

        def evaluate(self, X: np.array) -> np.array:
            return np.max(np.abs(X), axis=1)

    class Rosenbrock:
        def __init__(self, dim: int = 30, lb: float = -30.0, ub: float = 30.0):
            self.name = "Rosenbrock"
            self.lb = np.full(dim, lb)
            self.ub = np.full(dim, ub)
            self.opt_f = 0
            self.opt_x = np.full(dim, 1)

        def evaluate(self, X: np.array) -> np.array:
            f1 = 100 * (X[:, 1:] - X[:, :-1] ** 2) ** 2
            f2 = (X[:, :-1] - 1) ** 2
            return np.sum(f1 + f2, axis=1)

    class Step:
        def __init__(self, dim: int = 30, lb: float = -100.0, ub: float = 100.0):
            self.name = "Step"
            self.lb = np.full(dim, lb)
            self.ub = np.full(dim, ub)
            self.opt_f = 0
            self.opt_x = f"{np.full(dim, 0.5)}~{np.full(dim, -0.5)}"

        def evaluate(self, X: np.array) -> np.array:
            return np.sum(np.floor(np.abs(X + 0.5)) ** 2, axis=1)

    class Quartic:
        def __init__(self, dim: int = 30, lb: float = -1.28, ub: float = 1.28):
            self.name = "Quartic"
            self.lb = np.full(dim, lb)
            self.ub = np.full(dim, ub)
            self.opt_f = 0
            self.opt_x = np.full(dim, 0)

        def evaluate(self, X: np.array) -> np.array:
            P = X.shape[0]
            D = X.shape[1]
            i = np.arange(D) + 1
            return np.sum(i * X**4, axis=1) + np.random.uniform(size=[P])

    class Schwefel_226:
        def __init__(self, dim: int = 30, lb: float = -500.0, ub: float = 500.0):
            self.name = "Schwefel_226"
            self.lb = np.full(dim, lb)
            self.ub = np.full(dim, ub)
            self.opt_f = -418.982887272433799807913601398 * dim
            self.opt_x = np.full(dim, 420.968746)

        def evaluate(self, X: np.array) -> np.array:
            return -np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1)

    class Rastrigin:
        def __init__(self, dim: int = 30, lb: float = -5.12, ub: float = 5.12):
            self.name = "Rastrigin"
            self.lb = np.full(dim, lb)
            self.ub = np.full(dim, ub)
            self.opt_f = 0
            self.opt_x = np.full(dim, 0)

        def evaluate(self, X: np.array) -> np.array:
            return np.sum(X**2 - 10 * np.cos(2 * np.pi) + 10, axis=1)

    class Ackley:
        def __init__(self, dim: int = 30, lb: float = -5.12, ub: float = 5.12):
            self.name = "Ackley"
            self.lb = np.full(dim, lb)
            self.ub = np.full(dim, ub)
            self.opt_f = 0
            self.opt_x = np.full(dim, 0)

        def evaluate(self, X: np.array) -> np.array:
            D = X.shape[1]
            f1 = -0.2 * np.sqrt(np.sum(X**2, axis=1) / D)
            f2 = np.sum(np.cos(2 * np.pi * X), axis=1) / D
            return -20 * np.exp(f1) - np.exp(f2) + 20 + np.e

    class Griewank:
        def __init__(self, dim: int = 30, lb: float = -600, ub: float = 600):
            self.name = "Griewank"
            self.lb = np.full(dim, lb)
            self.ub = np.full(dim, ub)
            self.opt_f = 0
            self.opt_x = np.full(dim, 0)

        def evaluate(self, X: np.array) -> np.array:
            D = X.shape[1]
            i = np.arange(D) + 1

            f1 = np.sum(X**2, axis=1)
            f2 = np.prod(np.cos(X / np.sqrt(i)), axis=1)
            return 1 / 4000 * f1 - f2 + 1

    class Penalized1:
        def __init__(self, dim: int = 30, lb: float = -50.0, ub: float = 50.0):
            self.name = "Penalized1"
            self.lb = np.full(dim, lb)
            self.ub = np.full(dim, ub)
            self.opt_f = 0
            self.opt_x = np.full(dim, -1)

        def evaluate(self, X: np.array) -> np.array:
            def y(X):
                F = 1 + (X + 1) / 4
                return F

            def u(X, a, k, m):
                F = np.zeros_like(X)
                mask1 = X > a
                mask3 = X < -a
                mask2 = ~(mask1 + mask3)

                F[mask1] = k * (X[mask1] - a) ** m
                F[mask2] = 0
                F[mask3] = k * (-X[mask3] - a) ** m
                return np.sum(F, axis=1)

            D = X.shape[1]
            y1 = y(X[:, 0])
            yD = y(X[:, -1])
            yi = y(X[:, :-1])
            yi_1 = y(X[:, 1:])

            f1 = 10 * np.sin(np.pi * y1) ** 2
            f2 = np.sum((yi - 1) ** 2 * (1 + 10 * np.sin(np.pi * yi_1) ** 2), axis=1)
            f3 = (yD - 1) ** 2
            return np.pi / D * (f1 + f2 + f3) + u(X, 10, 100, 4)

    class Penalized2:
        def __init__(self, dim: int = 30, lb: float = -50.0, ub: float = 50.0):
            self.name = "Penalized2"
            self.lb = np.full(dim, lb)
            self.ub = np.full(dim, ub)
            self.opt_f = 0
            self.opt_x = np.full(dim, 1)

        def evaluate(self, X: np.array) -> np.array:
            def y(X):
                F = 1 + (X + 1) / 4
                return F

            def u(X, a, k, m):
                F = np.zeros_like(X)
                mask1 = X > a
                mask3 = X < -a
                mask2 = ~(mask1 + mask3)

                F[mask1] = k * (X[mask1] - a) ** m
                F[mask2] = 0
                F[mask3] = k * (-X[mask3] - a) ** m
                return np.sum(F, axis=1)

            X1 = X[:, 0]
            XD = X[:, -1]
            Xi = X[:, :-1]
            Xi_1 = X[:, 1:]

            f1 = np.sin(3 * np.pi * X1) ** 2
            f2 = np.sum((Xi - 1) ** 2 * (1 + np.sin(3 * np.pi * Xi_1) ** 2), axis=1)
            f3 = (XD - 1) ** 2 * (1 + np.sin(2 * np.pi * XD) ** 2)
            return 0.1 * (f1 + f2 + f3) + u(X, 5, 100, 4)

    class ShekelFoxholes:
        def __init__(self, dim: int = 2, lb: float = -65.536, ub: float = 65.536):
            self.name = "ShekelFoxholes"
            self.lb = np.full(2, lb)
            self.ub = np.full(2, ub)
            self.opt_f = 0.998003837794449325873406851315
            self.opt_x = np.full(2, -31.97833)

        def evaluate(self, X: np.array) -> np.array:
            P = X.shape[0]
            F = np.zeros([P])
            j = np.arange(25) + 1
            a1 = np.tile(np.array([-32, -16, 0, 16, 32]), 5)
            a2 = np.repeat(np.array([-32, -16, 0, 16, 32]), 5)
            X1 = X[:, 0]
            X2 = X[:, 1]

            for i in range(P):
                f1 = j + (X1[i] - a1) ** 6 + (X2[i] - a2) ** 6
                F[i] = (1 / 500 + np.sum(1 / f1)) ** -1
            return F

    class Kowalik:
        def __init__(self, dim: int = 4, lb: float = -5.0, ub: float = 5.0):
            self.name = "Kowalik"
            self.lb = np.full(4, lb)
            self.ub = np.full(4, ub)
            self.opt_f = 0.00030748610
            self.opt_x = np.array([0.192833, 0.190836, 0.123117, 0.135766])

        def evaluate(self, X: np.array) -> np.array:
            P = X.shape[0]
            F = np.zeros([P])
            a = np.array(
                [
                    0.1957,
                    0.1947,
                    0.1735,
                    0.1600,
                    0.0844,
                    0.0627,
                    0.0456,
                    0.0342,
                    0.0323,
                    0.0235,
                    0.0246,
                ]
            )
            b = np.array(
                [4, 2, 1, 1 / 2, 1 / 4, 1 / 6, 1 / 8, 1 / 10, 1 / 12, 1 / 14, 1 / 16]
            )
            X1 = X[:, 0]
            X2 = X[:, 1]
            X3 = X[:, 2]
            X4 = X[:, 3]

            for i in range(P):
                f1 = X1[i] * (b**2 + b * X2[i])
                f2 = b**2 + b * X3[i] + X4[i]
                F[i] = np.sum((a - f1 / f2) ** 2)
            return F

    class SixHumpCamelBack:
        def __init__(self, dim: int = 2, lb: float = -5.0, ub: float = 5.0):
            self.name = "SixHumpCamelBack"
            self.lb = np.full(2, lb)
            self.ub = np.full(2, ub)
            self.opt_f = -1.031628453489877
            self.opt_x = f"{np.array([-0.08984201368301331, 0.7126564032704135])} or {np.array([0.08984201368301331, -0.7126564032704135])}"

        def evaluate(self, X: np.array) -> np.array:
            X1 = X[:, 0]
            X2 = X[:, 1]
            return 4 * X1**2 - 2.1 * X1**4 + X1**6 / 3 + X1 * X2 - 4 * X2**2 + 4 * X2**4

    class Branin:
        def __init__(self, dim: int = 2, lb: float = -5.0, ub: float = 5.0):
            self.name = "Branin"
            self.lb = np.full(2, lb)
            self.ub = np.full(2, ub)
            self.opt_f = 0.39788735772973816
            self.opt_x = f"{np.array([-np.pi, 12.275])} or {np.array([np.pi, 2.275])} or {np.array([9.42478, 2.475])}"

        def evaluate(self, X: np.array) -> np.array:
            X1 = X[:, 0]
            X2 = X[:, 1]

            f1 = (X2 - 5.1 * X1**2 / (4 * np.pi**2) + 5 * X1 / np.pi - 6) ** 2
            f2 = 10 * (1 - 1 / (8 * np.pi)) * np.cos(X1)
            return f1 + f2 + 10

    class GoldsteinPrice:
        def __init__(self, dim: int = 2, lb: float = -2.0, ub: float = 2.0):
            self.name = "GoldsteinPrice"
            self.lb = np.full(2, lb)
            self.ub = np.full(2, ub)
            self.opt_f = 3
            self.opt_x = np.array([0, -1])

        def evaluate(self, X: np.array) -> np.array:
            X1 = X[:, 0]
            X2 = X[:, 1]

            f1 = 1 + (X1 + X2 + 1) ** 2 * (
                19 - 14 * X1 + 3 * X1**2 - 14 * X2 + 6 * X1 * X2 + 3 * X2**2
            )
            f2 = 30 + (2 * X1 - 3 * X2) ** 2 * (
                18 - 32 * X1 + 12 * X1**2 + 48 * X2 - 36 * X1 * X2 + 27 * X2**2
            )
            return f1 * f2

    class Hartmann3:
        def __init__(self, dim: int = 3, lb: float = 0.0, ub: float = 1.0):
            self.name = "Hartmann3"
            self.lb = np.full(3, lb)
            self.ub = np.full(3, ub)
            self.opt_f = -3.86278214782076
            self.opt_x = np.array([0.1, 0.55592003, 0.85218259])

        def evaluate(self, X: np.array) -> np.array:
            P = X.shape[0]
            F = np.zeros([P])
            alpha = np.array([1.0, 1.2, 3.0, 3.2])
            A = np.array(
                [
                    [3.00, 10.0, 30.0],
                    [0.10, 10.0, 35.0],
                    [3.00, 10.0, 30.0],
                    [0.10, 10.0, 35.0],
                ]
            )

            P = np.array(
                [
                    [0.36890, 0.1170, 0.2673],
                    [0.46990, 0.4387, 0.7470],
                    [0.10910, 0.8732, 0.5547],
                    [0.03815, 0.5743, 0.8828],
                ]
            )

            for i in range(4):
                f1 = alpha[i] * np.exp(-np.sum(A[i] * (X - P[i]) ** 2, axis=1))
                F = F + f1
            return -F

    class Hartmann6:
        def __init__(self, dim: int = 3, lb: float = 0.0, ub: float = 1.0):
            self.name = "Hartmann6"
            self.lb = np.full(6, lb)
            self.ub = np.full(6, ub)
            self.opt_f = -3.32236801141551
            self.opt_x = np.array(
                [0.20168952, 0.15001069, 0.47687398, 0.27533243, 0.31165162, 0.65730054]
            )

        def evaluate(self, X: np.array) -> np.array:
            P = X.shape[0]
            F = np.zeros([P])
            alpha = np.array([1.0, 1.2, 3.0, 3.2])
            A = np.array(
                [
                    [10.0, 3.00, 17.0, 3.50, 1.70, 8.00],
                    [0.05, 10.0, 17.0, 0.10, 8.00, 14.0],
                    [3.00, 3.50, 1.70, 10.0, 17.0, 8.00],
                    [17.0, 8.00, 0.05, 10.0, 0.10, 14.0],
                ]
            )
            P = np.array(
                [
                    [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                    [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                    [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
                    [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
                ]
            )

            for i in range(4):
                f1 = alpha[i] * np.exp(-np.sum(A[i] * (X - P[i]) ** 2, axis=1))
                F = F + f1
            return -F

    class Shekel5:
        def __init__(self, dim: int = 4, lb: float = 0.0, ub: float = 10.0):
            self.name = "Shekel"
            self.lb = np.full(4, lb)
            self.ub = np.full(4, ub)
            self.opt_f = -10.1532
            self.opt_x = np.array([4, 4, 4, 4])
            self.m = 5

        def evaluate(self, X: np.array) -> np.array:
            P = X.shape[0]
            F = np.zeros([P])
            a = np.array(
                [
                    [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                    [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
                    [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                    [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
                ]
            )
            c = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])

            for i in range(self.m):
                f1 = np.sum((X - a[:, i]) ** 2, axis=1) + c[i]
                F = F + 1 / f1
            return -F

    class Shekel7:
        def __init__(self, dim: int = 4, lb: float = 0.0, ub: float = 10.0):
            self.name = "Shekel"
            self.lb = np.full(4, lb)
            self.ub = np.full(4, ub)
            self.opt_f = -10.4029
            self.opt_x = np.array([4, 4, 4, 4])
            self.m = 7

        def evaluate(self, X: np.array) -> np.array:
            P = X.shape[0]
            F = np.zeros([P])
            a = np.array(
                [
                    [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                    [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
                    [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                    [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
                ]
            )
            c = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])

            for i in range(self.m):
                f1 = np.sum((X - a[:, i]) ** 2, axis=1) + c[i]
                F = F + 1 / f1
            return -F

    class Shekel10:
        def __init__(self, dim: int = 4, lb: float = 0.0, ub: float = 10.0):
            self.name = "Shekel10"
            self.lb = np.full(4, lb)
            self.ub = np.full(4, ub)
            self.opt_f = -10.5364
            self.opt_x = np.array([4, 4, 4, 4])
            self.m = 10

        def evaluate(self, X: np.array) -> np.array:
            P = X.shape[0]
            F = np.zeros([P])
            a = np.array(
                [
                    [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                    [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
                    [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                    [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
                ]
            )
            c = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])

            for i in range(self.m):
                f1 = np.sum((X - a[:, i]) ** 2, axis=1) + c[i]
                F = F + 1 / f1
            return -F
