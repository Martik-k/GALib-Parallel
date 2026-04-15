import numpy as np

def get_sphere(x, y):
    return x**2 + y**2

def get_rastrigin(x, y, A=10):
    return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))

def get_himmelblau(x, y):
    """
    Himmelblau's function.
    4 global minima. Best visualized on bounds x, y in [-5, 5].
    """
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def get_dejong_f5(x, y):
    """
    De Jong F5 (Shekel's Foxholes).
    25 local minima. Best visualized on bounds x, y in [-65.536, 65.536].
    """
    # X and Y coordinates for the 25 foxholes
    a_x = [-32, -16, 0, 16, 32] * 5
    a_y = [-32]*5 + [-16]*5 + [0]*5 + [16]*5 + [32]*5

    sum_term = 0.0
    for i in range(25):
        # i + 1 is used to match the 1-indexed mathematical formula
        term1 = (x - a_x[i])**6
        term2 = (y - a_y[i])**6
        sum_term += 1.0 / ((i + 1) + term1 + term2)

    return 1.0 / (0.002 + sum_term)

# Map string names to the function references
FUNCTIONS_MAP = {
    "Sphere": get_sphere,
    "Rastrigin": get_rastrigin,
    "Himmelblau": get_himmelblau,
    "DeJongF5": get_dejong_f5
}