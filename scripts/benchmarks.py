import numpy as np

def get_sphere(x, y):
    return x**2 + y**2

def get_rastrigin(x, y, A=10):
    return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))

FUNCTIONS_MAP = {
    "Sphere": get_sphere,
    "Rastrigin": get_rastrigin
}