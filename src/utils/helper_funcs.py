import numpy as np

def create_mesh(min_x: float, max_x: float, min_y: float, max_y: float, fine_x: int, fine_y: int):
    """
    Create a meshgrid of x and y values.

    Parameters:
        min_x (float): Minimum x value.
        max_x (float): Maximum x value.
        min_y (float): Minimum y value.
        max_y (float): Maximum y value.
        fine_x (int): Number of points in the x direction.
        fine_y (int): Number of points in the y direction.

    Returns:
        X (numpy.ndarray): Meshgrid of x values.
        Y (numpy.ndarray): Meshgrid of y values.
    """
    x = np.linspace(min_x, max_x, fine_x)
    y = np.linspace(min_y, max_y, fine_y)
    X, Y = np.meshgrid(x, y)
    return X, Y

def create_box_grid(min: float, max: float, fine:int):
    return create_mesh(min, max, min, max, fine, fine)

def unit_mesh_grid(fine=100):
    return create_box_grid(0, 1, fine)


def generateSPDmatrix(n):
    A = np.random.rand(n,n) 
    A = 0.5* (A+A.T)
    
    A = A + n*np.eye(n)
    return A