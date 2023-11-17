import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.utils.helper_funcs import create_box_grid

def plot_level_sets_and_points(points:np.ndarray, scalar_field, filename: str="plots/misc/test.png"):
    x = np.linspace(min(points[:, 0]) - 1, max(points[:, 0]) + 1, 100)
    y = np.linspace(min(points[:, 1]) - 1, max(points[:, 1]) + 1, 100)
    X, Y = np.meshgrid(x, y)

    # Plot level sets
    plt.contour(X, Y, scalar_field(np.stack((X, Y), axis=0)), cmap='viridis')

    # Scatter plot for points
    plt.plot(points[:, 0], points[:, 1], color='red', label='Points')
    for i, (x, y) in enumerate(points):
        plt.text(x, y, f'{i+1}', color='black', fontsize=8, ha='center', va='center')


    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Level Sets and Points')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)


def surface_plot(X, Y, Z, ax=None, cmap="viridis", save=None):
    """
    Create a surface plot on the given axes.

    Parameters:
        ax: Axes3D object.
        X (numpy.ndarray): Meshgrid of x values.
        Y (numpy.ndarray): Meshgrid of y values.
        Z (numpy.ndarray): Corresponding z values.
        cmap (str): Colormap name (default: 'viridis').
    """
    if ax is None:
        fig = plt.figure()  
        ax = fig.add_subplot(111, projection='3d') 
        

    ax.plot_surface(X, Y, Z, cmap=cmap)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Surface Plot')
    
    #fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
    if not (save is None):
        plt.savefig(save)
    return ax

def contour(X, Y, Z, ax=None, cmap="viridis", save=None):
    if ax is None:
        fig, ax = plt.subplots()
    contour = ax.contour(X, Y, Z, cmap=cmap)

    #fig.colorbar(contour, shrink=0.5, aspect=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Level Set Plot')
    if not (save is None):
        plt.savefig(save)
    return ax

def add_iterates_contour(points, ax=None):
    # Create a list of plot data for random points
    random_point_data = [{'position': point, 'marker': 'x',"color": "blue",  'markersize': 8, 'label': 'Random Point'} for point in points[:-1]]
    random_point_data.append({'position': points[-1], 'marker': 'x', 'color': 'red', 'markersize': 8, 'label': 'Last Point'})

    # Plot the random points using the list of data
    for data in random_point_data:
        position=data.pop("position")
        ax.plot(position[0], position[1], **data)
    return ax

# # Define a function to generate Z values
# def f(x, y):
#     return x**2-y**2

# def g(x, y):
#     return x**2+y**2

# X, Y=create_box_grid(-1, 1, 100)

# funcs=[f,g]
# Z=f(X,Y)
# save="plots/contour_plot.jpg"
# ax=contour(X, Y, Z, ax=None, cmap="viridis")

# num_random_points = 5
# random_points = np.random.rand(num_random_points, 2)
# ax=add_iterates_contour(random_points, ax=ax)

# plt.savefig(save)


# for i, func in enumerate(funcs):
#     Z = func(X, Y)
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     save=f"plots/Example_plot_{i}.jpg"
#     surface_plot(X,Y,Z, ax=ax, save=save)
    

