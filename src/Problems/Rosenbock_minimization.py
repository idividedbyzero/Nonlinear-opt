#from Algorithms import Unconstrained_Algorithms
#from . import *
#from Gradient_Descent import unconstrained_opt
from src.Algorithms.Unconstrained_Algorithms.Gradient_Descent import unconstrained_opt
import numpy as np
import logging

def Rosenbock(x):
    return 100*(x[1]-x[0]**2)**2+(1-x[0])**2
def dRosenbock(x):
    return np.array([2*(200*x[0]**3-200*x[0]*x[1]+x[0]-1), 200*(x[1]-x[0]**2)])
def ddRosenbock(x):
    return np.array([[-400*(x[1]-x[0]**2)+800*x[0]**2+2, -400*x[0]], [-400*x[0], 200]])



if __name__=="__main__":
    solver=unconstrained_opt(Rosenbock,dRosenbock, maxiter=100000)
    logging.info("Optimizing with Gradient descent")
    solver.solve(np.array([-1.2, 1]))

    solver=unconstrained_opt(Rosenbock,dRosenbock, ddRosenbock, optimizer="Newton")
    logging.info("Optimizing with Newton's method")
    solver.solve(np.array([-1.2,1]))

    logging.info("Finished all optimization processes")