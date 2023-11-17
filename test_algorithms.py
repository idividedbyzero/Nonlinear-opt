import numpy as np
from src.utils.Taylor_Test import taylor_test
from src.utils.helper_funcs import generateSPDmatrix

from src.Algorithms.Unconstrained_Algorithms.Quadratic_solver import conjugate_gradient


def test_cg():
    A=generateSPDmatrix(10)
    #np.linalg.eigh(A)       #genreate positive definit matrix A
    b=np.random.rand(10)
    cg_sol=conjugate_gradient(A, b).solve(np.ones(10))
    linalg_solve=np.linalg.solve(A, b)
    assert np.linalg.norm(cg_sol-linalg_solve)<10**-4

def test_taylor():
    def f(x):
        return np.sin(x)
    def df(x):
        return np.cos(x)
    def ddf(x):
        return -np.sin(x)

    for i in range(3):
        rate=taylor_test(f,df,ddf, 1, float(1), order =i)
        assert np.abs(rate- (i+1))<10**-2, f"Convergence rate missmatch {rate} with error {np.abs(rate- (i+1))}"

if __name__=="__main__":
    test_cg()
    test_taylor()