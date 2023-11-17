#from lib2to3.pgen2.pgen import DFAState
#from multiprocessing.reduction import steal_handle
import numpy as np
from scipy.linalg import solve
from ..stepsizing import Armijo
from src.utils.ploting import plot_level_sets_and_points
from typing import List

class general_descent_method():
    def __init__(self, f,df, maxiter: int=100000, tol:float =10**-9):
        self.f=f
        self.df=df
        self.maxiter=maxiter
        self.tol=tol
        self.xs=None

    def descent_direction(self, x):
        pass

    def step_size(self, x, s):
        pass
    
    def solve(self, x0:np.ndarray)-> np.ndarray:
            xs=[]
            x=np.array(x0)        
            for i in range(self.maxiter):
                xs.append(x)
                fx=self.f(x)
                dfx=self.df(x)
                norm_dfx=np.linalg.norm(dfx)
                if norm_dfx<self.tol:
                    print("Iteration: {:d}, Function value: {:1.5}, Norm of the gradient: {:1.8}".format(i, float(fx), norm_dfx))
                    print("Solver Converged!")
                    xs.append(x)
                    self.xs=np.array(xs)
                    return xs
                sk=self.descent_direction(x)
                sigmak=self.step_size(x,sk)
                x=x+sigmak*sk
                print("Iteration: {:d}, Function value: {:1.5}, Norm of the gradient: {:1.8}".format(i, float(fx), norm_dfx))
            print("Solver stopped with maxiteration!")
            self.xs=np.array(xs)
            return xs
    
    def plot_level_sets_and_points(self, filename:str="plots/misc/test.png"):
        xs=self.xs
        if xs is None:
           xs=self.solve(np.random.uniform(2)) 
        plot_level_sets_and_points(xs, self.f, filename)



class Gradient_Descent(general_descent_method):

    def __init__(self, f, df, maxiter: int = 100000, tol: float = 10 ** -9, beta:float=1/2, gamma:float=10**-2):
        super().__init__(f, df, maxiter, tol)
        self.beta=beta
        self.gamma=gamma

    def descent_direction(self, x):
        return -self.df(x)
    
    def step_size(self, x, s):
        armijo=Armijo(self.beta, self.gamma)
        return armijo.step(self.f, self.df, x, s)
    
    def plot_level_sets_and_points(self, filename: str = "plots/misc/gradient_descent.png"):
        return super().plot_level_sets_and_points(filename)
    
class Newton(general_descent_method):
    def __init__(self, f, df, ddf, maxiter: int = 100000, tol: float = 10 ** -9, beta:float=1/2, gamma:float=10**-2, alpha:List[float]=[10**-6, 10**-6], p:float=1/10):
        super().__init__(f, df, maxiter, tol)
        self.ddf=ddf
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.p=p

    
    def _angle_test(self,dfx,dk)->bool:
        return -np.inner(dfx, dk)>=min(self.alpha[0], self.alpha[1]* np.linalg.norm(dk)**self.p)*np.linalg.norm(dk)**2

    def descent_direction(self, x):
        assert not self.ddf is None, "No second derivative was given"
        dfx=self.df(x)
        ddfx=self.ddf(x)
        dk=solve(ddfx,-dfx, assume_a="sym")    #hessian is always symmetric     
        if self._angle_test:
            sk=dk
        else:
            sk=-dfx
        return sk
    
    def step_size(self, x, s):
        return 1
    
    def plot_level_sets_and_points(self, filename: str = "plots/misc/newtonmethod.png"):
        return super().plot_level_sets_and_points(filename)



# class unconstrained_opt():
#     """
#         An implementation of the gradient descent algorithm. For a reference consider e.g. https://link.springer.com/book/10.1007/978-3-0346-0654-7 [1].
#         All credit goes to the authors of the book.

#         Requirements:
#         -   f is required to be continously differentiable on a domain U \subset \mathbb{R}^n
#         -   if f is coercive, then powell wolfe is feasible step size Theorem 9.5 [1]
#     """


#     def __init__(self,f, df, ddf=None, optimizer="gradient_descent", **kwargs):
#         #beta=0.5, gamma=10**-2, eta=2*10**-2, tol=10**-8, p=2, alpha=[1,1], maxiter=1000, step_size_choice="armijo"
#         self.f=f    
#         self.df=df
#         self.ddf=ddf
#         self._set_step_sizer(kwargs.pop("Stepsize_parameters", {"choice": "armijo", "gamma": 10**-2, "beta": 2*10**-2}))
#         self.optimizer=optimizer

#     def _set_step_sizer(self, parameters: dict):
#         choice=parameters.get("choice")
#         if choice=="armijo":
#             self.step_sizer=Armijo(parameters.get("beta"), parameters.get("gamma"))
#         elif choice=="powell":
#             self.step_sizer=Powellwolfe(parameters.get("gamma"), parameters.get("eta"))
#         else:
#             raise Exception(f"Step size choice {choice} unknown.")


#     def solve(self,x0, **kwargs):
#         optimizer_kwargs=kwargs.pop("optimizer_parameters")
#         choice=optimizer_kwargs.pop("choice")
#         if choice.lower()=="gradient_descent":
#             gd=Gradient_Descent(self.f,self.df, self.step_sizer, **optimizer_kwargs)
#             xs=gd.solve(x0)
#             return xs[:-1]
#         elif choice.lower()=="newton":
            
#             return x
#         elif choice.lower()=="bfgs":
#             x=np.array(x0)
#             fx=self.f(x)
#             dfx=self.df(x)
#             norm_dfx=np.linalg.norm(dfx)
#             if norm_dfx<self.tol:
#                     print("Iteration: {:d}, Function value: {:1.5}, Norm of the gradient: {:1.8}".format(i, float(fx), norm_dfx))
#                     print("Solver Converged!")
#                     return x
#             if self.ddf==0:
#                 B0=np.eye(x.shape[0])       #if second derivative wasn't given, take the identity
#             else:
#                 B0=self.ddf(x)           #if second derivative given, compute hessian once, and take it as B_0
#             B=B0
#             for i in range(self.maxiter):
#                 s=-np.matmul(B, dfx)
#                 if not self.angle_test(dfx, s):
#                     B=B0
#                     s=-np.matmul(B, dfx)
#                 sigma=self.powell(x, s)
#                 x_old=x
#                 dfx_old=dfx
#                 x=x+sigma*s
#                 fx=self.f(x)
#                 dfx=self.df(x)
#                 norm_dfx=np.linalg.norm(dfx)
#                 if norm_dfx<self.tol:
#                     print("Iteration: {:d}, Function value: {:1.5}, Norm of the gradient: {:1.8}".format(i, float(fx), norm_dfx))
#                     print("Solver Converged!")
#                     return x
#                 d=x-x_old
#                 y=dfx-dfx_old
#                 inner=np.inner(d, y)
#                 diff=d-np.matmul(B, y)
#                 B=B+1/inner*(np.outer(diff,d)+np.outer(d,diff))-1/inner**2*np.inner(diff,y)*np.outer(d,d)
#                 print("Iteration: {:d}, Function value: {:1.5}, Norm of the gradient: {:1.8}".format(i, float(fx), norm_dfx))
#             print("Solver stopped with maxiteration!")
#             return x
#         else:
#             print("Optimizer not implemented!")
