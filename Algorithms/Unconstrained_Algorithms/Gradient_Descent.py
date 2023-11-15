#from lib2to3.pgen2.pgen import DFAState
#from multiprocessing.reduction import steal_handle
import numpy as np
from scipy.linalg import solve
from ..stepsizing import Armijo, Powellwolfe


class unconstrained_opt():
    """
        An implementation of the gradient descent algorithm. For a reference consider e.g. https://link.springer.com/book/10.1007/978-3-0346-0654-7 [1].
        All credit goes to the authors of the book.

        Requirements:
        -   f is required to be continously differentiable on a domain U \subset \mathbb{R}^n
        -   if f is coercive, then powell wolfe is feasible step size Theorem 9.5 [1]
    """


    def __init__(self,f, df, ddf=0, optimizer="gradient_descent", beta=0.5, gamma=10**-2, eta=2*10**-2, tol=10**-8, p=2, alpha=[1,1], maxiter=1000, step_size_choice="armijo"):
        self.f=f    
        self.df=df
        self.ddf=ddf
        self._set_step_sizer(step_size_choice, gamma, beta, eta)
        self.eta=eta
        self.tol=tol
        self.maxiter=maxiter
        self.p=p
        self.alpha=alpha

        self.optimizer=optimizer

    def _set_step_sizer(self, step_size_choice, gamma, beta, eta):
        if step_size_choice=="armijo":
            self.step_sizer=Armijo(beta, gamma)
        elif step_size_choice=="powell":
            self.step_sizer=Powellwolfe(gamma, eta)
        else:
            raise Exception(f"Step size choice {step_size_choice} unknown.")

    def descent_direction(self, x):
        #TODO implement other
        return -self.df(x)


    def step_size(self, x, s):
        return self.step_sizer.step(self.f, self.df, x,s, self.maxiter)

    def angle_test(self,dfx,dk):
        return -np.inner(dfx, dk)>=min(self.alpha[0], self.alpha[1]* np.linalg.norm(dk)**self.p)*np.linalg.norm(dk)**2

    def solve(self,x0):
        if self.optimizer=="gradient_descent":
            x=np.array(x0)
            for i in range(self.maxiter):
                fx=self.f(x)
                dfx=self.df(x)
                norm_dfx=np.linalg.norm(dfx)
                if norm_dfx<self.tol:
                    print("Iteration: {:d}, Function value: {:1.5}, Norm of the gradient: {:1.8}".format(i, float(fx), norm_dfx))
                    print("Solver Converged!")
                    return x
                sk=self.descent_direction(x)
                sigmak=self.step_size(x, sk)
                x=x+sigmak*sk
                print("Iteration: {:d}, Function value: {:1.5}, Norm of the gradient: {:1.8}".format(i, float(fx), norm_dfx))
            print("Solver stopped with maxiteration!")
            return x
        elif self.optimizer=="Newton":
            # [1] Algorithm 10.9
            x=np.array(x0)
            for i in range(self.maxiter):
                fx=self.f(x)
                dfx=self.df(x)
                ddfx=self.ddf(x)
                norm_dfx=np.linalg.norm(dfx)
                if norm_dfx<self.tol:
                    print("Iteration: {:d}, Function value: {:1.5}, Norm of the gradient: {:1.8}".format(i, float(fx), norm_dfx))
                    print("Solver Converged!")
                    return x
                dk=solve(ddfx,-dfx, assume_a="sym")        
                if self.angle_test:
                    sk=dk
                else:
                    sk=-dfx
                sigmak=self.step_size(x, sk)
                x=x+sigmak*sk
                print("Iteration: {:d}, Function value: {:1.5}, Norm of the gradient: {:1.8}".format(i, float(fx), norm_dfx))
            print("Solver stopped with maxiteration!")
            return x
        elif self.optimizer=="BFGS":
            x=np.array(x0)
            fx=self.f(x)
            dfx=self.df(x)
            norm_dfx=np.linalg.norm(dfx)
            if norm_dfx<self.tol:
                    print("Iteration: {:d}, Function value: {:1.5}, Norm of the gradient: {:1.8}".format(i, float(fx), norm_dfx))
                    print("Solver Converged!")
                    return x
            if self.ddf==0:
                B0=np.eye(x.shape[0])       #if second derivative wasn't given, take the identity
            else:
                B0=self.ddf(x)           #if second derivative given, compute hessian once, and take it as B_0
            B=B0
            for i in range(self.maxiter):
                s=-np.matmul(B, dfx)
                if not self.angle_test(dfx, s):
                    B=B0
                    s=-np.matmul(B, dfx)
                sigma=self.powell(x, s)
                x_old=x
                dfx_old=dfx
                x=x+sigma*s
                fx=self.f(x)
                dfx=self.df(x)
                norm_dfx=np.linalg.norm(dfx)
                if norm_dfx<self.tol:
                    print("Iteration: {:d}, Function value: {:1.5}, Norm of the gradient: {:1.8}".format(i, float(fx), norm_dfx))
                    print("Solver Converged!")
                    return x
                d=x-x_old
                y=dfx-dfx_old
                inner=np.inner(d, y)
                diff=d-np.matmul(B, y)
                B=B+1/inner*(np.outer(diff,d)+np.outer(d,diff))-1/inner**2*np.inner(diff,y)*np.outer(d,d)
                print("Iteration: {:d}, Function value: {:1.5}, Norm of the gradient: {:1.8}".format(i, float(fx), norm_dfx))
            print("Solver stopped with maxiteration!")
            return x
        else:
            print("Optimizer not implemented!")
