#from lib2to3.pgen2.pgen import DFAState
#from multiprocessing.reduction import steal_handle
import numpy as np
from scipy.linalg import solve



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
        self.beta=beta
        self.gamma=gamma
        self.eta=eta
        self.tol=tol
        self.maxiter=maxiter
        self.step_size_choice=step_size_choice
        self.p=p
        self.alpha=alpha

        self.optimizer=optimizer

    def descent_direction(self, x):
        #TODO implement other
        return -self.df(x)


    def step_size(self, x, s):
        if self.step_size_choice=="armijo":
            return self.armijo(x, s)
        elif self.step_size_choice=="powell":
            return self.powell(x,s)
        else:   #constant
            return 0.1

    def armijo(self,x,s):
        sigma=1
        while True:         #This will stop by Lemma 7.5 in [1]
            if self.armijo_check( x, s, sigma):
                return sigma
            else :
                sigma=sigma*self.beta

    def armijo_check(self, x, s, sigma):
        return self.f(x+sigma*s)-self.f(x)<= sigma*self.gamma* np.inner(self.df(x), s)

    def powell(self, x, s):
        """
            Algorithm 9.3 in [1]
            Requirements:
                -   f is continously differentiable
                -   s satisfies dfx' s < 0
                -   f is bounded from below in direction s
        """

        sigma=1
        if self.armijo_check(x, s, sigma):
            if self.powell_check(x,s,sigma):
                return 1
            else:
                sigma_max=2
                while True:         
                    if self.armijo_check( x, s, sigma_max):
                        sigma_max=2*sigma_max
                        
                    else :
                        sigma_min=0.5*sigma_max
                        break
        else:
            sigma_min=0.5 
            while True:         
                if self.armijo_check( x, s, sigma_min):
                    sigma_max=2*sigma_min
                    break
                else :
                    sigma_min=sigma_min*0.5
        while True:
            if self.powell_check(x, s, sigma_min):
                return sigma_min
            else:
                sigma=0.5*(sigma_min+sigma_max)
                if self.armijo_check(x,s, sigma):
                    sigma_min=sigma
                else:
                    sigma_max=sigma

    def powell_check(self, x, s, sigma):
        return np.inner(self.df(x+sigma*s), s)>=self.eta*np.inner(self.df(x), s)

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
