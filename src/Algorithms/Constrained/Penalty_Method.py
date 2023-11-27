from src.Algorithms.Unconstrained_Algorithms.descent_methods import Newton
from scipy.optimize import minimize
import numpy as np
from typing import Union


class Penalty():

    def __init__(self, f,df, ddf, g,dg, ddg, h, dh, ddh, maxiter: int=100000, tol:float =10**-6, alpha:float =10**-3, beta: float =2):
        """Sovles the constrained optimization Problem min_x f, g(x)<=0, h(x)=0 g: R^n-> R^m, h: R^n -> R^p

        Args:
            f (function): f:R^n->R
            df (_type_): R^n->R^n
            ddf: R^n -> R^{n x n}
            g (_type_): R^n-> R^m
            dg (_type_): R^n -> R^{mxn}
            ddg : R^n -> L(R^n, R^{m x n})
            h (_type_): R^n -> R^p
            dh (_type_): R^n-> R^{pxn}
            ddh : R^n -> L(R^n, R^{p x n})
            maxiter (int, optional): Defaults to 100000.
            tol (float, optional): Defaults to 10**-9.
            alpha (float, optional): Defaults to 10**-3.
            beta (float, optional): Defaults to 2.
        """
        self.f=f
        self.df=df
        self.ddf=ddf
        self.maxiter=maxiter
        self.tol=tol
        self.xs=[]

        self.g=g
        self.dg=dg
        self.ddg=ddg
        self.ddh=ddh
        self.h=h
        self.dh=dh

        self.alpha=alpha
        self.beta=beta

    

    def solve(self, x0: np.ndarray)-> Union[None, np.ndarray]:
        x=x0
        f=self.f
        df=self.df
        ddf=self.ddf
        alpha=self.alpha
        g=self.g
        dg=self.dg
        ddg=self.ddg
        h=self.h
        dh=self.dh
        ddh=self.ddh
        xs=self.xs
        def P(x,a):
            gpos=np.maximum(0, g(x))
            return f(x)+a/2*(np.linalg.norm(gpos)**2+np.linalg.norm(h(x))**2)
        def dP(x, a):
            return df(x)+a*np.sum(np.maximum(0, g(x)) * dg(x), axis=0)+a*np.sum(h(x) * dh(x), axis=0)
        def ddP(x, a):
            D=lambda x: np.diag(np.array(g(x)>=0, float))
            return ddf(x) + a* np.matmul(np.matmul(dg(x), D(x)), dg(x).transpose())+np.sum(np.inner(np.maximum(g(x), 0), ddg(x)), axis=0) +a*np.outer(h(x), h(x)) +a*np.sum(h(x)*ddh(x), axis=0)
        for i in range(self.maxiter):
            xs.append(x)
            #newton=Newton(lambda x: P(alpha, x), lambda x:dP(alpha, x), lambda x: ddP(alpha, x), self.maxiter, self.tol)
            #xnew=newton.solve(x)[-1]
            newres=minimize(P, x, args=(alpha), method="Newton-CG", jac=dP, hess=ddP)
            xnew=newres.x
            norm=np.linalg.norm(dP(alpha, xnew))
            stoping=self.tol* max(np.linalg.norm(df(x)), 1)
            if norm<=stoping:
                print(f"Penalty method converged at iteration {i}")
                return xnew
            print(f"Iteration {i}: Norm of the gradient of penalty function with norm: {norm} with a difference to goal: {stoping-norm}")
            x=xnew
            alpha=self.beta*alpha
        print(f"Penalty method stoped beacause maximum iteration reached")
        return None


