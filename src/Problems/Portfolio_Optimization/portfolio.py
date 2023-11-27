from scipy.io import loadmat
import numpy as np
from src.Algorithms.Constrained.Penalty_Method import Penalty
from scipy.optimize import minimize

if __name__=="__main__":
    data=loadmat("src/Problems/Portfolio_Optimization/data.mat")
    Q=data["Q"]
    mu=data["mu"].squeeze()
    stocks=data["stocks"]
    n=len(mu)
    x=1/n*np.ones(n)
    kappa=10
    alpha=kappa/10
    beta=10
    eps=10**-6


    df= lambda x: kappa*np.inner(Q, x)-mu
    

    def P(x, alpha, Q, kappa, mu):
        gpos=np.maximum(0, -x)
        return kappa/2*np.inner(x, np.dot(Q, x))-np.inner(mu, x)+alpha/2*(np.linalg.norm(gpos)**2+(np.sum(x)-1)**2)

    def dP(x, alpha, Q, kappa, mu):
        gpos=np.maximum(0, -x)
        return kappa*np.dot(Q, x)-mu+alpha*(-np.matmul(np.eye(len(mu)), gpos)+(np.sum(x)-1)*np.ones(len(mu)))
    
    def ddP(x, alpha, Q, kappa, mu):
        return kappa*Q+alpha*np.diag(np.array(x<=0, float))+alpha*np.eye(len(mu))
    # penalty_method=Penalty(f,df,ddf,g,dg,ddg, h,dh, ddh,beta=beta, alpha=alpha, maxiter=1000)
    # penalty_method.solve(x)
    maxiter=1000
    for i in range(maxiter):
        #newton=Newton(lambda x: P(alpha, x), lambda x:dP(alpha, x), lambda x: ddP(alpha, x), self.maxiter, self.tol)
        #xnew=newton.solve(x)[-1]
        newres=minimize(P, x, args=(alpha, Q, kappa, mu), method="Newton-CG", jac=dP, hess=ddP)
        xnew=newres.x
        norm=np.linalg.norm(dP(xnew, alpha, Q, kappa, mu))
        stoping=eps* max(np.linalg.norm(df(x)), 1)
        if norm<=stoping:
            print(f"Penalty method converged at iteration {i}")
            print(xnew)
        print(f"Iteration {i}: Norm of the gradient of penalty function with norm: {norm} with a difference to goal: {norm-stoping} and alpha {alpha}")
        x=xnew
        alpha=beta*alpha
    print(f"Penalty method stoped beacause maximum iteration reached")

