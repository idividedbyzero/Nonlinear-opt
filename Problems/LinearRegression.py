import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.stats as stats

# Add the higher directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
higher_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(higher_dir)

# Now you can import the module from the higher directory
from Unconstrained_Algorithms.Quadratic_solver import conjugate_gradient

class Regression():

    def __init__(self, x_values, y_values,basis="linear", order=None, gaussian_basis=None, logistic_basis=None):
        assert len(x_values)==len(y_values), f"Not the same number of x_values:  {len(x_values)}, and y_values :{len(y_values)}"
        self.x_values=np.array(x_values)
        self.y_values = np.array(y_values)
        self.basis=basis
        self.order=order
        self.gaussian_basis=gaussian_basis
        self.logistic_basis=logistic_basis

        self.system_size=self.y_values.shape

    
    def get_phi(self, x_values, basis: str):        
        ones_array = np.ones(len(x_values))
        if basis=="linear":
            # Horizontally stack xi_values and ones_array
            Phi = np.hstack([ones_array[:, None], x_values[:, None]])
            assert Phi.shape== (len(x_values), 2), "Matrix A has the wrong size"
        elif basis=="polynomial":
             # Horizontally stack xi_values and ones_array
            Phi = np.hstack([x_values[:, None]**i for i in range(self.order)])
            assert Phi.shape== (len(x_values), self.order+1), "Matrix A has the wrong size"
        elif basis=="gaussian":
            means=self.gaussian_basis.get("means", [-1,0,1])
            s=self.gaussian_basis.get("std", 1)
            f=lambda x, i: np.exp(-(x-means[i])**2/(2*s*+2))
             # Horizontally stack xi_values and ones_array
            Phi = np.hstack([x_values[:, None], *[f(x_values[:, None], i) for i in range(len(means))]])
            assert Phi.shape== (len(x_values), len(means)+1), "Matrix A has the wrong size"
        elif basis=="logistic":
            means=self.logistic_basis.get("means", [-1,0,1])
            s=self.logistic_basis.get("std", 1)
            f=lambda x, i: 1/(1+np.exp(-(x-means[i])**2/(2*s*+2)))
             # Horizontally stack xi_values and ones_array
            Phi = np.hstack([x_values[:, None], *[f(x_values[:, None], i) for i in range(len(means))]])
            assert Phi.shape== (len(x_values), len(means)+1), "Matrix A has the wrong size"
        else:
            raise NotImplementedError
        return Phi
    
    def plot(self, **kwargs):
        fig, ax= plt.subplots(1,1)

        ax.scatter(self.x_values, self.y_values, **kwargs)
        return ax



class deterministicReg(Regression):
    """
        Simply solve Phi^tPhix=Phi^ty
    """
    def solve(self, x0=None):
        self.Phi=self.get_phi(self.x_values,self.basis)
        cg=conjugate_gradient(self.Phi, self.y_values)
        self.x_min=cg.solve(x0)
        return self.x_min   
    
    def function(self,x):
        Phi=self.get_phi(x, self.basis)
        return np.dot(Phi, self.x_min)

    
    def plot(self, **kwargs):
        ax=super().plot(**kwargs)
        x_values=sorted(self.x_values)
        x=np.linspace(min(x_values), max(x_values))
        y=self.function(x)
        ax.plot(x, y)
        return ax
    
class bayesianReg(Regression):
    """
        see lecture notes...
    """
    def __init__(self, x_values, y_values,basis="linear", order=None, gaussian_basis=None, logistic_basis=None, alpha=1, beta=10):
        super().__init__(x_values, y_values,basis, order, gaussian_basis, logistic_basis)
        self.alpha=alpha
        self.beta=beta

    @property
    def Sigma_inv(self):
        Phi=self.get_phi(self.x_values, self.basis)
        return self.alpha*np.eye(Phi.shape[1]) + self.beta* np.dot(Phi.T, Phi)
    
    @property
    def mu(self):
        Phi=self.get_phi(self.x_values, self.basis)
        rhs=Phi.T @ self.y_values
        d=self.solve_sigma(rhs)
        return self.beta*d
    
    def solve_sigma(self, rhs):
        # Compute the left-hand side matrix: alpha * I + beta * Phi^T * Phi
        lhs_matrix = self.Sigma_inv
        

        # Solve the linear equation using Cholesky decomposition
        chol_factor = np.linalg.cholesky(lhs_matrix)
        d = np.linalg.solve(chol_factor.T, np.linalg.solve(chol_factor, rhs))
        return d

    def get_post_pred_distr(self, x):
        phi_new=self.get_phi(x, self.basis)
        mean=np.inner(self.mu, phi_new)
        d=np.apply_along_axis(self.solve_sigma, axis=1, arr=phi_new)#self.solve_sigma(phi_new)
        var=1/self.beta+np.diag(np.dot(phi_new, d.T))
        return mean, var
    
    def get_intervals(self, mu, sigma, tail=0.99):
        return stats.norm.interval(tail, loc=mu, scale=sigma)

    
    def plot(self, **kwargs):
        ax=super().plot(**kwargs)
        x_values=sorted(self.x_values)
        x=np.linspace(min(x_values), max(x_values))
        mean, var=self.get_post_pred_distr(x)
        ax.plot(x, mean)
        conf_interval=self.get_intervals(mean, var)
        #for x_val, y_min, y_max in zip(x, conf_interval[0], conf_interval[1]):
        ax.vlines(x=x, ymin=conf_interval[0], ymax=conf_interval[1], color='gray', linewidth=2, alpha=0.2)

        return ax





if __name__=="__main__":
    # Values for ξi
    np.random.seed(69)
    N=100
    xi_values = np.random.uniform(-np.pi, np.pi, N)

    eps=np.random.standard_normal(N)
    # Values for ηi
    eta_values = np.sin(xi_values)+eps*np.exp(-xi_values**2)

    # linreg=deterministicReg(xi_values, eta_values, basis="linear")
    # linreg.solve(np.ones(2))
    # linreg.plot(color="r", marker="x")
    # plt.savefig("linreg.jpg")


    # gaussian_basis=np.arange(-5,5,1)
    # gausreg=deterministicReg(xi_values, eta_values, basis="gaussian", gaussian_basis={"means":gaussian_basis, "std": 1})
    # gausreg.solve(np.ones(len(gaussian_basis)+1))
    # gausreg.plot(color="r", marker="x")
    # plt.savefig("gausreg.jpg")

    # logistic_basis=np.arange(-5,5,1)
    # logisticreg=deterministicReg(xi_values, eta_values, basis="logistic", logistic_basis={"means":logistic_basis, "std": 1})
    # logisticreg.solve(np.ones(len(logistic_basis)+1))
    # logisticreg.plot(color="r", marker="x")
    # plt.savefig("logisticreg.jpg")

    gaussian_basis=np.arange(-5,5,1)
    bayReg=bayesianReg(xi_values, eta_values, basis="gaussian", gaussian_basis={"means":gaussian_basis, "std": 1}, alpha=1, beta=10)
    bayReg.plot(color="r", marker="x")
    plt.savefig("bayesianRegression.jpg")



