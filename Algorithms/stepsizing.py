
import numpy as np


class Armijo():
    """Armijo stepsizing class
    """

    def __init__(self, beta: float=1/2, gamma: float =10**(-2)):
        """Creates an instance of the armijo stepsizing class and checks for valid parameters

        Args:
            beta (float, optional): Value that is used for each iteration. Defaults to 1/2.
            gamma (float, optional): Values that the rhs is multiplied with. Defaults to 10**(-2).
        """
        self.beta=beta
        self.gamma=gamma
        self._check_params()
        

    def _check_params(self):
        """Check the parameters for correctness
        """
        self._check_bound(self.beta, 0, 1, "Beta")
        self._check_bound(self.gamma, 0, 1, "Gamma")

    def _check_bound(self, param, lower: float, upper:float, name: str):
        assert param>lower & param<upper, f"{name} needs to be between {str(lower)} and {str(upper)}"


    def step(self,f: function, df: function, x:np.ndarray,s: np.ndarray, maxiter: 1000)-> float:
        """Finds the next armijo step from a given point in the given direction

        Args:
            f (function): Function
            df (function): Derivative of function f
            x (np.ndarray): Point from where we need the next step size
            s (np.ndarray): Direction of the next step

        Returns:
            float: Returns the step size from x in direction s
        """
        sigma=1
        while True:         #This will stop by Lemma 7.5 in [1]
            if self._armijo_check( f, df, x, s, sigma):
                return sigma
            else :
                sigma=sigma*self.beta
            maxiter=maxiter-1
            assert maxiter>0, "Maxiter has been reached while finding the armijo step size"    

    def _armijo_check(self, f, df, x, s, sigma):
        """Checks wether the armijo condition is fulfilled with the given parameters
        
        Params: 
            as in "step"
        """
        gamma=self.gamma
        return f(x+sigma*s)-f(x)<= sigma*gamma* np.inner(df(x), s)
    

class Powellwolfe(Armijo):

    def __init__(self, beta: float=1/2, gamma: float =10**(-2)):
        """Creates an instance of the Powellwolfe stepsizing class and checks for valid parameters

        Args:
            beta (float, optional): Value that is used for each iteration. Defaults to 1/2.
            gamma (float, optional): Values that the rhs is multiplied with. Defaults to 10**(-2).
        """
        self.beta=beta
        self.gamma=gamma
        self.check_params()

    def check_params(self):
        super().check_params()


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