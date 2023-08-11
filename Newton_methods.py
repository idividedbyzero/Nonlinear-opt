import numpy as np
#TODO make an unconstrained opt class
class newton_solver():
    def __init__(self, f,df,ddf,beta=0.5, gamma=0.1, alpha=[0.1, 1], p=2):
        self.f=f
        self.df=df
        self.ddf=ddf
        self.beta=beta
        self.gamma=gamma
        self.alpha=alpha
        self.p=p

    def solve(x):
        x=np.array(x)
