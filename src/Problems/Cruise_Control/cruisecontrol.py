import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import logging
from src.Algorithms.stepsizing import Armijo
from tqdm import tqdm

def solveforward(u: np.ndarray, data):
    u = u.reshape((data['m'], 2), order="F")  # Reshape to (g, b) row vectors format
    y = np.zeros_like(u)

    y[0]=data["y0"]

    for k in range(y.shape[0] - 1):
        y[k + 1] = y[k] + data['h'] * np.array([
            y[k,1],
            data['s'](y[k, 0]) + u[k, 0] - u[k, 1] - data['gamma'] * y[k, 1] **2
        ])

    return y

def solveadjoint(u: np.ndarray, y, data):
    u = u.reshape((data['m'], 2), order="F")  # Reshape to (g, b) row vectors format
    z = np.zeros_like(u)

    z[-1] = np.array([0, -(data['h'] / 2) * (y[-1,1] - data['vdes'])])

    for k in range(z.shape[0] - 1, 0, -1):
        fac = 1.0 - (k == 1) * 0.5
        z[k-1] = z[k] - data['h'] * np.array([
            -data['ds'](y[k-1, 0]) * z[k, 1],
            -z[k, 0] + fac * (y[k-1, 1] - data['vdes']) + 2 * data['gamma'] * np.abs(y[k-1, 1]) * z[k,1]
        ])

    return z

def plotresults(u: np.ndarray, data):
    y = solveforward(u, data)
    u = u.reshape((data['m'], 2), order="F")  # Reshape to (g, b) row vectors format
    plt.figure()
    handle = plt.plot(data['ts'], y[:, 1], data['ts'], u[:, 0], data['ts'], -u[:, 1], data['ts'], data['s'](y[:, 0]))
    plt.legend(['velocity', 'gas', '-break', 'acceleration by profile'])
    plt.xlabel('time (s)')
    plt.savefig("test")
    return handle

def redobjective(u, data):
    y = solveforward(u, data)
    fun = objective(y, u, data)
    
    if len(data) >= 2:
        z = solveadjoint(u, y, data)
        grad = gradient(z, u, data)
        return fun, grad
    
    return fun

def objective(y, u:np.ndarray, data):
    u = u.reshape((data['m'], 2), order="F")  # Reshape to (g, b) row vectors format
    fun = (data['h'] / 2) * np.sum(
        0.5 * (y[:-1, 1] - data['vdes'])**2 +
        0.5 * (y[1:, 1] - data['vdes'])**2 +
        data['phi'](u[:-1, 0]) +
        data['phi'](u[1:, 0])
    )
    return fun

def gradient(z, u:np.ndarray, data):
    u = u.reshape((data['m'], 2), order="F")  # Reshape to (g, b) row vectors format
    grad = data['h'] * np.concatenate([
        [data['dphi'](u[0, 0]) / 2 - z[1, 1]],
        data['dphi'](u[1:-1, 0]) - z[2:, 1],
        [data['dphi'](u[-1, 0]) / 2],
        z[1:, 1],
        [0]
    ])
    return grad


# projected gradient method:
def projgradient(f, P, x0, tau:float=0.3, beta:float=1/2, gamma:float =10**-2, eps=10**-8, maxiter:int=10000):
    x= x0
    for i in range(maxiter):
        d_tau=lambda x: (x-P(x-tau* f(x)[1]))
        d=-d_tau(x)
        if np.linalg.norm(d)/tau<eps:
            logging.info("Iteration stoped at iteration {:1} at point {:2}, with norm {:9}".format(i, np.linalg.norm(d)))
            return x
        armijo=Armijo(beta, gamma)
        sigma=armijo.step(lambda x: f(x)[0], lambda x: f(x)[1], x, d)
        x=x+sigma*d
        if i%100==0:
            print("Iteration: {:1} with norm {:9}".format(i, np.linalg.norm(d)))
            
    logging.info("Maximum iterations in proj gradient method reached!")
    return x
    



if __name__=="__main__":
    import yaml

    # Replace 'your_file.yaml' with the path to your YAML file
    file_path = 'src/Problems/Cruise_Control/parameters.yaml'

    with open(file_path, 'r') as file:
        odedata = yaml.safe_load(file)

    odedata["y0"]=np.array([odedata["x0"], odedata["v0"]])
    #gmin, gmax = 0, 3
    #bmin, bmax = 0, 5
    #nsteps = 150
    odedata['h'] = odedata['T'] / odedata["nsteps"]
    odedata['m'] = odedata["nsteps"] + 1
    odedata['ts'] = np.linspace(0, odedata['T'], odedata['m'])

    sinfac = 0.003 #peridiccity
    sins = -1      #amplitude
    cosfac = 0.001  #peridicity
    coss = -2       #amplitude
    # to location dependent accelaration  e.g. rolling resistance we define a function s
    odedata['s'] = lambda x: sins * np.sin(sinfac * x) + coss * np.cos(cosfac * x) + 1
    odedata['ds'] = lambda x: sins * sinfac * np.cos(sinfac * x) - coss * cosfac * np.sin(cosfac * x)
    
    phiexp = 1.5
    phifac = 1.0
    #regularizer in the cost
    odedata['phi'] = lambda u: phifac * u**phiexp
    odedata['dphi'] = lambda u: phifac * phiexp * u**(phiexp - 1)
    #air resistence
    odedata['gamma'] = 0.002

    #initial acceliration
    ginit = 2 * np.ones(odedata['m'])
    #initial break
    binit = np.zeros(odedata['m'])
    #initial control
    uinit = np.hstack([ginit, binit])

    # Plot initial results
    #plotresults(uinit, odedata)

    # Formulate bounds on u
    lower = np.concatenate([odedata["gmin"] * np.ones(odedata['m']), odedata["bmin"] * np.ones(odedata['m'])])
    upper = np.concatenate([odedata["gmax"] * np.ones(odedata['m']), odedata["bmax"] * np.ones(odedata['m'])])

    #Projection to the feasible set
    P=lambda x: np.minimum(upper, np.maximum(lower, x))

    # # Define the reduced objective function
    def fred(u):
        return redobjective(u, odedata)

    # # Solve the optimization problem
    #result = minimize(lambda x: fred(x)[0], uinit, bounds=list(zip(lower, upper)))

    
    # # Get the optimal solution
    #u_optimal = result.x

    #Solve with projected gradient descent
    u_optimal=projgradient(fred, P, uinit)


    # Plot the optimal results
    plotresults(u_optimal, odedata)

    plt.savefig("Test_Cruise_control")
