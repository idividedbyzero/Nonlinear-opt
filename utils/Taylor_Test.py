import logging

import numpy as np


def taylor_test(f, df, ddf, x, h, order =1):
    x=np.array(x)
    h=np.array(h)
    fx=f(x)
    dfx=df(x)
    ddfx=ddf(x)

	#dJ = obj.gradient(m)


	#dJdm = assemble(dJ*h*dx)

	#dm = m.copy(deepcopy=True)

    print("Running Taylor test")
    residuals = []              #create empty list of residuals
    epsilons = [0.01 / 2 ** i for i in range(4)]            #generate different epsilon values
    for eps in epsilons:
        dh=x+eps*h
		#dm.assign(m+Constant(eps)*h)
        fdh=f(dh)
		#Jp = obj.evaluate_J(dm) #perturb with eps in direction h   
        if order==0:
            res = abs(fdh - fx )   #check if J is ok 
        elif order==1:
            res = abs(fdh- fx -eps*np.inner(dfx, h))   #1st order taylor test 
        else:
            res = abs(fdh - fx - eps*np.inner(dfx, h) - 0.5 * eps ** 2 * np.inner(h,np.dot(ddfx , h) ))   #2nd-taylor expansion 
        residuals.append(res)
	
    if min(residuals) < 1E-15:
        logging.warning("The taylor remainder is close to machine precision.")
    print("Computed residuals: {}".format(residuals))
    return np.median(convergence_rates(residuals, epsilons))
	
	
def convergence_rates(E_values, eps_values, show=True):
	from numpy import log
	r = []
	for i in range(1, len(eps_values)):
		r.append(log(E_values[i] / E_values[i - 1])
		/ log(eps_values[i] / eps_values[i - 1]))
	if show:
		print("Computed convergence rates: {}".format(r))
	return r

