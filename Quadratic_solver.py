import numpy as np

class conjugate_gradient():
    """
        Goal is to solve min 1/2 x^tAx -b^tx+c, without solving the expansive equation Ax=b. Where A is symmteric and positive definite
        Requirement:
            A has full rank

        This can be also used to solve linear equations Ax=b
        #TODO instead of solving A^tAx=A^tb, also implement a more efficient version if A is positiv definit
    """
    def __init__(self, A, b):
        self.A=A
        self.b=b

    def solve(self, x0):
        x = np.array(x0)
        A=self.A
        b=self.b
        system_size=np.array(A).shape[1]
        r = np.matmul(A.T, b - np.matmul(A,x))
        d = r
        delta_new = np.dot(r.T, r)

        for i in range(system_size):

            # compute alpha_k
            # <Ad,Ad>
            q = np.matmul(A,d)
            alpha = delta_new / np.dot(q.T,q)

            # update x
            x = x + alpha * d

            # updatethe residual
            # r = r - alpha * A.T * A * d
            r = r - alpha * np.matmul(A.T,q)

            # update beta
            delta_old = delta_new
            delta_new = np.dot(r.T,r)
            beta = delta_new / delta_old

            # update dk
            d = r + beta * d
        print("CG converged, Error {:1.8}".format(np.linalg.norm(np.matmul(A,x)-b)))
        return x
