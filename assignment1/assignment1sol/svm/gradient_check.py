import numpy as np
from random import randrange

def grad_check_sparse(f, Theta, dTheta, num_checks = 10, epsilon = 1e-5): 
    '''
    sample a few random elements and only return numerical
    in this dimensions.
    '''
    for i in xrange(num_checks):
        idx = tuple([randrange(m) for m in Theta.shape])

        Theta_j = Theta[idx]    # remember old val.
        Theta[idx] = Theta_j + epsilon     # increment by epsilon
        fplus = f(Theta)              # evaluate f(Theta + epsilon)
        Theta[idx] = Theta_j - epsilon      # decrease by epsilon
        fminus = f(Theta)           # evaluate f(Theta - epsilon)
        Theta[idx] = Theta_j       # reset

        grad_num = (fplus - fminus) / (2 * epsilon)
        grad_anal = dTheta[idx]
        rel_err = abs(grad_num - grad_anal) / (abs(grad_anal) + abs(grad_num))
        print 'rel. err.', rel_err, 'numerical:', grad_num, 'analytical:', grad_anal