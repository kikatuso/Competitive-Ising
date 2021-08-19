


import numpy as np 
import torch
import sys


def projection_simplex_sort(v, z=1):

    """
    Bounds control field to agent's magnetic field budget. 
    ...

    Parameters
    ----------
    v : numpy.array
        Control field allocation of the agent. 
    z : float
        Magnetic field budget (default 1.0) 

    """

    n_features = v.shape[0]
    v = np.abs(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    try:
        rho = ind[cond][-1]
    except IndexError:
        print(v)
        sys.exit()
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


def crit_b(J):
    """
    Calculates critical interaction strenght for the graph

    Parameters
    ----------
    J : numpy.array
        Adjacency matrix for the graph. 

    """
    eigenvalues,_ = np.linalg.eig(J)
    return 1./np.max(np.abs(eigenvalues))

def average_degree(system,control,budget):
    """
    Calculates average degree of nodes targeted by the agent.

    Parameters
    ----------
    system : class instance
        Class instance of MF-IIM class 
    control : numpy.array
        Control field allocation of the agent.
    budget : float
        Magnetic field budget. 

    """
    return np.sum([system.graph.degree[i]*con for i,con in enumerate(control)])/budget





