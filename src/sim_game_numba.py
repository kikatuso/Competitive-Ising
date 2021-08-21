import networkx as nx 
import numpy as np
import math
from tqdm import tqdm
import numba
from numba.experimental import jitclass
from numba import jit

import time



steadyspec = [
    ('adj_matrix',numba.float64[:,:]),
    ('graph_size',numba.int32),
    ('background_field',numba.float64[:]),
    ('fixed_point_iter',numba.int32),
    ('fp_tol_fac',numba.float64),
]


@jitclass(steadyspec)
class steady_state(object):

    """
    A class to calculate steady state of magnetisation. 

    ...

    Attributes
    ----------
    adj_matrix : numpy.array
        Adjacency matrix of the graph
    fixed_point_iter : float, optional
        Max number of iterations used in self-consistency equations (default 1e5)
    fp_tol_fac : float, optional
        Tolerance factor in stoppping condition for consistency equations (default 1e-6)
    
    """
    def __init__(self,adj_matrix,fixed_point_iter=10000,fp_tol_fac=1e-6):

        self.adj_matrix = adj_matrix
        self.graph_size = self.adj_matrix.shape[0]
        self.fixed_point_iter=fixed_point_iter
        self.fp_tol_fac=fp_tol_fac


    def single_mag(self,i,m,beta,field):

        """
        Calculates magnetisation for a single node. Subfunction of magnetisation function. 

        Parameters
        ------------
        i : int
            Index of the node in question.
        m : numpy.array
            magnetisation array for all nodes.
        beta: float
            Interaction strength
        field: numpy.array
            Array of agent's control field for each node

        """

        gamma=1.0
        spin_field = self.adj_matrix[i].dot(m)
        term = math.tanh(beta*(spin_field+field[i]))
        return (1.0-gamma)*m[i] + gamma*term

    def magnetisation(self,mag,beta,field):
    
        """
        Calculates magnetisation for the whole system 

        Parameters
        ------------
        m : numpy.array
            magnetisation array for all nodes.
        beta: float
            Interaction strength
        field: numpy.array
            Array of agent's control field for each node
        
        """

        m_old = mag
        m_new = np.zeros(len(m_old))
        for i in range(self.graph_size):
            m_new[i]=self.single_mag(i,m_old,beta,field)
        return m_new

    def aitken_method(self,mag0,beta,field):
    
        """
        Solves self-consistency equation by following Aitken method* for accelerating convergence.
        * Numerical Analysis Richard L.Burden 9th Edition, p. 105

        Parameters
        ------------
        m0 : numpy.array
            Initial guess of magnetisation for all nodes.
        beta: float
            Interaction strength
        field: numpy.array
            Array of agent's control field for each node

        """
  
        mag1=self.magnetisation(mag0,beta,field)
        for i in range(self.fixed_point_iter):     
            mag2=self.magnetisation(mag1,beta,field)   
            if ((mag0+mag2-2*mag1)!=0).all():
                mag_d = mag0 - (mag1-mag0)**2/(mag0+mag2-2*mag1) 
            else:
                mag_d = mag1

            if abs(np.sum(mag0)-np.sum(mag_d))<self.fp_tol_fac: 
                break
            mag0=mag1
            mag1=mag2
            if i+1==self.fixed_point_iter:
                mag_d = mag1

        return mag_d

@jit(nopython=True) 
def isclose(a,b):
    return abs(a-b) <= max(1e-9 * max(abs(a), abs(b)), 1e-5)


@jit(nopython=True) 
def susc_grad(beta,mag,adj_matrix):

    """
    Calculates mean field susceptibility.

    Parameters
    ------------
    beta: float
        Interaction strength.
    mag : numpy.array
        Magnetisation array for all nodes.
    adj_matrix : numpy.array
        Adjacency matrix of the network.
    """
    D=np.identity(mag.shape[0])*np.array([(1-i**2) for i in mag]) 
    inv = np.linalg.inv(np.identity(mag.shape[0])-beta*D.dot(adj_matrix))
    susc_matrix = beta*inv.dot(D)
    gradient = np.sum(susc_matrix,axis=1).flatten()

    return gradient

def mag_grad(beta,mag,adj_matrix):

    """
    Calculates gradient of the magnetisation with respect to change in the external control field. Nominally mean field susceptibility.

    Parameters
    ------------
    beta : float
        Interaction strength.
    mag : numpy.array
        Magnetisation array for all nodes.
    adj_matrix : numpy.array
        Adjacency matrix of the network.

    """

    if np.all([isclose(i,j) for i,j in zip(mag,np.ones(mag.shape[0]))]):
        return np.zeros(len(mag))
    else:
        return susc_grad(beta,mag,adj_matrix)



@jit(nopython=True)
def projection_simplex_sort(v, z):

    """
    Bounds control field to agent's magnetic field budget. 
    ...

    Parameters
    ----------
    v : numpy.array
        Control field allocation of the agent. 
    z : float
        Magnetic field budget.

    """

    n_features = v.shape[0]
    v = np.abs(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


@jit(nopython=True) 
def lr_1(x,iim_iter=5000):
    return np.exp(-x/(0.1*iim_iter))

@jit(nopython=True) 
def lr_2(x,iim_iter=5000):
    return np.exp(-x/(0.1*iim_iter))

@jit(nopython=True) 
def adam(grad,it,typ,ms,vs,iim_iter,beta1=0.9,beta2=0.999,eps=0.1):

    """
        Adam optimiser.

        Parameters
        ------------
        grad : numpy.array
            Gradient array for all nodes.
        it : int
            Iteration index.
        typ : string
            'pos' if calculating for positive agent; 'neg' if calculating for negative agent.
        ms : numpy.array
            Momentum array with linear gradients
        vs : numpy.array
            Momentum array with squared gradients
        iim_iter : int
            Max number of iteration in optimisation algorithm.
        beta1 : float
            Default 0.9
        beta2 : float
            Default 0.999
        eps : float
            Default 0.1

    """

    if typ=='pos':
        lr=lr_1(it)
    elif typ=='neg':
        lr=lr_2(it)
    ms_new = beta1 * ms + (1.0 - beta1) * grad
    vs_new = beta2 * vs + (1.0 - beta2) * grad**2
    mhat = ms_new / (1.0 - beta1**(it+1))
    vhat = vs_new / (1.0 - beta2**(it+1))
    ms = ms_new
    vs = vs_new
    change = lr* mhat/(np.sqrt(vhat) + eps)
    return change,ms,vs

class mf_ising_system():

    """
    A class to calculate mean field Ising Maximisation Influence problem for a two agents. 

    ...

    Attributes
    ----------
    graph : networkx graph
        undirected graph
    background_field : numpy.array
        Background field applied to the system
    iim_iter: float, optional
        Number of gradient ascent iterations (default 1000)
    iim_tol_fac: float, optional
        Tolerance factor in stopping condition for gradient ascent algorithm (default 1e-3)
    """

    def __init__(self,graph,background_field,iim_iter=10000,iim_tol_fac=1e-3):

        self.graph=graph
        self.adj_matrix = nx.to_numpy_matrix(graph).astype(np.float64)
        self.graph_size = self.adj_matrix.shape[0]
        self.background_field = background_field.astype(np.float64)
        self.iim_iter=iim_iter
        self.iim_tol_fac = iim_tol_fac
        

    def positive_agent(self,mag_i,it,pos_budget,beta):

        """
        Single move by the positive agent. 

        ...

        Parameters
        ----------
        mag_i : numpy.array
            Magnetisation array for all nodes.
        it : int
            Iteration of the algorithm.
        pos_budget : float
            Magnetic field budget for the positive agent.
        beta: float
            Interaction strength
        """

        mag_i_grad = mag_grad(beta,mag_i,self.adj_matrix)
        control_field = self.control_field_history_pos[-1]
        change,ms,vs = adam(mag_i_grad,it,'pos',self.ms_pos,self.vs_pos,self.iim_iter)
        self.ms_pos=ms
        self.vs_pos=vs
        control_field_update = control_field  + change
        control_field_new = projection_simplex_sort(control_field_update.T,pos_budget)
        self.control_field_history_pos.append(control_field_new)
        return control_field_new,mag_i_grad
    
    def negative_agent(self,mag_i,it,neg_budget,beta):
    
        """
        Single move by the negative agent. 

        ...

        Parameters
        ----------
        mag_i : numpy.array
            Magnetisation array for all nodes.
        it : int
            Iteration of the algorithm.
        neg_budget : float
            Magnetic field budget for the negative agent.
        beta: float
            Interaction strength
   
        """

        mag_i = -1.0*mag_i
        mag_i_grad = -mag_grad(beta,mag_i,self.adj_matrix)
        control_field = self.control_field_history_neg[-1]
        change,ms,vs=adam(mag_i_grad,it,'neg',self.ms_neg,self.vs_neg,self.iim_iter)
        self.ms_neg=ms
        self.vs_neg=vs
        control_field_update = control_field + change
        control_field_new = projection_simplex_sort(control_field_update.T,neg_budget)
        self.control_field_history_neg.append(control_field_new)
        
        return control_field_new,mag_i_grad
        
    def second_partial_dffs(self,state,mag_ii,tot_field,beta,a=1e-5):

        """
        Calculates 2nd gradients of magnetisation with respect to each agents' control field.
        Calculated using central difference formula. 

        ...

        Parameters
        ----------

        state : class instance
            steady_state class instance
        mag_ii : numpy.array
            Magnetisation array for all nodes.
        tot_field : numpy.array
            Total net magnetic field experienced by each node. 
        beta: float
            Interaction strength
        a : float, optional
            Used in central difference formula. Specifies magnitude of change of control field. 
   
        """

        update = a*np.ones(self.graph_size)
        upper_change=tot_field+update
        mag_plus= -state.aitken_method(mag_ii,beta,upper_change)
        grad_plus = -mag_grad(beta,mag_plus,self.adj_matrix)

        lower_change = tot_field-update
        mag_minus= -state.aitken_method(mag_ii,beta,lower_change)
        grad_minus = -mag_grad(beta,mag_minus,self.adj_matrix)
        second_total_grad =  (grad_plus - grad_minus)/(2*update) # central difference formula
        curv_player_neg = - second_total_grad # minus because product rule : H_pos = H_pos - H_neg
        curv_player_pos = curv_player_neg
        return np.array([curv_player_pos,curv_player_neg])
    
    def init_lists(self):

        """
        Initialises lists for storing variables.

        """
        self.control_field_history_pos =[]
        self.control_field_history_neg = []
        self.mag_history = []
        self.pos_gradient_history=np.zeros((self.iim_iter,self.graph_size))
        self.neg_gradient_history=np.zeros((self.iim_iter,self.graph_size))
        
        self.ms_pos = np.zeros(self.graph_size,dtype=np.float64)
        self.vs_pos = np.zeros(self.graph_size,dtype=np.float64)
        self.ms_neg = np.zeros(self.graph_size,dtype=np.float64)
        self.vs_neg = np.zeros(self.graph_size,dtype=np.float64)
        

    def MF_IIM(self,pos_budget,neg_budget,beta,init_alloc='random',progress=True,max_time=3.0):
        """
        Calculates competitive MF-IIM by following stochastic gradient ascent optimisation with
        Adam optimiser.

        Parameters
        ------------
        pos_budget : float
            Maximum magnetic field budget to be spent by the positive agent.
        neg_budget : float
            Maximum magnetic field budget to be spent by the negative agent.
        beta : float
            Interaction strength
        init_alloc : string or numpy.array, optional
            Either 'uniform' which corresponds to uniform spread of financial budget equaly among nodes.
            'random' corresponds to random initialisations. Alternatively, provide custom numpy.array 
            allocation of your own choice. Default 'random'.
        progress : boolean
            If True shows progress bar; False otherwise. 

        Outputs
        -----------
        control_field_pos : numpy.array
            Positive agent's control field allocation that results in the equilibrium.
        control_field_neg : numpy.array
            Negative agent's control field allocation that results in the equilibrium.
        final_mag : numpy.array
            Final magnetisation of the system. 

        """
 
        if isinstance(init_alloc,(np.ndarray, np.generic)):
            control_field_pos = init_alloc[0,:]
            control_field_neg = init_alloc[1,:] 
        elif isinstance(init_alloc,str):  
            if  init_alloc=='aligned':
                control_field_pos =( pos_budget /self.graph_size)*np.ones(self.graph_size)
                control_field_neg = ( neg_budget /self.graph_size)*np.ones(self.graph_size)
            elif init_alloc=='random':
                control_field_pos  = np.random.dirichlet(np.ones(self.graph_size))*pos_budget
                control_field_neg  = np.random.dirichlet(np.ones(self.graph_size))*neg_budget
  
        init_mag = np.array([np.random.choice([-1,1]) for i in range(self.graph_size)]).astype(np.float64)

        self.init_lists()

        self.control_field_history_pos.append(control_field_pos)
        self.control_field_history_neg.append(control_field_neg)
        
        
        state = steady_state(self.adj_matrix)
        self.state=state

        tot_field = control_field_pos-control_field_neg
        
        mag_i = state.aitken_method(init_mag,beta,tot_field)
        self.mag_history.append(mag_i)
        start_time = time.time()
        self.converged=False
        for it in tqdm(range(self.iim_iter)) if progress else range(self.iim_iter):
            
            gradients=[]
            if pos_budget!=0:
                control_pos,pos_gradient = self.positive_agent(mag_i,it,pos_budget,beta)
                tot_field +=control_pos
                self.pos_gradient_history[it]=pos_gradient
                gradients.append(pos_gradient)

            if neg_budget!=0:
                control_neg,neg_gradient = self.negative_agent(mag_i,it,neg_budget,beta)
                tot_field-=control_neg
                self.neg_gradient_history[it]=neg_gradient
                gradients.append(neg_gradient)

            mag_ii= state.aitken_method(mag_i,beta,tot_field)
            
            self.mag_history.append(mag_ii)
            time_now = time.time()
            elapsed_time = (time_now - start_time)/60.0
            if (all(np.abs(self.control_field_history_pos[-1]-self.control_field_history_pos[-2])<self.iim_tol_fac) and 
                all(np.abs(self.control_field_history_neg[-1]-self.control_field_history_neg[-2])<self.iim_tol_fac)):
                self.converged=True
                break
#             if np.all([(abs(gradient)<self.iim_tol_fac).all() for gradient in gradients]):
#                 second_dffs=self.second_partial_dffs(state,mag_ii,tot_field,beta)
#                 if (second_dffs[0]<0).all() and (second_dffs[1]<0).all():
#                     break
            if elapsed_time>max_time:
                print('Timeout')
                break
            mag_i=mag_ii
            tot_field=0.0
        if it==self.iim_iter-1:
            print('Failed to converge after {} iterations'.format(self.iim_iter))
            final_mag = mag_ii
            
        elif it < self.iim_iter-1:
            final_mag = mag_ii
        
        self.control_field_history_pos = np.array(self.control_field_history_pos)
        self.control_field_history_neg = np.array(self.control_field_history_neg)
        self.mag_history = np.array(self.mag_history)

        return self.control_field_history_pos[-1],self.control_field_history_neg[-1],final_mag