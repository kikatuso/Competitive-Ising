
import networkx as nx 
import numpy as np
import math  
from tqdm import tqdm



from . import helperfunctions as hf


class mf_ising_system():

    """
    A class to calculate mean field Ising Maximisation Influence problem for a single agent. 

    ...

    Attributes
    ----------
    graph : networkx graph
        undirected graph
    background_field : numpy.array
        Background field applied to the system
    fixed_point_iter : float, optional
        Max number of iterations used in self-consistency equations (default 5*1e4)
    fp_tol_fac : float, optional
        Tolerance factor in stoppping condition for consistency equations (default 1e-6)
    iim_iter: float, optional
        Number of gradient ascent iterations (default 1000)
    iim_tol_fac: float, optional
        Tolerance factor in stopping condition for gradient ascent algorithm (default 1e-5)
    optimiser_type: string, optional
        Type of optimiser used in gradient ascent algorithm. Can choose from 'sgd' (Stochastic Gradient Ascent),
        'sgdm' (Stochastic Gradient Ascent with Momentum), 'adagrad','adadelta' or 'adam'. Default 'adam'.

    """

    def __init__(self,graph,background_field,fixed_point_iter=int(5*1e4),fp_tol_fac=1e-5,
                 iim_iter=5000,iim_tol_fac=1e-4,optimiser_type='adam',**kwargs):
        
        self.adj_matrix = nx.to_numpy_matrix(graph,dtype=np.float64)
        self.graph_size = len(graph.nodes.keys())
        self.background_field = background_field.astype(np.float64)
        self.fixed_point_iter=fixed_point_iter
        self.iim_iter = iim_iter
        self.fp_tol_fac=fp_tol_fac
        self.iim_tol_fac=iim_tol_fac
        self.optimiser_type=optimiser_type
        for k, v in kwargs.items():
             setattr(self, k, v)

        self.init_mag=np.array([np.random.choice([-1,1]) for i in range(self.graph_size)])
      

    def lr_1(self,x):
        return np.exp(-x/(0.5*self.iim_iter))
        
    def lr_2(self,x):
        return np.exp(-x/(0.5*self.iim_iter))
   
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
        spin_field = np.dot(self.adj_matrix[i],m)
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
            if np.all((mag0+mag2-2*mag1)!=0):
                mag_d = mag0 - (mag1-mag0)**2/(mag0+mag2-2*mag1) 
            else:
                mag_d = mag1
            
            if abs(np.sum(mag0)-np.sum(mag_d))<self.fp_tol_fac: 
                break
            mag0=mag1
            mag1=mag2
            if i+1==self.fixed_point_iter:
                print('Failed to solve self-consistency equation. Consider increasing fixed_point_iter parameter')
                mag_d = mag1
 
        return mag_d

    def mag_grad(self,beta,mag):

        """
        Calculates gradient of the magnetisation with respect to change in the external control field. Nominally mean field susceptibility.

        Parameters
        ------------
        m : numpy.array
            Magnetisation array for all nodes.
        beta: float
            Interaction strength

        """
        
        if all([math.isclose(i,j,abs_tol=1e-5) for i,j in zip(mag,np.ones(len(mag)))]):
            gradient = np.zeros(len(mag))
        else:
            D=np.identity(self.graph_size)*np.array([(1-i**2) for i in mag]) # equals to 0 if all m's are 1 or close to 1
            inv = np.linalg.inv(np.identity(self.graph_size)-beta*D*self.adj_matrix)
            susc_matrix = beta*inv*D
            gradient = np.sum(susc_matrix,axis=1).A1
        return gradient
    
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

        # maximising positive of magnetisation
        mag_i_grad = self.mag_grad(beta,mag_i)
        self.gradient_history_pos.append(mag_i_grad)
        control_field = self.control_field_history_pos[it]
        gradient = mag_i_grad
        if self.optimiser_type=='sgd':
            control_field_update = control_field + self.lr_1(it)*gradient
        else:
            method = getattr(self,self.optimiser_type)
            control_field_update = control_field + method(mag_i_grad,'pos',it)
            
        control_field_new = hf.projection_simplex_sort(control_field_update.T,z=pos_budget)
        self.control_field_history_pos.append(control_field_new)
        
        return control_field_new,gradient
    
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
        # maximising negative of magnetisation
        mag_i = -1.0*mag_i
        mag_i_grad = -self.mag_grad(beta,mag_i) # minus because product rule: since H_tot = H_pos - H_neg
        self.gradient_history_neg.append(mag_i_grad)
        control_field = self.control_field_history_neg[it]
        gradient = mag_i_grad
        if self.optimiser_type=='sgd':
            control_field_update = control_field - self.lr_2(it)*gradient
        else:
            method = getattr(self,self.optimiser_type)
            control_field_update = control_field -  method(mag_i_grad,'neg',it)
            
        control_field_new = hf.projection_simplex_sort(control_field_update.T,z=neg_budget)
        self.control_field_history_neg.append(control_field_new)
        
        return control_field_new,gradient

    def sgdm(self,grad,typ,it):

        """
        Stochastic gradient descent with momentum optimiser

        Parameters
        ------------
        grad : numpy.array
            Gradient array for all nodes.
        typ : string
            'pos' if calculating for positive agent; 'neg' if calculating for negative agent.
         it : int
            Iteration index.
        """

        if typ=='pos':
            step_size=self.lr_1(it)
            name='changes_pos'
        elif typ=='neg':
            step_size=self.lr_2(it)
            name='changes_neg'

        new_change = step_size * grad + self.momentum * getattr(self,name)[-1]
        getattr(self,name).append(new_change)
        return new_change

    def adagrad(self,grad,typ,it):

        """
        Adagrad optimiser.

        Parameters
        ------------
        grad : numpy.array
            Gradient array for all nodes.
        typ : string
            'pos' if calculating for positive agent; 'neg' if calculating for negative agent.
         it : int
            Iteration index.
        """

        if typ=='pos':
            step_size=self.lr_1(it)
            name='grad_sums_pos'
        elif typ=='neg':
            step_size=self.lr_2(it)
            name='grad_sums_neg'
        grad_sums = getattr(self,name) 
        grad_sums += grad**2
        alpha = step_size / (1e-8 + np.sqrt(grad_sums))
        change = alpha*grad
        setattr(self,name,grad_sums)
        return change

    def adadelta(self,grad,typ,it,ep=1e-4):

        """
        Adadelta optimiser.

        Parameters
        ------------
        grad : numpy.array
            Gradient array for all nodes.
        typ : string
            'pos' if calculating for positive agent; 'neg' if calculating for negative agent.
         it : int
            Iteration index.
        """

        if typ=='pos':
            rho=self.rho1
            name1='grad_sums_pos'
            name2='para_pos'
        elif typ=='neg':
            rho=self.rho2
            name1='grad_sums_neg'
            name2='para_neg'
        sg = grad**2
        grad_avg = getattr(self,name1) 
        para_avg = getattr(self,name2)
        grad_avg_new= (grad_avg * rho) + (sg * (1.0-rho))
        alpha = (ep + np.sqrt(para_avg)) / (ep + np.sqrt(grad_avg_new))
        change = alpha*grad
        setattr(self,name1,grad_avg_new)
        para_avg_new = (para_avg * rho) + (change**2.0 * (1.0-rho))
        setattr(self,name2,para_avg_new)
        return change
    
    def adam(self,grad,typ,it):

        """
        Adam optimiser.

        Parameters
        ------------
        grad : numpy.array
            Gradient array for all nodes.
        typ : string
            'pos' if calculating for positive agent; 'neg' if calculating for negative agent.
         it : int
            Iteration index.
        """

        if typ=='pos':
            lr=self.lr_1(it)
            name1='ms_pos'
            name2='vs_pos'
        elif typ=='neg':
            lr=self.lr_2(it)
            name1='ms_neg'
            name2='vs_neg'
        ms_previous = getattr(self,name1)
        vs_previous = getattr(self,name2)
        if not (hasattr(self,'beta1') and hasattr(self,'beta2') and hasattr(self,'eps') ):
            self.beta1=0.8
            self.beta2=0.99
            self.eps=0.1
        ms_new = self.beta1 * ms_previous + (1.0 - self.beta1) * grad
        vs_new = self.beta2 * vs_previous + (1.0 - self.beta2) * grad**2
        mhat = ms_new / (1.0 - self.beta1**(it+1))
        vhat = vs_new / (1.0 - self.beta2**(it+1))
        setattr(self,name1,ms_new)
        setattr(self,name2,vs_new)
        change = lr* mhat/(np.sqrt(vhat) + self.eps)
        return change
    
    def second_partial_dffs(self,mag_ii,tot_field,beta,a=1e-5):

        """
        Calculates 2nd gradients of magnetisation with respect to each agents' control field.
        Calculated using central difference formula. 

        ...

        Parameters
        ----------

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
        mag_plus= -self.aitken_method(mag_ii,beta,upper_change)
        grad_plus = -self.mag_grad(beta,mag_plus)

        lower_change = tot_field-update
        mag_minus= -self.aitken_method(mag_ii,beta,lower_change)
        grad_minus = -self.mag_grad(beta,mag_minus)
        second_total_grad =  (grad_plus - grad_minus)/(2*update) # central difference formula
        curv_player_neg = - second_total_grad # minus because product rule : H_pos = H_pos - H_neg
        self.grad2neg.append(curv_player_neg)
        'gradient for positive'
        # upper_change=tot_field+update
        # mag_plus= self.aitken_method(mag_ii,beta,upper_change)
        # grad_plus = self.mag_grad(beta,mag_plus)

        # lower_change = tot_field-update
        # mag_minus= self.aitken_method(mag_ii,beta,lower_change)
        # grad_minus = self.mag_grad(beta,mag_minus)
        # second_total_grad = (grad_plus - grad_minus)/(2*update) # central difference formula
        curv_player_pos = curv_player_neg
        self.grad2pos.append(curv_player_pos)
        return np.array([curv_player_pos,curv_player_neg])

    def init_optimiser(self):
        """
        Initialise lists specific to optimiser_type.

        """
        if self.optimiser_type=='sgdm':
            self.changes_pos = [np.zeros(self.graph_size)]
            self.changes_neg = [np.zeros(self.graph_size)]
        if any(t==self.optimiser_type for t in ['adagrad','adadelta']):
            self.grad_sums_pos = np.zeros(self.graph_size)
            self.grad_sums_neg = np.zeros(self.graph_size)
            self.para_pos=np.zeros(self.graph_size)
            self.para_neg = np.zeros(self.graph_size)
        if self.optimiser_type=='adam':
            self.ms_pos = np.zeros(self.graph_size)
            self.vs_pos = np.zeros(self.graph_size)
            self.ms_neg = np.zeros(self.graph_size)
            self.vs_neg = np.zeros(self.graph_size)

    def MF_IIM(self,pos_budget,neg_budget,beta,init_alloc='random',progress=True):

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
  
        tot_field = np.array(self.background_field+control_field_pos-control_field_neg)

        self.control_field_history_pos = []
        self.control_field_history_neg = []
        self.gradient_history_pos = []
        self.gradient_history_neg = []
        self.mag_delta_history = []
        self.total_field =[]

        self.total_field.append(tot_field)
        self.control_field_history_pos.append(control_field_pos)
        self.control_field_history_neg.append(control_field_neg)
        
        mag_i = self.aitken_method(self.init_mag,beta,tot_field)
        self.grad2pos =[]
        self.grad2neg =[]   
        self.init_optimiser()
        for it in tqdm(range(self.iim_iter)) if progress else range(self.iim_iter):
            gradients =  []
            if pos_budget!=0:
                control_pos,pos_gradient = self.positive_agent(mag_i,it,pos_budget,beta)
                tot_field += control_pos
                gradients.append(pos_gradient)
            if neg_budget!=0:
                control_neg,neg_gradient = self.negative_agent(mag_i,it,neg_budget,beta)
                tot_field -= control_neg
                gradients.append(neg_gradient)
                                    
            self.total_field.append(tot_field)
            mag_ii= self.aitken_method(mag_i,beta,tot_field)
            self.mag_delta_history.append(mag_ii)
        
            if np.all([all(np.abs(gradient)<self.iim_tol_fac) for gradient in gradients]):
                second_dffs=self.second_partial_dffs(mag_ii,tot_field,beta)
                if (second_dffs[0]<0).all() and (second_dffs[1]<0).all():
                    break
 
            
            mag_i=mag_ii
            tot_field = 0 
        if it==self.iim_iter-1:
            print('Failed to converge after {} iterations'.format(self.iim_iter))
            final_mag = mag_ii
            
        elif it < self.iim_iter-1:
            print('iteration',it)
            final_mag = mag_ii
        
        self.control_field_history_pos = np.array(self.control_field_history_pos)
        self.control_field_history_neg = np.array(self.control_field_history_neg)
        

        return self.control_field_history_pos[-1],self.control_field_history_neg[-1],final_mag