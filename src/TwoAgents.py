
import networkx as nx 
import numpy as np
import math  



from . import helperfunctions as hf

class mf_ising_system:

    def __init__(self,graph,background_field,fixed_point_iter=int(5*1e4),init_mag='random',fp_tol_fac=10-6,
        iim_iter=1000,step_size=1.0,iim_tol_fac=1e-5,momentum=0.4,optimiser_type='sgdm'):
        
        self.graph = graph
        self.adj_matrix = nx.to_numpy_matrix(graph)
        self.graph_size = len(graph.nodes.keys())
        self.background_field = background_field
        self.fixed_point_iter=fixed_point_iter
        self.iim_iter = iim_iter
        self.fp_tol_fac=fp_tol_fac
        self.iim_tol_fac=iim_tol_fac
        self.optimiser_type = optimiser_type
        self.step_size=step_size
        self.momentum = momentum
        if init_mag=='aligned_neg':
            self.init_mag=(-1)*np.ones(self.graph_size)
        if init_mag=='aligned_pos':
            self.init_mag=np.ones(self.graph_size)
        if init_mag=='random':
            self.init_mag=np.array([np.random.choice([-1,1]) for i in range(self.graph_size)])
        else:
            self.init_mag = init_mag 


            
        
    def single_mag(self,i,m,beta,field):
        gamma=1.0
        spin_field = np.dot(self.adj_matrix[i],m)
        term = math.tanh(beta*(spin_field+field[i]))
        return (1.0-gamma)*m[i] + gamma*term
    
    def magnetisation(self,mag,beta,field):
        m_old = mag
        m_new = np.zeros(len(m_old))
        for i in range(self.graph_size):
            m_new[i]=self.single_mag(i,m_old,beta,field)
        return m_new

    def Steffensen_method(self,mag,beta,field,it):      
        # Numerical Analysis Richard L.Burden 9th Edition, p. 107
        
        if len(self.mag_delta_history)==0:
            mag0=mag
        else:
            mag0=self.mag_delta_history[it]
        
        for i in range(self.fixed_point_iter):     
            mag1=self.magnetisation(mag0,beta,field)
            mag2=self.magnetisation(mag1,beta,field)   

            if np.all((mag+mag2-2*mag1)!=0):
                mag_d = mag - (mag1-mag)**2/(mag+mag2-2*mag1) 
            else:
                mag_d = mag1
            
            if abs(np.mean(mag1)-np.mean(mag_d))<self.fp_tol_fac: 
                break
            if i+1==self.fixed_point_iter:
                print('Failed to solve self-consistency equation. Consider increasing fixed_point_iter parameter')
                mag_d = mag1
                
        self.mag_delta_history.append(mag_d)
        return mag_d
    
    def mag_grad(self,beta,mag):
        # Mean Field Susceptibility;
        
        if all([math.isclose(i,j,abs_tol=1e-5) for i,j in zip(mag,np.ones(len(mag)))]):
            gradient = np.zeros(len(mag))
        else:
            D=np.identity(self.graph_size)*np.array([(1-i**2) for i in mag]) # equals to 0 if all m's are 1 or close to 1
            inv = np.linalg.inv(np.identity(self.graph_size)-beta*D*self.adj_matrix)
            susc_matrix = beta*inv*D
            gradient = np.sum(susc_matrix,axis=1).A1
            gradient = (1.0 / np.linalg.norm(gradient))*gradient #Normalise

        return gradient
    
    def positive_agent(self,mag_i,it,pos_budget,beta,jt):
        mag_i_grad = self.mag_grad(beta,mag_i)
        control_field = self.control_field_history_pos[it]
        if self.optimiser_type=='sgd':
            control_field_update = control_field + self.step_size*mag_i_grad
        elif self.optimiser_type == 'sgdm':
            control_field_update = control_field + self.sgdm(mag_i_grad,self.changes)
        control_field_new = hf.projection_simplex_sort(control_field_update.T,z=pos_budget)
        self.control_field_history_pos.append(control_field_new)
        
        last_control_field_neg = self.control_field_history_neg[-1] # adding contribution from the negative agent
        tot_field = np.array(self.background_field+last_control_field_neg+control_field_new)
        mag_ii= self.Steffensen_method(mag_i,beta,tot_field,jt)
        return mag_ii
    
    
    def negative_agent(self,mag_i,it,neg_budget,beta,jt):
        mag_i_grad = self.mag_grad(beta,mag_i)
        control_field = self.control_field_history_neg[it]

        if self.optimiser_type=='sgd':
            control_field_update = control_field - self.step_size*mag_i_grad
        elif self.optimiser_type=='sgdm':
            control_field_update = (control_field - self.step_size*mag_i_grad)
            
        control_field_new = -hf.projection_simplex_sort(control_field_update.T,z=neg_budget)
        self.control_field_history_neg.append(control_field_new)
        
        last_control_field_pos = self.control_field_history_pos[-1]        
        tot_field = np.array(self.background_field+control_field_new+last_control_field_pos)
        mag_iii= self.Steffensen_method(mag_i,beta,tot_field,jt)
        return mag_iii
    
    
    def sgdm(self,grad,changes):
        if len(changes) >=2:
            new_change = self.step_size * grad + self.momentum * changes[-2] # -2 the last time agent made the change
        else:
            new_change = self.step_size * grad + self.momentum * changes[-1]
        self.changes.append(new_change)
        return new_change
    
    def MF_IIM(self,pos_budget,neg_budget,beta,order='positive'):
              
        control_field_pos =( pos_budget /self.graph_size)*np.ones(self.graph_size)
        control_field_neg = -( neg_budget /self.graph_size)*np.ones(self.graph_size)

        # initial magnetisation as influenced by initial budget spread
        # note: different from init_mag which denotes initial magnetisation *without* the external field      
        tot_field = np.array(self.background_field+control_field_pos+control_field_neg)

        self.control_field_history_pos = []
        self.control_field_history_neg = []
        self.mag_delta_history = []

        self.control_field_history_pos.append(control_field_pos)
        self.control_field_history_neg.append(control_field_neg)
        
        mag_i = self.Steffensen_method(self.init_mag,beta,tot_field,0)
        
        self.changes = [np.zeros(self.graph_size)]
        
        for it in range(self.iim_iter):
            jt = 2*it
            
            if order=='positive':
                mag_ii = self.positive_agent(mag_i,it,pos_budget,beta,jt)
            
                mag_iii = self.negative_agent(mag_ii,it,neg_budget,beta,jt+1)
            elif order=='negative':
                mag_ii = self.negative_agent(mag_i,it,neg_budget,beta,jt)
            
                mag_iii = self.positive_agent(mag_ii,it,pos_budget,beta,jt+1)
                
        
            pos_diff=np.sum(np.abs(self.control_field_history_pos[-2]-self.control_field_history_pos[-1]))
            neg_diff=np.sum(np.abs(self.control_field_history_neg[-2]-self.control_field_history_neg[-1]))
               

            if pos_diff <= self.iim_tol_fac and neg_diff <= self.iim_tol_fac:
                break
            
            mag_i=mag_iii
        if it==self.iim_iter-1:
            print('Failed to converge after {} iterations'.format(self.iim_iter))
            final_mag = None
            
        elif it < self.iim_iter-1:
            final_mag = mag_iii
        
        self.control_field_history_pos = np.array(self.control_field_history_pos)
        self.control_field_history_neg = np.array(self.control_field_history_neg)
        return self.control_field_history_pos[-1],self.control_field_history_neg[-1],final_mag