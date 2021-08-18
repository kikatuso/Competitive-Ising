

import networkx as nx 
import numpy as np
import math    
from . import helperfunctions as hf


class mf_ising_system:

    def __init__(self,graph,background_field,fixed_point_iter=int(5*1e4),init_mag='aligned',fp_tol_fac=1e-6,
        iim_iter=1000,step_size=1.0,iim_tol_fac=1e-5,optimiser_type='sgd',momentum=0.4):
        
        self.graph = graph
        self.adj_matrix = nx.to_numpy_matrix(graph)
        self.graph_size = len(graph.nodes.keys())
        self.background_field = background_field
        self.fixed_point_iter=fixed_point_iter
        self.iim_iter = iim_iter
        self.fp_tol_fac=fp_tol_fac
        self.iim_tol_fac=iim_tol_fac
        self.optimiser_type = optimiser_type
        self.momentum = momentum
        self.step_size=step_size
        if init_mag=='aligned':
            self.init_mag=np.ones(self.graph_size)
        elif init_mag=='random':
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

    def aitken_method(self,mag0,beta,field):      
        # Numerical Analysis Richard L.Burden 9th Edition, p. 105
        
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
            self.gradient_history.append(gradient)
        return gradient
    
    def sgdm(self,grad,control_field,changes,it):
        new_change = self.step_size * grad + self.momentum * changes[it]
        control_field_update = control_field + new_change
        changes.append(new_change)
        return control_field_update,changes
    

    def MF_IIM(self,field_budget,beta,init_control_field='uniform'):
              
        if init_control_field=='uniform':
            control_field = (field_budget/self.graph_size)*np.ones(self.graph_size)
        else:
            control_field = init_control_field

        # initial magnetisation as influenced by initial budget spread
        # note: different from init_mag which denotes initial magnetisation *without* the external field      
        tot_field = np.array(self.background_field+control_field)

        self.control_field_history=[]
        self.control_field_history.append(control_field)
        self.mag_delta_history =[]
        self.gradient_history=[]
        mag_i= self.aitken_method(self.init_mag,beta,tot_field)
        

        changes = [np.zeros(self.graph_size)]
        for it in range(self.iim_iter):
            if field_budget!=0:
                mag_i_grad = self.mag_grad(beta,mag_i)
                control_field = self.control_field_history[it]

                if self.optimiser_type=='sgd':
                    control_field_update = (control_field + self.step_size*mag_i_grad)
                elif self.optimiser_type =='sgdm':
                    control_field_update,changes = self.sgdm(mag_i_grad,control_field,changes,it)

                control_field_new = hf.projection_simplex_sort(control_field_update.T,z=field_budget)

            elif field_budget==0:
                control_field_new = np.zeros(self.graph_size)
    
            tot_field = np.array(self.background_field+control_field_new)
            mag_ii= self.aitken_method(mag_i,beta,tot_field)

            if np.abs(np.mean(mag_ii)-np.mean(mag_i)) <= self.iim_tol_fac:
                final_mag=mag_ii
                break
            self.control_field_history.append(control_field_new)
            mag_i=mag_ii
        if it==self.iim_iter-1:
            print('Failed to converge after {} iterations'.format(self.iim_iter))
            final_mag = mag_ii
            
        return control_field,final_mag
            

from itertools import permutations
from functools import reduce
import numdifftools as nd


class TrueSolution:

    def __init__(self,graph,beta,init_mag='random',mu=2.0):
        self.adj_matrix = nx.to_numpy_matrix(graph)
        self.graph_size = len(graph.nodes.keys())
        possible_configs = np.zeros([2**self.graph_size,self.graph_size])
        all_pos,all_neg = np.ones(self.graph_size),(-1)*np.ones(self.graph_size)
        for i,p in enumerate(self.unique_permutations(np.concatenate((all_pos,all_neg)),self.graph_size)):
            possible_configs[i]=p
        self.mu = mu
        self.possible_configs = possible_configs # possible spin configurations
        self.beta=beta


    def unique_permutations(self,iterable, r=None):
        previous = tuple()
        for p in permutations(sorted(iterable), r):
            if p > previous:
                previous = p
                yield p    

    def hamiltonian(self,m,h):
        x=np.sum([float(m[i]*self.adj_matrix[i,:]@m) for i in range(self.graph_size)])/2.0 + h@m
        return -x  


    def partition_function(self,h,beta):
        term = np.array([np.exp(-beta*self.hamiltonian(self.possible_configs[i],h)) for i in range(self.possible_configs.shape[0])])
        
        return np.sum(term)

    def boltzmann(self,h,beta,j):
        " j - configuration index"
        mag_ins = self.possible_configs[j]
        term = 1/self.partition_function(h,beta) * np.exp(-beta*self.hamiltonian(mag_ins,h))
        return term


    def magnetisation(self,h):
        beta = self.beta
        m = np.zeros(self.graph_size)
        for ix in range(m.shape[0]):
            m_i = np.array([self.possible_configs[j,ix]*self.boltzmann(h,beta,j) for j in range(self.possible_configs.shape[0])])
            m[ix]=np.sum(m_i)
        return m

    
    def projection_simplex_sort(v, z=1.0):
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
    
    def max_mag(self,beta,budget,iters=100,lr=0.5): 
        per_spin = budget/self.graph_size
        h = per_spin*np.ones(self.graph_size)
        for i in range(iters):
            print(i)
            grad=nd.Gradient(self.magnetisation)([h])
            h+=lr*np.sum(grad,axis=1)
            h = self.projection_simplex_sort(h,budget)
        return h
    
    
