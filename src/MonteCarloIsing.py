
from networkx.algorithms.centrality import closeness_centrality
import random 
import numpy as np 
import networkx as nx 

class monte_carlo_simulations:

    """
    A class to calculate mean field Ising Maximisation Influence problem for a single agent
    using Markov Chain Monte Carlo sampling. 

    ...
    Attributes
    ----------
    graph : networkx graph
        undirected graph
    background_field : numpy.array
        Background field applied to the system
    T_burn : int
        Transient period - number of iterations to be burnt. Default 500.
    """


    def __init__(self,graph,background_field,T_burn=500):
        self.graph = graph
        self.graph_size = len(self.graph.nodes())
        self.adj_matrix = nx.to_numpy_matrix(graph)
        self.background_field = background_field
        self.T_burn = T_burn
        
    def initialise_spins(self): 
        """
        Subfunction that initialises spins

        """
        spins =np.ones(self.graph_size)
        return spins

    def monte_carlo_metropolis(self,control,beta,T):

        """
        Monte carlo Metropolis algorithm.

        ...
        Parameters
        ----------
        control : numpy.array
            Control field allocation of the agent.
        beta: float
            Interaction strength.
        T : int
            Number of iterations (on top of transient period).

        """


        total_mag_history = []
        mag_old = self.initialise_spins()
        for it in range(T+self.T_burn):
            spin_int = random.randint(0,self.graph_size-1)
            surr_spin = float(self.adj_matrix[spin_int,:]@mag_old)
            delta_e = 2*mag_old[spin_int]*(surr_spin+control[spin_int])
            if delta_e > 0:
                prob=np.exp( -1.0*beta*delta_e)
            else:
                prob = 1.0
            random_prob = random.uniform(0, 1)
            if random_prob <=prob:
                mag_rev = (-1)*mag_old[spin_int]
                mag_old[spin_int] = mag_rev
            if it > self.T_burn:
                total_mag_history.append(np.sum(mag_old))
        return total_mag_history

    def run_MC(self,control,T,MC,beta):

        """
        Runs independent Markov Chain for Monte Carlo samplings.

        ...
        Parameters
        ----------
        control : numpy.array
            Control field allocation of the agent.
        T : int
            Number of iterations for Monte Carlo sampling (on top of transient period).
        MC : int
            Number of Markov Chains.
        beta: float
            Interaction strength.

        """

        time_averaged_mag = np.zeros(MC)
        for it in range(MC):
            sample = self.monte_carlo_metropolis(control,beta,T)
            time_averaged_mag[it]=np.mean(sample)

        mag_final = np.mean(time_averaged_mag)
        mag_error = np.std(time_averaged_mag)
        return mag_final,mag_error

    
    def degree(self,budget,T,MC,beta):

        """
        Calculates magnetisation based on distribution of
        magnetic field budget proportional to nodes degree.

        ...
        Parameters
        ----------
        budget : float
            Magnetic field budget for the agent.
        T : int
            Number of iterations for Monte Carlo sampling (on top of transient period).
        MC : int
            Number of Markov Chains.
        beta: float
            Interaction strength.

        """

        sum_degree = np.sum([self.graph.degree[node] for node in self.graph.nodes])
        control_field = np.array([(self.graph.degree[node]/sum_degree)*budget for node in self.graph.nodes])

        assert np.round(np.sum(control_field),3)==budget

        tot_field = np.array(self.background_field+control_field)

        mag_mean,mag_std=self.run_MC(tot_field,T,MC,beta)

        return mag_mean,mag_std

    def centrality(self,budget,T,MC,beta):

        """
        Calculates magnetisation based on distribution of 
        magnetic field budget proportional to nodes centrality measure.

        ...
        Parameters
        ----------
        budget : float
            Magnetic field budget for the agent.
        T : int
            Number of iterations for Monte Carlo sampling (on top of transient period).
        MC : int
            Number of Markov Chains.
        beta: float
            Interaction strength.

        """

        centrality = closeness_centrality(self.graph)
        sum_centrality=sum(centrality.values())
        control_field = np.array([(centrality[node]/sum_centrality)*budget for node in self.graph.nodes])

        assert np.round(np.sum(control_field),3)==budget

        tot_field = np.array(self.background_field+control_field)
        mag_mean,mag_std=self.run_MC(tot_field,T,MC,beta)

        return mag_mean,mag_std


    def random_set(self,budget,T,MC,beta):

        """
        Calculates magnetisation based on distribution of magnetic field budget spread 
        across nodes randomly.

        ...
        Parameters
        ----------
        budget : float
            Magnetic field budget for the agent.
        T : int
            Number of iterations for Monte Carlo sampling (on top of transient period).
        MC : int
            Number of Markov Chains.
        beta: float
            Interaction strength.

        """

        random_arr = [int(100*random.random()) for i in range(self.graph_size)]
        random_dic = {i:k for i,k in zip(self.graph.nodes,random_arr)}
        sum_random = np.sum(random_arr)
        control_field = np.array([(random_dic[node]/sum_random)*budget for node in self.graph.nodes])

        assert np.round(np.sum(control_field),3)==budget
        tot_field = np.array(self.background_field+control_field)
        mag_mean,mag_std=self.run_MC(tot_field,T,MC,beta)

        return mag_mean,mag_std
