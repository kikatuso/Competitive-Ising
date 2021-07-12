
from networkx.algorithms.centrality import closeness_centrality
import random 
import numpy as np 
import networkx as nx 

class monte_carlo_simulations:
    def __init__(self,graph,background_field,positive_ratio):
        self.graph = graph
        self.graph_size = len(self.graph.nodes())
        self.adj_matrix = nx.to_numpy_matrix(graph)
        pos_no = int(positive_ratio*int(self.graph_size))
        neg_no = int(self.graph_size) - pos_no
        spins = np.concatenate([np.ones(pos_no),(-1)*np.ones(neg_no)])
        np.random.shuffle(spins)
        self.init_spins = spins
        self.background_field = background_field
        
        
    def monte_carlo_metropolis(self,control,beta,T,T_burn=1000):
        total_mag_history = []
        mag_old = self.init_spins
        for it in range(T+T_burn):
            spin_int = random.randint(0,len(self.init_spins)-1)
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
            if it > T_burn:
                total_mag_history.append(mag_old)
        return np.sum(np.array(total_mag_history),axis=1)

    def run_MC(self,control,T,MC,beta):
        time_averaged_mag = np.zeros(MC)
        for it in range(MC):
            sample = self.monte_carlo_metropolis(control,beta,T)
            time_averaged_mag[it]=np.mean(sample)

        mag_final = np.mean(time_averaged_mag)
        mag_error = np.std(time_averaged_mag)
        return mag_final,mag_error

    
    def degree(self,budget,T,MC,beta):
        sum_degree = np.sum([self.graph.degree[node] for node in self.graph.nodes])
        control_field = np.array([(self.graph.degree[node]/sum_degree)*budget for node in self.graph.nodes])

        assert np.round(np.sum(control_field),3)==budget

        tot_field = np.array(self.background_field+control_field)

        mag_mean,mag_std=self.run_MC(tot_field,T,MC,beta)

        return mag_mean,mag_std

    def centrality(self,budget,T,MC,beta):
        centrality = closeness_centrality(self.graph)
        sum_centrality=sum(centrality.values())
        control_field = np.array([(centrality[node]/sum_centrality)*budget for node in self.graph.nodes])

        assert np.round(np.sum(control_field),3)==budget

        tot_field = np.array(self.background_field+control_field)
        mag_mean,mag_std=self.run_MC(tot_field,T,MC,beta)

        return mag_mean,mag_std


    def random_set(self,budget,T,MC,beta):
        random_arr = [int(100*random.random()) for i in range(self.graph_size)]
        random_dic = {i:k for i,k in zip(self.graph.nodes,random_arr)}
        sum_random = np.sum(random_arr)
        control_field = np.array([(random_dic[node]/sum_random)*budget for node in self.graph.nodes])

        assert np.round(np.sum(control_field),3)==budget
        tot_field = np.array(self.background_field+control_field)
        mag_mean,mag_std=self.run_MC(tot_field,T,MC,beta)

        return mag_mean,mag_std
