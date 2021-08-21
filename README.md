# Optimal Strategies in Two player Ising Influence Maximisation Problem.
**Aims of the Project**: 
The project aims to analyse quantitatively behaviours and strategies of two adversarial agents with an objective to maximise their influence on 
the same subset of a social network. This scenario is widely known in real world and is applicable to e.g. campaigns of two political counter-candidates 
or smartphone companies aiming to sell their product. This framework will be implemented with the use of the **Ising Model** – 
a physical model that quantitatively can express e.g. relationships between social network members. 

The project studies the influence allocation of each of two external agents given the allocation of the counter-competitor and agents’ 
budgets for advertising campaigns. The model used in this study will be based on the framework presented by Lynn and Lee in [1], which considers **the Ising Model 
under the Mean Field approximation**. The first part of the project, focused solely on a single-agent magnetisation maximisation,
involves reproducing the results presented in the aforementioned paper. 

Since the paper by Lynn and Lee considers a single external agent, 
following reproduction stage the project extends the original framework to include two adversarial external agents. 
We structured the allocation problem between two adversarial agents in a game theoretic setting. 


### In particular we have written two competitive optimisation algorithms: 

1. One that structures the problem as a **simultaneous game** and tries to converge to an equilibrium
point that will correspond to a local **Nash** Equilibrium.

2. And a second one that structures the problem as a **sequential game** and tries to converge to an equilibrium
point that will correspond to a local **Stackelberg** Equilibrium.

[1] Lynn, Christopher W. and Daniel D. Lee. “Maximizing Influence in an Ising Network: A Mean-Field Optimal Solution.” NIPS (2016).


## Initilisation of the code
To install libraries, run in terminal 
```
python setup.py install
```
This will install 'src' library with all the necessary functions to run simulations.
Upon installation, function can be imported in the following way:
```python
from src import seq_game as seq
```

### Reproduction of the paper by Lynn and Lee
Function *MonteCarloIsing.py* to run Markov Chain Monte Carlo simulations on calculating network's magnetisation when control field is distributed across nodes
using a range of different heuristic methods.
1. Distribution of budget proportionally to nodes degree.
2. Distribution of budget proportionally to nodes centrality measure.
3. Distribution of budget across nodes randomly.

Run function *single_agent.py* to run Mean-Field Ising Magnetisation Maximisation algorithm or to
compute exact magnetisation solutions in simple networks (like hub-and-spoke network considered in this study). 




<p align="center">
<img src="https://github.com/kikatuso/Master-Thesis/blob/main/img/MFIIM_vs_True_Optimal.png" alt="hub_and_spoke" width="900"/>
</p>

**Single  agent  MF-IIM  for  a  hub-and-spoke  network  with  4  peripheral  nodes.**
**Left  plot**:  Average  degree  of  magnetic  field  budget  allocation in  the  answer  of  MF-IIM  (yellow  line)  and  IIM  (blue  line)  as  a  function of interaction strength, β.
**Right plot**:  Relative performance of h<sub>MF-IIM</sub> ascompared  to h measured  as  the  ratio  of  corresponding  magnetisations,  i.e. M(h<sub>MF-IIM</sub>)/M(h).


<p align="center">
<img src="https://github.com/kikatuso/Master-Thesis/blob/main/img/exp2.png" alt="stochastic_block" width="900"/>
</p>

**Single agent MF-IIM for a stochastic network with 100 nodes, split into two equal-sized blocks - one tightly-connected and one loosely-connected block.**
**Left plot**:  Share of magnetic budget allocation to each blockas a function of interaction strength.
**Right plot**:  Comparison of MF-IIM solution to other commonly-used node-selection heuristic methods as a function of total available budget.


### Competitive optimisation models 

To run **simultaneous** game simulations use either *sim_game_numba.py* or *sim_game_numpy.py*. These two yield the same results. 

*sim_game_numpy.py* gives a wide range of optimisers to use (sgd,sgdm,adadelta,adam), whereas *sim_game_numba.py* only uses **adam** optimiser. *sim_game_numba.py* uses *numba compiler* that results in a faster execution of simulations. 

This performance improvement has not been exactly measured, but it is approximately at least **20-times** faster than *sim_game_numpy.py*. *sim_game_numba.py* is hence recommended when dealing with large networks. 

For **sequential** game, please use *seq_game_numba.py*. Sequential game algorithm only has *numba* implementation. 

<p align="center">
<img src="https://github.com/kikatuso/Master-Thesis/blob/main/img/Budget_pos1.0_Budget_neg2.0.png" alt="competitive_hub_and_spoke" width="400"/>
</p>

Two-agent  competitive  MF-IIM  structured  as  a  simultaneous game  for  a  hub-and-spoke  network  with  4  peripheral  nodes. 
Figure  shows multiple paths of convergence of competitive optimisation with simultaneousgame update dynamics for a set of different initial external field allocation and interaction strength of β=0.2, B<sub>pos</sub>=1.0 and and B<sub>neg</sub>=2.0 

<p align="center">
<img src="https://github.com/kikatuso/Master-Thesis/blob/main/img/network.png" alt="network_influenced" width="600"/>
</p>

Two-agent  competitive  MF-IIM  structured  as  a  simultaneousgame for a real world collaboration network with 379 nodes and 914 edges.
Figure shows spatial location of influenced nodes for equal budgets, each equalto B=40.0, at β=0.96.


