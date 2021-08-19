# Optimal Strategies in Two player Ising Influence Maximisation Problem.
**Aims of the Project**: 
The project aims to analyse quantitatively behaviours and strategies of two adversarial agents with an objective to maximise their influence on 
the same subset of a social network. This scenario is widely known in real world and is applicable to e.g. campaigns of two political counter-candidates 
or smartphone companies aiming to sell their product. This framework will be implemented with the use of the **Ising Model** – 
a physical model that quantitatively can express e.g. relationships between social network members. 

In particular, the project studies the influence allocation of each of two external agents given the allocation of the counter-competitor and agents’ 
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

Function *single_agent* to run Mean-Field Ising Magnetisation Maximisation algorithm. 

