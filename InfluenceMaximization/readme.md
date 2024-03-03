### Introduction
#### Influence Maximization
Traditional Heuristic approach: Susceptible-Infected-Recovered (SIR) model, the Independent Cascade (IC) Model, and the Linear Threshold Model (LTM) (Greedy and CELF).

Machine Learning: Node2Vec using RandomForestRegressor

Metrics defined: Degree centrality, Pagerank, HITS, k-core, Neighborhood coreness, Number of infected nodes.

#### Script Implementation
pip install -r requirements.txt

##### To run the script for deifferent approach, Consider selecting a method adding --method (Can be IC, TM, CM) 

```
python methods.py ./twitch/ENGB/musae_ENGB_edges.csv 
```

Note:
- The script will analyze the influence maximization of the all models in the network and visualize the identified communities.
- The command-line arguments are used to specify the path to the file containing the network data.
- Edge file should follow format of the examples.
- Save all the results to the Result directory.
    
