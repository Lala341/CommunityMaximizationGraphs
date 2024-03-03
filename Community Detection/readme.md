### Introduction
#### Community Detection
Traditional Heuristic approach: Clique Percolation Method, Custom Clique Percolation Method (Non-overlapping), Louvain Method, and Girvan-Newman Method.

Machine Learning: Node2Vec or Graph Attention Networks

Metrics defined: density, edge cuts, inter-community pairwise cut,  modularity, clustering coefficient and conductance, node degrees and time taken to run algorithm.

#### Script Implementation
pip install -r requirements.txt

##### To run the script for Traditional Heuristic approach, Consider selecting a method from the below:

```
python Traditional_method.py facebook_combined.txt --method palla --k 3
python Traditional_method.py facebook_combined.txt --method palla --k 3 --overlapping
python Traditional_method.py facebook_combined.txt --method louvain
python Traditional_method.py facebook_combined.txt --method girvan_newman --num_levels 3
```
Different values of k and num_levels can be indicated.



##### To run the script for Machine Learnining approach:
```
Edit the script to initially select the embeddings method to use: 'n2vec' or 'gat'
# For n2vec embeddings, clustering type options are : Kmeans, Spectral, Agglomerative
# For GAT embeddings, clustering type options are: Kmeans

Then execute script:

python Graph_Embeddings.py "facebook_combined.txt"
```

For both approaches 
To indicate a range of nodes or sample size of the network. Execute as above and include other arguments. For example:
```
 python Traditional_method.py facebook_combined.txt --method palla --k 3 --overlapping --node_range 1 1000 --sample_size 1000
```

Note:
- The script will analyze the community structure in the network and visualize the identified communities.
- The command-line arguments are used to specify the path to the file containing the network data, the range of nodes to consider, and the desired sample size of the network.
- The script will also use test data if the file does not exist.
- Edge file should follow format of the examples provided and be without header.
- The identified communities will be analyzed, and the community structure will be visualized.
- All the results are saved to the root directory.
    
