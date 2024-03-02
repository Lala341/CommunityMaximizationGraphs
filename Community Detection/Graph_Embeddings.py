# !pip install Node2Vec
# !pip install torch_geometric

import networkx as nx
from node2vec import Node2Vec
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import haversine_distances
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.decomposition import PCA
from networkx.algorithms.community import modularity
import os
import random
import time
import torch
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader
import networkx as nx
import warnings
import argparse
import os
import random
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F


warnings.simplefilter('ignore', DeprecationWarning)


# Select the method to use for embeddings: 'n2vec' or 'gat'
emb = 'n2vec'

# Change the method to use for comparison of the clustering results if it exists
# Select the method to use: 'n2vec' or 'gat'
emb_2 = 'n2vec'

# Clustering algorithms
# For n2vec embeddings, options are: Kmeans, Spectral, Agglomerative
# For GAT embeddings, options are: Kmeans
clustering_type = 'Kmeans'

# Change the type to use for comparison of the clustering results if it exists
clustering_type_2 = 'Kmeans'

# Put the folder name based on the dataset to save the results
folder_name = 'facebook'

# Directory to save the embeddings and clustering results
save_path = './' + emb + "/" + folder_name + "/"
save_path_2 = './' + emb_2 + "/" + folder_name + "/"

algo_path = save_path + clustering_type + "/"

lv_1 = save_path  + clustering_type + '_labels' + '.npy'
lv_2 = save_path_2  + clustering_type_2 + '_labels' + '.npy'



### SAVE RESULTS
## Create a folder to save the results
def create_folder():
    """
    Create a folder to save the clustering results.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(algo_path):
        os.makedirs(algo_path)


### CONSTRUCT NETWORK
## Description of the graph
def construct_network(file_path, test_data, node_range=None, sample_size=None):
    """
    Construct a network from a file or test data.

    Args:
        file_path (str): Path to the file containing the network data.
        test_data (list): Test data to use if the file does not exist.
        node_range (tuple, optional): Range of nodes to include in the network.
        sample_size (int, optional): Number of nodes to sample from the network.

    Returns:
        G (networkx.Graph): Constructed network.
    """
    # Check if the file exists
    if os.path.exists(file_path) and file_path.endswith('.txt'):
        # If the file exists, load the data from the file
        G = nx.read_edgelist(file_path, nodetype=int)
    elif os.path.exists(file_path) and file_path.endswith('.csv'):
        G = nx.read_edgelist(file_path, delimiter=',', nodetype=int)
    elif os.path.exists(file_path) and file_path.endswith('.json'):
        G = nx.readwrite.json_graph.node_link_graph(file_path)
    else:
        # If the file does not exist, use the test data
        G = nx.Graph()
        print("File does not exist. Using test data.")
        for edge in test_data:
            for i in range(len(edge) - 1):
                G.add_edge(edge[i], edge[i+1])

    # If a node range is provided, filter out the edges that do not have both nodes within the desired range
    if node_range is not None:
        edges_to_remove = [(u, v) for u, v in G.edges() if not (node_range[0] <= u <= node_range[1] and node_range[0] <= v <= node_range[1])]
        G.remove_edges_from(edges_to_remove)

        # Remove isolated nodes
        G.remove_nodes_from(list(nx.isolates(G)))

    nodes = list(G.nodes())

    # If the graph has more nodes than the desired sample size
    if sample_size is not None and len(nodes) > sample_size:
        # Randomly select a subset of nodes
        sampled_nodes = random.sample(nodes, sample_size)

        # Create a new graph that contains only the sampled nodes and the edges between them
        G = G.subgraph(sampled_nodes).copy()
    elif sample_size is None:
        print("No sample size selected. Using the original graph.")
    else:
        # If the graph has less or equal nodes than the desired sample size, return the original graph
        print("Sample size too large, returning the original graph.")

    return G


## Draw the network
def draw_large_network(G):
    """
    Visualize a subset of the network for large graphs.

    Args:
        G (networkx.Graph): The graph to visualize.
    """
    # plt.figure(figsize=(5, 5))
    pos = nx.spring_layout(G, seed=42)  # Using seed for reproducibility

    # Draw only a subset of nodes and edges
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_color='lightblue', node_size=200, alpha=0.5)

    # Draw labels
    labels = {node: node for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=2)

    plt.title("Node Network")

    plt.savefig(os.path.join(save_path + "node_network.png"))

    # plt.show()
    plt.close()


### GENERATE EMBEDDINGS USING NODE2VEC AND GAT
## Using Node2Vec
# Define node2vec
def generate_node2vec_emb(G):
    """
    Generate embeddings using node2vec.

    Args:
        G (networkx.Graph): The graph to generate embeddings for.

    Returns:
        embeddings (numpy.ndarray): The generated embeddings.
    """
    # Generate embeddings using node2vec
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # Get embeddings for all nodes in the graph
    embeddings = model.wv.vectors

    # Save embeddings for later use
    model.save(os.path.join(save_path, "node2vec_model.model"))
    model.wv.save_word2vec_format(os.path.join(save_path,"node2vec_embeddings.txt"))

    return embeddings


## Using GAT
# Define a simple GAT model
class Net(torch.nn.Module):
    """
    A simple Graph Attention Network (GAT) model.
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GATConv(1, 16)  # 1 input feature
        self.conv2 = GATConv(16, 1)  # 1 output class (change this to your desired output dimension)

    def forward(self, data):
        """
        Forward pass of the GAT model.

        Args:
            data (torch_geometric.data.Data): The input data containing node features and edge indices.

        Returns:
            torch.Tensor: The output logits of the model.
        """
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


def train_GAT_model(G, data, epochs=200, print_every=20, test=False):
    """
    Train a GAT model on the given graph data.

    Args:
        G (networkx.Graph): The input graph.
        data (list): The input data for the graph.
        epochs (int): The number of training epochs.
        print_every (int): The interval for printing the loss during training.
        test (bool): Whether to test the model after training.

    Returns:
        torch.Tensor: The learned node embeddings.
        torch.nn.Module: The trained GAT model.
        torch_geometric.data.Data: The processed graph data.
    """

    # create edges from G
    edges = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()

    # Determine the size of the tensors based on the total number of nodes
    num_nodes = G.number_of_nodes()

    # Create a tensor of ones for node features
    x = torch.ones((num_nodes, 1))

    # Create a tensor of ones for the training mask
    train_mask = torch.ones(num_nodes, dtype=torch.bool)

    # Create a tensor for the labels (replace this with your actual labels if you have them)
    y = torch.zeros(num_nodes, dtype=torch.long)

    data = Data(x=x, edge_index=edges, train_mask=train_mask, y=y)

    # Initialize and train the model
    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % print_every == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')
            # Save epoch and loss to a file
            with open(os.path.join(algo_path, 'loss.txt'), 'a') as f:
                f.write(f'Epoch: {epoch}, Loss: {loss.item()}\n')

    embeddings = gen_GAT_emb(model, data)

    if test:
        test_GAT_model(model, data)

    return embeddings, model, data


def gen_GAT_emb(model, data):
    """
    Generate node embeddings using a trained GAT model.

    Args:
        model (torch.nn.Module): The trained GAT model.
        data (torch_geometric.data.Data): The processed graph data.

    Returns:
        torch.Tensor: The node embeddings.
    """
    model.eval()
    with torch.no_grad():
        x, edge_index = data.x, data.edge_index
        x = model.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=False)
        embeddings = model.conv2(x, edge_index)

    # Save the embeddings
    torch.save(embeddings, os.path.join(save_path, 'gat_embeddings.pt'))

    return embeddings


def test_GAT_model(model, data):
    """
    Test the performance of a trained GAT model on the test data.

    Args:
        model (torch.nn.Module): The trained GAT model.
        data (torch_geometric.data.Data): The processed graph data.

    Returns:
        float: The accuracy of the model on the test data.
    """
    test_mask = torch.ones(max(max(data)) + 1, dtype=torch.bool)  # replace with your actual test mask
    test_y = torch.zeros(max(max(data)) + 1, dtype=torch.long)  # replace with your actual test labels

    model.eval()
    with torch.no_grad():
        logits = model(data)
        logits = logits[test_mask]
        labels = test_y[test_mask]
        _, predicted = torch.max(logits, dim=1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / test_mask.sum().item()

    print(f'Test Accuracy: {accuracy}')

    return accuracy


## Load embeddings
def load_embeddings(emb='n2vec'):
    """
    Load the embeddings from the specified method.

    Parameters:
    emb (str): The method used for generating the embeddings. Options are 'n2vec' or 'gat'.

    Returns:
    embeddings: The loaded embeddings.
    """
    if emb == 'n2vec':
        # Load the Word2Vec model
        model = Word2Vec.load(save_path + "node2vec_model.model")
        embeddings = model.wv.vectors
    elif emb == 'gat':
        embeddings = torch.load(save_path + 'gat_embeddings.pt')

    return embeddings


## Metrics for the embeddings
def get_metrics(embeddings):
    """
    Calculate metrics for the embeddings.

    Parameters:
    embeddings: The embeddings to calculate metrics for.
    """
    cosine_sim = cosine_similarity(embeddings)
    euclidean_dist = euclidean_distances(embeddings)
    manhattan_dist = manhattan_distances(embeddings)

    print('Embedding metrics')

    metrics_emb = {'cosine_sim': cosine_sim[-1],
                   'euclidean_dist': euclidean_dist[-1],
                   "Manhattan Distance": manhattan_dist[-1]}

    df = pd.DataFrame([metrics_emb])

    # Save the DataFrame to a CSV file in the created folder
    df.to_csv(os.path.join(algo_path, 'metrics_emb.csv'), index=False)




### CLUSTERING/COMMUNITY DETECTION METHODS
## Node2Vec clustering using KMeans, Spectral, or Agglomerative
def node2vec_clustering(embeddings, n_clusters=2, clustering_algorithm=clustering_type, provide_community=False):
    """
    Perform clustering on the embeddings using the specified clustering algorithm.

    Parameters:
    embeddings: The embeddings to perform clustering on.
    n_clusters (int): The number of clusters to create.
    clustering_algorithm (str): The clustering algorithm to use. Options are 'Kmeans', 'Spectral', or 'Agglomerative'.
    provide_community (bool): Whether to provide the community of each node.

    Returns:
    clustering_label: The labels assigned to each node by the clustering algorithm.
    """
    print(f"This is {clustering_algorithm}")

    if clustering_algorithm == 'Kmeans':
        # Use KMeans to cluster the embeddings as default
        clustering = KMeans(n_clusters, random_state=0).fit(embeddings)
    elif clustering_algorithm == 'Spectral':
        clustering = SpectralClustering(n_clusters, assign_labels="discretize", random_state=0).fit(embeddings)
    elif clustering_algorithm == 'Agglomerative':
        clustering = AgglomerativeClustering(n_clusters).fit(embeddings)

    # Create a NetworkX graph from the cosine similarity of the embeddings
    G = nx.from_numpy_array(cosine_similarity(embeddings))

    # Convert the labels from the clustering into a list of sets
    communities = [set(np.where(clustering.labels_ == i)[0]) for i in range(clustering.n_clusters)]

    # Calculate the modularity
    mod = modularity(G, communities)
    mod = round(mod, 3)

    # Save modularity to a txt file
    with open(os.path.join(algo_path, 'modularity.txt'), 'w') as f:
        f.write(f"Modularity: {mod}")

    # Calculate the conductance for each community
    conductances = [nx.algorithms.cuts.conductance(G, community) for community in communities]

    # Calculate the density of each community
    densities = [nx.density(G.subgraph(community)) for community in communities]

    # Calculate the average degree of each community
    avg_degrees = [np.mean([d for n, d in G.subgraph(community).degree()]) for community in communities]

    # Create a DataFrame
    df = pd.DataFrame({
        'Community': range(clustering.n_clusters),
        'Conductance': conductances,
        'Density': densities,
        'Number of Nodes': avg_degrees
    })

    df = df.round(3)

    # Save the DataFrame to a CSV file
    df.to_csv(os.path.join(algo_path, 'community_metrics.csv'), index=False)

    # Visualize the clustering
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    clustering_label = clustering.labels_

    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=clustering.labels_, cmap='viridis')
    plt.xlabel('Embedding dimension 1')
    plt.ylabel('Embedding dimension 2')

    # Save the figure in the new directory
    plt.savefig(os.path.join(algo_path + "n2v_clustering.png"))
    plt.close()

    if provide_community:
        # Provide the community of each node
        for node_id, community in enumerate(clustering.labels_):
            print(f"Node {node_id} belongs to community {community}")

    # Create a variable name dynamically
    labels_variable = clustering_algorithm + '_labels'

    # Save labels to a file in the created folder
    np.save(os.path.join(save_path + labels_variable + '.npy'), clustering.labels_)

    calculate_clustering_metrics(embeddings, clustering)

    view_clustering_results(clustering)

    return clustering_label


## GAT clustering using KMeans
def GAT_clustering_kmeans(embeddings, n_clusters=2):
    """
    Perform KMeans clustering on the given embeddings.

    Parameters:
    - embeddings (numpy.ndarray): The embeddings to be clustered.
    - n_clusters (int): The number of clusters to create.

    Returns:
    - None
    """

    # Get embeddings
    embeddings = embeddings.numpy()

    # Perform KMeans clustering
    clustering = KMeans(n_clusters, random_state=0).fit(embeddings)

    # Create a mapping from KMeans labels to original indices
    label_mapping = {kmeans_label: original_index for original_index, kmeans_label in enumerate(clustering.labels_)}

    # Use the mapping when plotting
    plt.scatter(range(len(embeddings)), embeddings[:, 0], c=[label_mapping[label] for label in clustering.labels_], cmap='viridis')

    # Calculate clustering metrics
    calculate_clustering_metrics(embeddings, clustering)

    # View clustering results
    view_clustering_results(clustering)

    # Create a variable name dynamically
    labels_variable = 'Kmeans_labels'

    # Save labels to a file in the created folder
    np.save(os.path.join(save_path + labels_variable + '.npy'), clustering.labels_)

    print("This is GAT_kmeans")

    # Scatter plot the embeddings with cluster assignments
    plt.title('Clustering of Embeddings')
    plt.xlabel('Data Point Index')
    plt.ylabel('Embedding Dimension 1')
    plt.savefig(os.path.join(algo_path + "gat_cluster.png"))
    plt.close()

    return


### CLUSTERING/COMMUNITY DETECTION ANALYSIS
## calculate metrics for the clustering
def calculate_clustering_metrics(embeddings, clustering):
    """
    Calculate various clustering metrics for evaluating the quality of clustering results.

    Parameters:
    - embeddings (numpy.ndarray): The embeddings used for clustering.
    - clustering (sklearn.cluster): The clustering algorithm used.

    Returns:
    - clustering_metrics (dict): A dictionary containing the calculated clustering metrics.
    """

    silhouette = None
    inertia = None
    db = None
    ch = None

    try:
        silhouette = silhouette_score(embeddings, clustering.labels_)
    except AttributeError:
        print("Silhouette score is not available for this model.")

    try:
        inertia = clustering.inertia_
    except AttributeError:
        print("Inertia is not available for this model.")
    try:
        db = davies_bouldin_score(embeddings, clustering.labels_)
    except AttributeError:
        print("Davies Bouldin score is not available for this model.")
    try:
        ch = calinski_harabasz_score(embeddings, clustering.labels_)
    except AttributeError:
        print("Calinski Harabasz score is not available for this model.")

    print('Clustering metrics')

    clustering_metrics = {'silhouette': round(silhouette, 3) if silhouette else None,
                        'inertia': round(inertia, 3) if inertia else None,
                        'db': round(db, 3) if db else None,
                        'ch': round(ch, 3) if ch else None}

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame([clustering_metrics])

    # Save the DataFrame to a CSV file in the created folder
    df.to_csv(os.path.join(algo_path, 'clustering_metrics.csv'), index=False)

    return clustering_metrics



# View clustering results
def view_clustering_results(clustering_algorithm):


       # Convert labels to list of sets
    emb_Communities = [set(np.where(clustering_algorithm.labels_ == i)[0])
                      for i in range(clustering_algorithm.n_clusters)]

    # Convert the list of sets to a list of lists
    emb_Communities = [list(community) for community in emb_Communities]



    # Convert the list of lists to a DataFrame
    df = pd.DataFrame(emb_Communities)

    # Transpose the DataFrame so each list becomes a column
    df = df.transpose()

            # # Save the DataFrame to a CSV file in the new directory
    df.to_csv(os.path.join(algo_path, 'communities.csv'), index=False)



    return emb_Communities



# compare the clustering results with the ground truth
def compare_clustering_results(lv_1 = lv_1, lv_2 =lv_2):

    #read from npy file
    kmeans_labels = np.load(lv_1, allow_pickle=True)
    clustering_labels = np.load(lv_2, allow_pickle=True)

    # Compute confusion matrix
    cm = confusion_matrix(kmeans_labels, clustering_labels)

    # Solve linear assignment problem
    rows, cols = linear_sum_assignment(-cm)

    # Create a mapping from old labels to new labels
    mapping = {old: new for old, new in zip(cols, rows)}

    # Use this mapping to transform the labels
    clustering_labels_mapped = np.array([mapping[label] for label in clustering_labels])

    # Now you can compute the metrics
    # accuracy = accuracy_score(kmeans_labels, clustering_labels_mapped)
    # f1 = f1_score(kmeans_labels, clustering_labels_mapped)
    # precision = precision_score(kmeans_labels, clustering_labels_mapped)
    # recall = recall_score(kmeans_labels, clustering_labels_mapped)
    # roc_auc = roc_auc_score(kmeans_labels, clustering_labels_mapped)

    # Now you can compute the metrics
    accuracy = accuracy_score(kmeans_labels, clustering_labels_mapped)
    f1 = f1_score(kmeans_labels, clustering_labels_mapped, average='macro')
    precision = precision_score(kmeans_labels, clustering_labels_mapped, average='macro')
    recall = recall_score(kmeans_labels, clustering_labels_mapped, average='macro')

    adj_rand_score = adjusted_rand_score(kmeans_labels, clustering_labels)
    adj_mut_info_score = adjusted_mutual_info_score(kmeans_labels, clustering_labels)
    hom_score = homogeneity_score(kmeans_labels, clustering_labels)

    comparison_metrics = {'accuracy': round(accuracy, 3) if accuracy else None,
                          'f1': round(f1, 3) if f1 else None,
                          'precision': round(precision, 3) if precision else None,
                            'recall': round(recall, 3) if recall else None,
                            # 'roc_auc': round(roc_auc, 3) if roc_auc else None,
                            'adj_rand_score': round(adj_rand_score, 3) if adj_rand_score else None,
                            'adj_mut_info_score': round(adj_mut_info_score, 3) if adj_mut_info_score else None,
                            'hom_score': round(hom_score, 3) if hom_score else None}
        

        # Convert the dictionary to a DataFrame
    df = pd.DataFrame([comparison_metrics])

    # Save the DataFrame to a CSV file in the created folder
    df.to_csv(os.path.join(algo_path, clustering_type_2 + '_comp_met.csv'), index=False)

    return comparison_metrics


### Main function
if __name__ == "__main__":
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Tool to analyze community structure in networks")
    parser.add_argument("file_path", type=str, help="Path to the file containing the network data")
    parser.add_argument("--node_range", type=int, nargs=2, help="Range of nodes to consider", default=None)
    parser.add_argument("--sample_size", type=int, help="Desired sample size of the network", default=None)
    args = parser.parse_args()

    # Print the method to use
    print(f"Using {emb} method")

    # Print the clustering algorithm to use
    print(f"Using {clustering_type} clustering algorithm")

    # Create the folder to save the results
    create_folder()

    # Define the network data
    data = [(1, 2, 6), (0, 1, 5, 9), (2, 3, 9), (0, 2, 4), (2, 5), (3, 5), (3, 4), (4, 5), (4, 6), (6, 7), (5, 7), (7, 9), (7, 8), (5, 10), (10, 11), (11, 12), (10, 12), (8, 9)]

    # Construct the network
    G = construct_network(args.file_path, data, args.node_range, args.sample_size)

    # Calculate time taken to run the code
    emb_start_time = time.time()

    if emb == "n2vec":
        if os.path.exists(save_path + 'node2vec_model.model'):
            embeddings = load_embeddings(emb='n2vec')
        else:
            # Generate Node2Vec embeddings
            embeddings = generate_node2vec_emb(G)

    elif emb == "gat":
        if os.path.exists(save_path + 'gat_embeddings.pt'):
            embeddings = load_embeddings(emb='gat')
        else:
            # Train GAT model and generate embeddings
            min_label = min(G.nodes())
            if min_label != 0:
                raise ValueError("The minimum node label is not 0. Tensorflow Geometric requires node labels to start from 0. Please renumber the nodes and try again.")
            else:
                embeddings, model, data = train_GAT_model(G, data, epochs=500, print_every=50)

    # Calculate metrics for the embeddings
    get_metrics(embeddings)

    # Save time to a file
    with open(os.path.join(algo_path, 'emb_time.txt'), 'w') as f:
        f.write("--- %s seconds to generate embedding ---" % (time.time() - emb_start_time))

    cluster_start_time = time.time()

    if emb == "n2vec":
        # Perform Node2Vec clustering
        clustering_label = node2vec_clustering(embeddings, n_clusters=5, clustering_algorithm=clustering_type, provide_community=False)
        
    elif emb == "gat":
        # Perform GAT clustering
        clustering_label = GAT_clustering_kmeans(embeddings, n_clusters=2)

    # Compare the clustering results
    try: 
        compare_clustering_results(lv_1, lv_2)
    except Exception as e:
        print(e)
        print("There are no results to compare with. Add Comparison results")

    # Save time to a file
    with open(os.path.join(algo_path, 'cluster_time.txt'), 'w') as f:
        f.write("--- %s seconds to create cluster ---" % (time.time() - cluster_start_time))

    print("The code ran successfully")







        










