import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import os
import random
from networkx.algorithms import community
import community as community_louvain
from collections import defaultdict
from collections import Counter
import argparse
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import time
import argparse



### Select the method to create folder 
# options: 'palla', 'louvain', 'girvan_newman'
method = 'palla'

# Put Folder name based on dataset to save the results
folder_name = 'facebook_combined'

# Directory to save the embeddings and clustering results
save_path = './' + folder_name + "/" + method + "/"


### SAVE RESULTS
## Create a folder to save the results
def create_folder():
    """
    Create a folder to save the clustering results.
    
    This function checks if the specified save_path exists. If it doesn't, it creates the folder
    and prints a message indicating that the folder has been created.
    
    Returns:
        None
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Folder created")
    
    return


### CONSTRUCT NETWORK
## Description of the graph
def description_of_graph(G):
    """
    Calculate and save various properties of a graph.

    Parameters:
    - G: NetworkX graph object

    Returns:
    None
    """
    # Calculate the number of nodes and edges in the graph
    number_of_nodes = G.number_of_nodes()
    number_of_edges = G.number_of_edges()

    # Calculate the average degree of nodes
    average_degree = calculate_average_degree(G)

    # Calculate the density of the graph
    density = nx.density(G)

    # Calculate the clustering coefficients
    clustering_coefficients, average_clustering_coefficient = calculate_clustering_coefficients(G)

    # Create a dictionary to store the description of the graph
    description = {
        'number_of_nodes': number_of_nodes,
        'number_of_edges': number_of_edges,
        'average_degree': round(average_degree, 3),
        'density': round(density, 3),
        'clustering_coefficients': {node: round(coefficient, 3) for node, coefficient in clustering_coefficients.items()},
        'average_clustering_coefficient': round(average_clustering_coefficient, 3),
    }

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(description, orient='index', columns=['Value'])
    df.index.name = 'Description'

    # Save the DataFrame to a csv file
    df.to_csv(save_path + method + '_description.csv')

    print("Network Description saved to csv")

    return


## Draw the network
def draw_large_network(G):
    """
    Visualize a subset of the network for large graphs.
    
    Args:
        G (networkx.Graph): The graph to visualize.
        
    Returns:
        None
    """
    # Set the layout for the graph
    pos = nx.spring_layout(G, seed=42)  # Using seed for reproducibility

    # Draw only a subset of nodes and edges
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_color='lightblue', node_size=200, alpha=0.5)

    # Draw labels
    labels = {node: node for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=2)

    plt.title("Subset of the Large Network")

    # Save the plot as an image
    plt.savefig(os.path.join(save_path, "node_network.png"))

    # Close the plot
    plt.close()

## Construct a network from a file or test data
def construct_network(file_path, test_data, node_range=None, sample_size=None):
    """
    Construct a network from a file or test data.

    Parameters:
    - file_path: str, path to the file containing the network data
    - test_data: list of tuples, test data representing edges in the network
    - node_range: tuple of ints, optional, range of nodes to consider
    - sample_size: int, optional, desired sample size of the network

    Returns:
    - G: NetworkX graph
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

## Select the community detection method and identify communities in the graph
def select_community_detection(method="palla"):
    """
    Select the community detection method and identify communities in the graph.

    Parameters:
    - method: str, optional, the community detection method to use (default: "palla")

    Returns:
    - communities: list of sets of nodes representing communities
    """

    if method == "palla":
        # Identify communities using the Clique Percolation Method
        communities = identify_communities_palla(G, k=args.k, overlapping=args.overlapping)
        
    elif method == "louvain":
        # Identify communities using the Louvain method
        communities = identify_communities_louvain(G)  

    elif method == "girvan_newman":
        # Identify communities using the Girvan-Newman method
        communities = identify_communities_girvan_newman(G)

    else:
        raise ValueError("Invalid community detection method. \
                         Please select 'palla' or 'louvain' or 'girvan_newman'.")

    
    return communities



### COMMUNITY DETECTION METHODS
# Using Clique Percolation Method to Identify Communities (palla)
def identify_communities_palla(G, k=3, overlapping=False):
    """
    Identifies communities in the network using the Clique Percolation Method.

    Parameters:
        G (NetworkX graph): The input graph.
        k (int): The minimum size of cliques to consider as communities. Default is 3.
        overlapping (bool): Whether to allow overlapping communities. Default is False.

    Returns:
        list: A list of communities, where each community is represented as a set of nodes.
    """
    cliques = list(nx.find_cliques(G))

    k_cliques = [clique for clique in cliques if len(clique) >= k]

    if overlapping:     
        communities = []
        for k_clique in k_cliques:
            for node_combination in combinations(k_clique, k):
                subgraph = G.subgraph(node_combination)
                if nx.is_connected(subgraph):
                    communities.append(set(node_combination))

    if not overlapping:
        node_community_mapping = {}
        communities = []

        for k_clique in k_cliques:
            k_clique_set = set(k_clique)

            existing_community = None
            for node in k_clique_set:
                if node in node_community_mapping:
                    existing_community = node_community_mapping[node]
                    break

            if existing_community is not None:
                existing_community.update(k_clique_set)
            else:
                new_community = set(k_clique_set)
                communities.append(new_community)
                for node in new_community:
                    node_community_mapping[node] = new_community

        for node in G.nodes:
            if node not in node_community_mapping:
                new_community = {node}
                communities.append(new_community)
                node_community_mapping[node] = new_community

    return communities    

# identify community using louvain method
def identify_communities_louvain(G):
    """
    Identifies communities in the network using the Louvain method.

    Parameters:
    - G: NetworkX graph object

    Returns:
    - communities: A dictionary mapping nodes to their respective community labels
    """
    communities = community_louvain.best_partition(G)
    return communities


# identify community using girvan_newman method
def identify_communities_girvan_newman(G):
    """
    Identifies communities in a graph using the Girvan-Newman algorithm.

    Parameters:
    - G: NetworkX graph
        The input graph on which to identify communities.

    Returns:
    - communities: list of tuples
        A list of communities, where each community is represented as a tuple of nodes.

    """
    communities = community.girvan_newman(G)

    return communities




### CLUSTERING ANALYSIS - HELPER FUNCTIONS
## Calculate the average degree of a graph
def calculate_average_degree(G):
    """
    Calculate the average degree of a graph.

    Parameters:
    G (networkx.Graph): The input graph.

    Returns:
    float: The average degree of the graph.
    """
    # Calculate the sum of degrees of all nodes
    total_degrees = sum(dict(G.degree()).values())
    
    # Calculate the average degree
    average_degree = total_degrees / len(G.nodes())
    
    return average_degree


## Calculate the clustering coefficients of each node in a graph and the average clustering coefficient of the graph
def calculate_clustering_coefficients(G):
    """
    Calculate the clustering coefficients of each node in a graph and the average clustering coefficient.

    Parameters:
    G (networkx.Graph): The input graph.

    Returns:
    clustering_coefficients (dict): A dictionary mapping each node to its clustering coefficient.
    average_clustering_coefficient (float): The average clustering coefficient of the graph.
    """
    # Calculate the clustering coefficient of each node
    clustering_coefficients = nx.clustering(G)
    
    # Calculate the average clustering coefficient
    average_clustering_coefficient = nx.average_clustering(G)
    
    return clustering_coefficients, average_clustering_coefficient




### ANALYZE COMMUNITIES DETECTED
## Analyze community using palla method
def analyze_communities_palla(G, communities):
    """
    Analyzes the community structure of a graph using the Palla method.

    Args:
        G (networkx.Graph): The input graph.
        communities (list): A list of communities, where each community is represented as a list of nodes.

    Returns:
        None

    Raises:
        None

    """

    # Analyze the community structure
    try:
        modularity = community.modularity(G, communities)
    except Exception:
        print("Cannot calculate modularity for overlapping or communities not in clique")
        modularity = None

    # Flatten the list of communities into a list of nodes
    all_nodes = [node for commun in communities for node in commun]

    # Count the number of times each node appears
    node_counts = Counter(all_nodes)

    # Find the number of nodes that appear more than once
    overlapping_nodes = sum(1 for node, count in node_counts.items() if count > 1)

    # Calculate the graph cut for each community
    graph_cuts = []
    for commun in communities:
        cut = 0
        for node in commun:
            for neighbor in G.neighbors(node):
                if neighbor not in commun:
                    cut += 1
        graph_cuts.append(cut)


    # Calculate the graph cut for each pair of communities
    graph_cuts_pairs = {}
    for i in range(len(communities)):
        for j in range(i+1, len(communities)):
            cut = 0
            for node in communities[i]:
                for neighbor in G.neighbors(node):
                    if neighbor in communities[j]:
                        cut += 1
            graph_cuts_pairs[(i, j)] = cut

    # Calculate the conductance for each community
    conductances = []
    for commun in communities:
        volume = len(commun)

        # Calculate the cut of the community
        cut = sum(1 for u, v in G.edges(commun) if v not in commun)

        # Calculate the conductance
        conductance = cut / volume if volume != 0 else 0
        conductances.append(round(conductance, 3))

    
    # calculate the density for each community
    densities = []
    for commun in communities:
        subgraph = G.subgraph(commun)
        density = nx.density(subgraph)
        densities.append(round(density, 3))


    analysis_2 = {
        'number_of_communities': len(communities),
        'average_community_size': round(sum(len(commun) for commun in communities) / len(communities)) if communities else 0,
        'modularity': round(modularity, 3) if modularity is not None else None,
    }


    analysis_1 = {
        'community_number': [i for i, _ in enumerate(communities, start=1)],
        'community_size': [len(commun) for commun in communities],
        'community_details': communities,
        'overlapping_nodes': overlapping_nodes, 
        'edge_graph_cuts': graph_cuts, 
        'conductance': conductances,  
        'density': densities, 
    }

    
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(list(graph_cuts_pairs.items()), columns=['Pair', 'Cut'])

    # Save the DataFrame to a csv file
    df.to_csv(save_path + method +'_graph_cuts_pairs.csv', index=False)


    # Convert the dictionary to a Series
    analysis_2_series = pd.Series(analysis_2)
    # Convert the Series to a DataFrame and reset the index
    df_2 = pd.DataFrame(analysis_2_series).reset_index()
    # Rename the columns
    df_2.columns = ['Analysis', 'Value']
    # Save the DataFrame to a csv file
    df_2.to_csv(save_path + method + '_community_analysis.csv', index=False)


    # save to csv
    df_1 = pd.DataFrame(analysis_1)
    df_1.to_csv(save_path + method + '_hl_analysis.csv', index=False)

    print("Community Analysis saved to csv")

    return 


## Analyze community using louvain method
def analyze_communities_louvain(G, communities):
    """
    Analyzes the community structure of a graph using the Louvain method.

    Args:
        G (networkx.Graph): The input graph.
        communities (dict): A dictionary mapping nodes to their corresponding communities.

    Returns:
        None
    """

    # Convert the node-community mapping into a list of communities
    communities_dict = defaultdict(list)
    for node, commun in communities.items():
        communities_dict[commun].append(node)
    communities_list = list(communities_dict.values())

    # Analyze the community structure using the Louvain method
    modularity = community_louvain.modularity(communities, G)

    # Calculate the graph cut for each community
    graph_cuts = []
    for commun in communities_list:
        cut = 0
        for node in commun:
            for neighbor in G.neighbors(node):
                if neighbor not in commun:
                    cut += 1
        graph_cuts.append(cut)

    # Calculate the graph cut for each pair of communities
    graph_cuts_pairs = {}
    for i in range(len(communities_list)):
        for j in range(i+1, len(communities_list)):
            cut = 0
            for node in communities_list[i]:
                for neighbor in G.neighbors(node):
                    if neighbor in communities_list[j]:
                        cut += 1
            graph_cuts_pairs[(i, j)] = cut

    # Calculate the conductance for each community
    conductances = []
    for commun in communities_list:
        volume = len(commun)
        cut = sum(1 for u, v in G.edges(commun) if v not in commun)
        conductance = cut / volume if volume != 0 else 0
        conductances.append(round(conductance, 3))

    # Calculate the density for each community
    densities = []
    for commun in communities_list:
        E = G.subgraph(commun).number_of_edges()
        V = len(commun)
        density = 2*E / (V*(V-1)) if V > 1 else 0
        density = round(density, 3)
        densities.append(density)

    # Prepare analysis results
    analysis_2 = {
        'average_community_size': round(sum(len(commun) for commun in communities_list) / len(communities_list)) if communities_list else 0,
        'modularity': round(modularity, 3) if modularity is not None else None,
    }

    analysis_1 = {
        'edge_graph_cuts': graph_cuts,
        'conductance': conductances,
        'density': densities,
    }

    # Convert the dictionary to a DataFrame
    df_3 = pd.DataFrame(list(communities_dict.items()), columns=['Community', 'Nodes'])
    df_3['Size'] = df_3['Nodes'].apply(len)

    # Save the DataFrame to a csv file
    df_3.to_csv(save_path + method + 'comun.csv', index=False)

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(list(graph_cuts_pairs.items()), columns=['Pair', 'Cut'])

    # Save the DataFrame to a csv file
    df.to_csv(save_path + method + '_graph_cuts_pairs.csv', index=False)

    # Convert the dictionary to a Series
    analysis_2_series = pd.Series(analysis_2)
    df_2 = pd.DataFrame(analysis_2_series).reset_index()
    df_2.columns = ['Analysis', 'Value']
    df_2.to_csv(save_path + method + '_community_analysis.csv', index=False)

    # Save analysis_1 to a csv file
    df_1 = pd.DataFrame(analysis_1)
    df_1.to_csv(save_path + method + '_hl_analysis.csv', index=False)

    # Merge the two dataframes
    df_4 = pd.concat([df_1, df_3], axis=1)
    df_4.to_csv(save_path + method + '_hl_analysis.csv', index=False)

    print("Community Analysis saved to csv")

    return


## Analyze community using girvan_newman method
def analyze_communities_girvan_newman(G, communities, num_levels=None):
    """
    Analyze the community structure using the Girvan-Newman method.
    
    Parameters:
    - G: NetworkX graph
    - communities: iterator of sets of nodes representing communities
    - num_levels: int, optional, number of levels to analyze
    
    Returns:
    - None
    """
    if num_levels is None:
        num_levels = 3

    # Analyze the community structure using the Girvan-Newman method
    level_communities = []
    for _ in range(num_levels):
        level_communities.append(next(communities))

    modularities = []
    for level in level_communities:
        modularity = community.modularity(G, level)
        modularities.append(round(modularity, 3) if modularity is not None else None)

    graph_cuts = []
    for level in level_communities:
        graph_cuts_level = []
        for commun in level:
            cut = 0
            for node in commun:
                for neighbor in G.neighbors(node):
                    if neighbor not in commun:
                        cut += 1
            graph_cuts_level.append(cut)
        graph_cuts.append(graph_cuts_level)

    graph_cuts_pairs = []
    for level in level_communities:
        graph_cuts_pairs_level = []
        for i in range(len(level)):
            for j in range(i + 1, len(level)):
                cut = 0
                for node in level[i]:
                    for neighbor in G.neighbors(node):
                        if neighbor in level[j]:
                            cut += 1
                graph_cuts_pairs_level.append(cut)
        graph_cuts_pairs.append(graph_cuts_pairs_level)

    # Calculate the conductance for each community
    conductances = []
    for level in level_communities:
        conductances_level = []
        for commun in level:
            volume = len(commun)

            # Calculate the cut of the community
            cut = sum(1 for u, v in G.edges(commun) if v not in commun)

            # Calculate the conductance
            conductance = cut / volume if volume != 0 else 0
            conductances_level.append(round(conductance, 3))
        conductances.append(conductances_level)

    analysis = {
        'number_of_last_level_communities': len(level_communities[-1]),
        'average_community_size': round(sum(len(commun) for commun in level_communities[-1]) / len(level_communities[-1])) if level_communities else 0,
        'community_details': level_communities,
        "top level communities: ": sorted(map(sorted, level_communities[0])),
        'Last_level_communities': sorted(map(sorted, level_communities[-1])),
        'modularities': modularities,
        'edge_graph_cuts': graph_cuts,
        'inter_community_edge_graph_cuts': graph_cuts_pairs,
        'conductances': conductances,
    }

    # Convert the analysis dictionary to a list of tuples
    analysis_items = list(analysis.items())

    # Create a DataFrame from the list of tuples
    df = pd.DataFrame(analysis_items, columns=['Analysis', 'Value'])

    # Save to csv
    df.to_csv(save_path + method + '_feeranalysis.csv', index=False)

    return



### VISUALIZE COMMUNITIES
## Draw network with identified communities using palla method
def draw_network_with_palla_communities(G, communities):
    """
    Draw the network with identified communities using the Palla method.
    
    Parameters:
    - G: NetworkX graph
    - communities: list of sets of nodes representing communities
    
    Returns:
    - None
    """
    # Set the layout for the graph
    pos = nx.spring_layout(G, seed=42)  # Using seed for reproducibility

    # Draw the network with identified communities
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_color='lightblue', node_size=200, alpha=0.5)

    # Highlight communities
    for i, commun in enumerate(communities):
        nx.draw_networkx_nodes(G, pos, nodelist=list(commun), node_color=f"C{i+1}", node_size=200, label=f'Community {i+1}')

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    # Draw labels
    labels = {node: node for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=10)

    plt.title("Network with Identified Communities")
    plt.legend()
    
    plt.savefig(os.path.join(save_path + method +"_community_network.png"))
    plt.close()

## Draw network with identified communities using louvain method
def draw_network_with_louvain_communities(G, communities):
    """
    Draw the network with identified communities using the Louvain method.
    
    Parameters:
    - G: NetworkX graph
    - communities: dictionary mapping nodes to community labels
    
    Returns:
    - None
    """
    # Create a color map
    colors = [communities[node] for node in G.nodes()]

    # Draw the graph
    nx.draw(G, node_color=colors, with_labels=True)

    # Create a custom legend
    unique_communities = set(communities.values())
    legend_handles = [mpatches.Patch(color=plt.cm.viridis(i / len(unique_communities)), label=f'Community {i+1}') for i in unique_communities]
    plt.legend(handles=legend_handles)

    plt.title("Network with Louvain Communities")
    plt.savefig(os.path.join(save_path + method +"_community_network.png"))
    plt.close()

## Draw network with identified communities using girvan_newman method
def draw_network_with_girvan_newman_communities(G, communities, num_levels=None):
    """
    Draw the network with identified communities using the Girvan-Newman method.
    
    Parameters:
    - G: NetworkX graph
    - communities: iterator of sets of nodes representing communities
    - num_levels: int, optional, number of levels to visualize
    
    Returns:
    - None
    """
    # Identify communities using the Girvan-Newman method
    communities = community.girvan_newman(G)

    if num_levels is None:
        num_levels = 3

    level_communities = []
    for _ in range(num_levels):
        level_communities.append(next(communities))

    last_level_communities = sorted(map(sorted, level_communities[-1]))

    # Create a dictionary where the keys are nodes and the values are normalized colors
    node_colors = {}
    for i, commun in enumerate(last_level_communities):
        for node in commun:
            node_colors[node] = i / len(last_level_communities)  # Normalize the color

    # Create a color map
    colors = [node_colors[node] for node in G.nodes()]

    # Create a colormap
    cmap = plt.cm.viridis

    # Draw the graph
    nx.draw(G, node_color=colors, cmap=cmap, with_labels=True)

    # Create a custom legend
    legend_handles = [mpatches.Patch(color=cmap(i / len(last_level_communities)), 
                                     label=f'Community {i + 1}') for i in range(len(last_level_communities))]
    
    plt.legend(handles=legend_handles)

    plt.title("Network with Girvan-Newman Communities")

    plt.savefig(os.path.join(save_path + method +"_community_network.png"))
    plt.close()



### Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool to analyze community structure in networks")
    parser.add_argument("file_path", type=str, help="Path to the file containing the network data")
    parser.add_argument("--node_range", type=int, nargs=2, help="Range of nodes to consider")
    parser.add_argument("--sample_size", type=int, help="Desired sample size of the network")
    parser.add_argument("--overlapping", action="store_true", default=False, help="Flag to indicate whether to allow overlapping communities")
    parser.add_argument("--k", type=int, default=None, help="Size of the k-cliques to use for identifying communities")
    parser.add_argument("--method", type=str, default="palla", help="Method to use for community detection (palla, louvain or girvan_newman)")
    parser.add_argument("--num_levels", type=int, default=3, help="Number of levels for the Girvan-Newman method")
    args = parser.parse_args()

    # Check for valid input
    if args.method != 'palla' and (args.k is not None or args.overlapping):
        print("Wrong input: --k and --overlapping options are only valid for the 'palla' method")
        exit(1)

    if args.method == 'palla' and (args.k is None):
        print("Wrong input: --k option is required for the 'palla' method")
        exit(1)

    if args.method == 'girvan_newman' and args.num_levels is None:
        print("Wrong input: --num_levels option is required for the 'girvan_newman' method")
        exit(1)

    # test data
    data = [(1, 2), (1, 5), (2, 3), (2, 4), (2, 5), (3, 5), (3, 4), (4, 5), (4, 6), (6, 7), (5, 7), (7, 9), (7, 8), (5, 10), (10, 11), (11, 12), (10, 12), (8, 9)]
    
    # Create a folder to save the results
    create_folder()

    # Construct the network
    G = construct_network(args.file_path, data, args.node_range, args.sample_size)

    # Draw a subset of the large network
    draw_large_network(G)

    # Description of the graph
    description = description_of_graph(G)
    # print("Description:", description)

    print("node range:", args.node_range)
    print("sample size:", args.sample_size)

    # Calculate time taken to run the code
    start_time = time.time()

    # Description of communities detection method
    if args.method == "palla" or args.method == "louvain" or args.method == "girvan_newman":
        print("Community detection method:", args.method)
    elif args.method is None:
        print("Community detection method: palla (default)") 
    else:
        raise ValueError("Invalid community detection method. Please select 'palla' or 'louvain' or 'girvan_newman'.")

    if args.method == "palla":
        print("Size of k-cliques:", args.k if args.k is not None else "default (3)")
        print("Allow overlapping communities:", args.overlapping)

    if args.method == "girvan_newman":
        print("Number of levels:", args.num_levels if args.num_levels is not None else "default (4)")

    # Identify communities
    communities = select_community_detection(args.method)

    # Analyze the community structure
    if args.method == "palla":
        analysis = analyze_communities_palla(G, communities)
    elif args.method == "louvain":
        analysis = analyze_communities_louvain(G, communities)
    elif args.method == "girvan_newman":
        analysis = analyze_communities_girvan_newman(G, communities, args.num_levels)
    else:
        raise ValueError("Invalid community detection method. \
                         Please select 'palla' or 'louvain' or 'girvan_newman'.")

    # print("Analysis:", analysis)

    try:
        if args.method == "palla":
            draw_network_with_palla_communities(G, communities)
        elif args.method == "louvain":
            draw_network_with_louvain_communities(G, communities)
        elif args.method == "girvan_newman":
            draw_network_with_girvan_newman_communities(G, communities, args.num_levels)
    except Exception as e:
        print(f"An error occurred: {e}")

    # save time to a file
    with open(os.path.join(save_path, 'time.txt'), 'w') as f:
        f.write("--- %s seconds ---" % (time.time() - start_time))

    print("The code ran successfully")




    
# To run the script, use the following command:
# python Traditional_method.py <datafilepath> --method <palla or louvain or girvan_newman>

# python Traditional_method.py facebook_combined.txt --method palla --k 3 --overlapping
# python Traditional_method.py facebook_combined.txt --method louvain
# python Traditional_method.py facebook_combined.txt --method girvan_newman --num_levels 3
# python Traditional_method.py facebook_combined.txt --method palla --k 3 --overlapping --node_range 1 1000
# python Traditional_method.py facebook_combined.txt --method palla --k 3 --overlapping --sample_size 1000
# python Traditional_method.py facebook_combined.txt --method palla --k 3 --overlapping --node_range 1 1000 --sample_size 1000
# python Traditional_method.py facebook_combined.txt --method palla --k 3 --overlapping --node_range 1 1000 --sample_size 1000
# python Traditional_method.py facebook_combined.txt --method palla --k 3 --overlapping --node_range 1 1000 --sample_size 1000
# python Traditional_method.py facebook_combined.txt --method palla --k 3 --overlapping --node_range 1 1000 --sample_size 1000


