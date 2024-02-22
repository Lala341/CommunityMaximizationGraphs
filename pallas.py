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
# from node2vec import Node2Vec


def draw_large_network(G):
    # Visualize a subset of the network for large graphs
    # plt.figure(figsize=(5, 5))
    pos = nx.spring_layout(G, seed=42)  # Using seed for reproducibility

    # Draw only a subset of nodes and edges
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_color='lightblue', node_size=200, alpha=0.5)

    # Draw labels
    labels = {node: node for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=2)

    plt.title("Subset of the Large Network")
    plt.show()


# # Embdedding using node2vec
# def node2vec_embedding(G, dimensions=64, walk_length=30, num_walks=200, workers=4, window=10, min_count=1, batch_words=4):
#     # Precompute probabilities and generate walks
#     node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)

#     # Embed nodes
#     model = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)  # Any keywords acceptable by gensim.Word2Vec can be passed

#     # Get node embeddings
#     for node, _ in model.most_similar('2'):  # Use the node name as string
#         print(node)


# identify community using louvain method
def identify_communities_louvain(G):
    # Identify communities in the network using the Louvain method
    communities = community_louvain.best_partition(G)
    return communities


# identify community using girvan_newman method
def identify_communities_girvan_newman(G):
  
    communities = community.girvan_newman(G)

    return communities


## Using Clique Percolation Method to Identify Communities (palla)
def identify_communities_palla(G, k=3, overlapping=False):
    # Identify communities in the network using the Clique Percolation Method
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
        # Create a mapping of nodes to the communities they belong to
        node_community_mapping = {}
        communities = []

        for k_clique in k_cliques:
            # Convert k_clique to a set
            k_clique_set = set(k_clique)

            # Check if any node in the k_clique is already assigned to a community
            existing_community = None
            for node in k_clique_set:
                if node in node_community_mapping:
                    existing_community = node_community_mapping[node]
                    break

            # Update the existing community or create a new one
            if existing_community is not None:
                existing_community.update(k_clique_set)
            else:
                new_community = set(k_clique_set)
                communities.append(new_community)
                # Update node_community_mapping for each node in the new community
                for node in new_community:
                    node_community_mapping[node] = new_community

        # Ensure that every node is part of a community
        for node in G.nodes:
            if node not in node_community_mapping:
                # Create a new community for the node
                new_community = {node}
                communities.append(new_community)
                node_community_mapping[node] = new_community

    return communities


def calculate_average_degree(G):
    # Calculate the sum of degrees of all nodes
    total_degrees = sum(dict(G.degree()).values())
    
    # Calculate the average degree
    average_degree = total_degrees / len(G.nodes())
    
    return average_degree


def calculate_clustering_coefficients(G):
    # Calculate the clustering coefficient of each node
    clustering_coefficients = nx.clustering(G)
    
    # Calculate the average clustering coefficient
    average_clustering_coefficient = nx.average_clustering(G)
    
    return clustering_coefficients, average_clustering_coefficient


def description_of_graph(G):
    # Calculate the number of nodes and edges in the graph
    number_of_nodes = G.number_of_nodes()
    number_of_edges = G.number_of_edges()

    # Calculate the average degree of nodes
    average_degree = calculate_average_degree(G)

    # Calculate the density of the graph
    density = nx.density(G)

    # Calculate the clustering coefficients
    clustering_coefficients, average_clustering_coefficient = calculate_clustering_coefficients(G)

    description = {
        'number_of_nodes': number_of_nodes,
        'number_of_edges': number_of_edges,
        'average_degree': round(average_degree, 3),
        'density': round(density, 3),
        'clustering_coefficients': {node: round(coefficient, 3) for node, coefficient in clustering_coefficients.items()},
        'average_clustering_coefficient': round(average_clustering_coefficient, 3),
    }

    return description


def analyze_communities_palla(G, communities):

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


    analysis = {
        'number_of_communities': len(communities),
        'average_community_size': round(sum(len(commun) for commun in communities) / len(communities)) if communities else 0,
        'community_details': communities,
        'modularity': round(modularity, 3) if modularity is not None else None,
        'overlapping_nodes': overlapping_nodes,  # Add this line
        'edge_graph_cuts': graph_cuts,  # Add this line
        'inter_community_edge_graph_cuts': graph_cuts_pairs,  # Add this line
        'conductance': conductances,  # Add this line
    }


    return analysis


def analyze_communities_louvain(G, communities):
    # Convert the node-community mapping into a list of communities
    communities_dict = defaultdict(list)
    for node, commun in communities.items():
        communities_dict[commun].append(node)
    communities_list = list(communities_dict.values())

    # Analyze the community structure using the Louvain method
    modularity = community_louvain.modularity(communities, G)  # Removed comma here

    # Calculate the graph cut for each community
    graph_cuts = []
    for commun in communities_list:  # Use communities_list here
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

        # Calculate the cut of the community
        cut = sum(1 for u, v in G.edges(commun) if v not in commun)

        # Calculate the conductance
        conductance = cut / volume if volume != 0 else 0
        conductances.append(round(conductance, 3))


    analysis = {
        'number_of_communities': len(communities_list),
        'average_community_size': round(sum(len(commun) for commun in communities_list) / len(communities_list)) if communities_list else 0,
        'community_details': dict(communities_dict),  # Use communities_dict here
        'modularity': round(modularity,3) if modularity is not None else None,
        'edge_graph_cuts': graph_cuts,  # Add this line
        'inter_community_edge_graph_cuts': graph_cuts_pairs,  # Add this line
        'conductances': conductances,  # Add this line
    }

    return analysis


def analyze_communities_girvan_newman(G, communities, num_levels=None):

    if num_levels is None:
        num_levels = 3

    # Analyze the community structure using the Girvan-Newman method
    level_communities = []
    for _ in range(num_levels):
        level_communities.append(next(communities))

    
    modularities = []
    for level in level_communities:
        modularity = community.modularity(G, level)
        modularities.append(round(modularity,3) if modularity is not None else None)
        

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
            for j in range(i+1, len(level)):
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

    return analysis


def draw_network_with_palla_communities(G, communities):
    # plt.figure(figsize=(6, 6))
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
    plt.show()


def draw_network_with_louvain_communities(G, communities):
    # Create a color map
    colors = [communities[node] for node in G.nodes()]

    # Draw the graph
    nx.draw(G, node_color=colors, with_labels=True)

    # Create a custom legend
    unique_communities = set(communities.values())
    legend_handles = [mpatches.Patch(color=plt.cm.viridis(i / len(unique_communities)), label=f'Community {i+1}') for i in unique_communities]
    plt.legend(handles=legend_handles)

    plt.title("Network with Louvain Communities")
    plt.show()


def draw_network_with_girvan_newman_communities(G, communities, num_levels=None):
    # Identify communities using the Girvan-Newman method
    communities = community.girvan_newman(G)

    if num_levels is None:
        num_levels = 3

    level_communities = []
    for _ in range(num_levels):
        level_communities.append(next(communities))

    last_level_communities = sorted(map(sorted, level_communities[-1]))

    # print("last_level_communities:", last_level_communities)

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

    plt.title("Network with Girvan-newman Communities")
    plt.show()


def construct_network(file_path, test_data, node_range=None, sample_size=None):
  
    # Check if the file exists
    if os.path.exists(file_path):
        # If the file exists, load the data from the file
        G = nx.read_edgelist(file_path, nodetype=int)
    else:
        # If the file does not exist, use the test data
        G = nx.Graph()
        print("File does not exist. Using test data.")
        G.add_edges_from(test_data)
        

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
         print("No Sample size selected. Using the original graph.")
    
    else:
        # If the graph has less or equal nodes than the desired sample size, return the original graph
        print("Sample size too large, returning the original graph.")


    return G


def select_community_detection(method="palla"):

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





# # Node range
# node_range = (45, 70)

# sample_size = 5

# file_path = './content/facebook_combined.txt'


# # test data
# data = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (6, 7), (7, 8), (8, 6), (9, 10)]

# # Construct the network
# G = construct_network(file_path, data, node_range, sample_size)

# # Draw a subset of the large network
# draw_large_network(G)


# # Identify communities
# communities = identify_communities(G, k=3)

# # Analyze the community structure
# analysis = analyze_communities(communities)

# print("Analysis:", analysis)

# # Draw the network with identified communities
# draw_network_with_communities(G, communities)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tool to analyze community structure in networks")
    parser.add_argument("file_path", type=str, help="Path to the file containing the network data")
    parser.add_argument("--node_range", type=int, nargs=2, help="Range of nodes to consider")
    parser.add_argument("--sample_size", type=int, help="Desired sample size of the network")
    parser.add_argument("--overlapping", action="store_true", default=False, help="Flag to indicate whether to allow overlapping communities")
    parser.add_argument("--k", type=int, default=None, help="Size of the k-cliques to use for identifying communities")
    parser.add_argument("--method", type=str, default="palla", help="Method to use for community detection (palla, louvain or girvan_newman)")
    parser.add_argument("--num_levels", type=int, default=3, help="Number of levels for the Girvan-Newman method")
    args = parser.parse_args()

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
    
    # data = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (6, 7), (7, 8), (8, 6), (9, 10)]

    
    # Construct the network
    G = construct_network(args.file_path, data, args.node_range, args.sample_size)

    # Draw a subset of the large network
    draw_large_network(G)

    # Description of the graph
    description = description_of_graph(G)
    print("Description:", description)

    print("node range:", args.node_range)
    print("sample size:", args.sample_size)


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

    # nums_levels = args.num_levels - 1

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

    print("Analysis:", analysis)


    try:
        if args.method == "palla":
            draw_network_with_palla_communities(G, communities)
        elif args.method == "louvain":
            draw_network_with_louvain_communities(G, communities)
        elif args.method == "girvan_newman":
            draw_network_with_girvan_newman_communities(G, communities, args.num_levels)
    except Exception as e:
        print(f"An error occurred: {e}")



    

    






# Run the script with the following command:
    
# python pallas.py ./content/facebook_combined.txt --node_range 45 70 --sample_size 5 --k 3 --overlapping
    
# python pallas.py ./content/tred.txt --node_range 45 70 --sample_size 5 --k 3 --overlapping
    
    
# The script will analyze the community structure in the network and visualize the identified communities.
# The command-line arguments are used to specify the path to the file containing the network data, the range of nodes to consider, and the desired sample size of the network.
    
# The script will also use test data if the file does not exist.
# The test data is a list of edges that will be used to construct the network if the file does not exist.
# The script will visualize a subset of the large network and identify communities using the Clique Percolation Method.
# The identified communities will be analyzed, and the community structure will be visualized.
    


