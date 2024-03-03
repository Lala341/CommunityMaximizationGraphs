import time
import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import eigh
import community
import matplotlib.pyplot as plt
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as si
from ndlib.models.epidemics import ThresholdModel, IndependentCascadesModel
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
import os
import glob
from sklearn import metrics
from scipy.sparse import lil_matrix
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import seaborn as sns
import random


random.seed(42)


def load_graph(file_path='twitch/ENGB/musae_ENGB_edges.csv'):

    if os.path.exists(file_path) and file_path.endswith('.txt'):
        # If the file exists, load the data from the file
        G = nx.read_edgelist(file_path, nodetype=int)

    elif os.path.exists(file_path) and file_path.endswith('.csv'):
        edge_data = pd.read_csv(file_path)
        G = nx.Graph()
        for _, row in edge_data.iterrows():
            G.add_edge(row['from'], row['to'])

    # Print the number of nodes and edges in the graph
    print("Graph Loaded")
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())
    return G


# Helper function of seedset visualization
def visualize_seedsets(graph, blue_seed_set, red_seed_set, node_size=30):
    '''
    :param graph:
    :param blue_seed_set: the list of nodes
    :param red_seed_set: the list of nodes
    :param node_size: size of the nodes
    '''
        
    # Find the intersection of the lists
    inter_set = set(blue_seed_set).intersection(set(red_seed_set))
    
    print("Blue seed set: ", blue_seed_set)
    print("Red seed set: ", red_seed_set)
    print("Intersection set: ", inter_set)
    
    
    pos = nx.spring_layout(graph)
    plt.figure()
    nx.draw_networkx_edges(graph, pos, alpha=0.2)
    nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color='k', alpha=0.3)
    nx.draw_networkx_nodes(graph, pos, nodelist=red_seed_set, node_color='r', alpha=0.5, 
                                node_size=node_size)
    nx.draw_networkx_nodes(graph, pos, nodelist=blue_seed_set, node_color='b', alpha=0.5, 
                                node_size=node_size)
    nx.draw_networkx_nodes(graph, pos, nodelist=inter_set, node_color='g', alpha=1, 
                                node_size=node_size)
    

    plt.axis('off')
    plt.show()
    plt.savefig(save_path +'figure_seeds.png')


def get_connected_graph(G):
    between = nx.betweenness_centrality(G)
    new_nodes = {k:float(v) for (k,v) in between.items() if float(v) >= 0.00015 }

    subgraph = G.subgraph(new_nodes.keys())

    print("Edges", subgraph.number_of_edges())
    print("Nodes", subgraph.number_of_nodes())
    G_connected = subgraph
    G_connected.nodes(data=True)

def get_disconnected_graph(G):
    between = nx.betweenness_centrality(G)
    between_disconnected = between
    disconnected_nodes = {k:float(v) for (k,v) in between_disconnected.items() if float(v) <= 0.0001 }

    disconnected_graph = G.subgraph(disconnected_nodes.keys())
    print("Edges", disconnected_graph.number_of_edges())
    print("Nodes", disconnected_graph.number_of_nodes())
    G_disconnected = disconnected_graph
    G_disconnected.nodes(data=True)



# Simulation of SIR Model
def SIR(graph, beta, gamma, seed_set):
    """
    The model performing SIR simulation
    """
    # Model selection
    model = si.SIRModel(graph)
    config = mc.Configuration()
    
    # Model configuration
    config.add_model_parameter('beta', beta)
    config.add_model_parameter('gamma', gamma)
    config.add_model_initial_configuration("Infected", seed_set)
    
    #---------- Run the simulation
    model.set_initial_status(config)
    return model
   



def independent_cascade(graph, threshold, seed_set):
    """
    The model performing independent cascade simulation
    """
    # Model selection
    model = IndependentCascadesModel(graph)
    
    # Model configuration
    config = mc.Configuration()
    ## Set edge parameters
    for edge in G.edges():
        config.add_edge_configuration("threshold", edge, threshold)        
    ## Set the initial infected nodes
    config.add_model_initial_configuration("Infected", seed_set)
    
    # Set the all configuations
    model.set_initial_status(config)
    return model
    


def linear_threshold(graph, threshold, seed_set):
    # Model selection
    model = ThresholdModel(graph)
    
    # Model configuration
    config = mc.Configuration()
    ## Set edge parameters
    for edge in G.edges():
        config.add_edge_configuration("threshold", edge, threshold)        
    ## Set the initial infected nodes
    config.add_model_initial_configuration("Infected", seed_set)
    
    # Set the all configuations
    model.set_initial_status(config)
    return model


def comparison_models(sir_iterations, ic_iterations,lt_iterations  ):
    sir_infected_count = [iteration["node_count"][1]+iteration["node_count"][2] for iteration in sir_iterations]
    ic_infected_count = [iteration["node_count"][1]+iteration["node_count"][2] for iteration in ic_iterations]
    lt_infected_count = [iteration["node_count"][1] for iteration in lt_iterations]

    plt.xlabel("Number of iterations")
    plt.ylabel("Number of infected nodes")
    line1, = plt.plot(sir_infected_count, label="SIR")
    line2, = plt.plot(ic_infected_count, label="Independent Cascade")
    line3, = plt.plot(lt_infected_count, label="Linear Threshold")
    plt.legend(handles=[line1, line2, line3])
    plt.show()
    plt.savefig(save_path +'figure_comparison.png')



def calculate_metrics(G, number_steps):
    
    # Number of nodes in the seed set
    seed_set_size = 10

    ## Construct the seed sets
    # Degree centrality
    degree = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    degree_seed = [node for node, value in degree[0:seed_set_size]]

    # Pagerank
    pagerank = sorted(nx.pagerank(G).items(), key=lambda x: x[1], reverse=True)
    pagerank_seed = [node for node, value in pagerank[0:seed_set_size]]

    #HITS
    hub, authority = nx.hits(G)
    hub = sorted(hub.items(), key=lambda x: x[1],reverse=True)
    hub_seed = [node for node, value in hub[0:seed_set_size]]


    authority = sorted(authority.items(), key=lambda x: x[1] ,reverse=True)
    authority_seed = [node for node, value in authority[0:seed_set_size]]

    # k-core
    kcore = sorted(nx.core_number(G).items(), key=lambda x: x[1], reverse=True)
    kcore_seed = [node for node, value in kcore[0:seed_set_size]]


    # neighborhood coreness
    node2nbcore =  {node: np.sum([G.degree(nb) for nb in G.neighbors(node)]) for node in G.nodes() } 
    nbkcore = sorted(node2nbcore.items(), key=lambda x: x[1], reverse=True)
    nbkcore_seed = [node for node, value in nbkcore[0:seed_set_size]]

    # ViralRank

    # Assuming G is your graph

    #katz_centralities = nx.katz_centrality(G, alpha=0.1, max_iter=2000)
    #katz = sorted(katz_centralities.items(), key=lambda x: x[1], reverse=True)
    #katz_seed = [node for node, value in degree[0:seed_set_size]]


    def construct_model(model_name, threshold, beta, gamma):
        if model_name == "SIR":
            model_func = SIR
            model_params = dict(graph=G, 
                        beta=beta, 
                        gamma=gamma)
        elif model_name == "IC":
            model_func = independent_cascade
            model_params = dict(graph=G, 
                        threshold=threshold)
        elif model_name == "LT":
            model_func = linear_threshold
            model_params = dict(graph=G, 
                        threshold=threshold)
        
        model_params["seed_set"] = degree_seed
        degree_model = model_func(**model_params)
        degree_iters = degree_model.iteration_bunch(num_steps)
        
        model_params["seed_set"] = pagerank_seed
        pagerank_model = model_func(**model_params)
        pagerank_iters = pagerank_model.iteration_bunch(num_steps)
        
        model_params["seed_set"] = kcore_seed
        kcores_model = model_func(**model_params)
        kcore_iters = kcores_model.iteration_bunch(num_steps)
        
        model_params["seed_set"] = nbkcore_seed
        nbkcores_model = model_func(**model_params)
        nbkcore_iters = nbkcores_model.iteration_bunch(num_steps)


        #model_params["seed_set"] = katz_seed
        #katz_model = model_func(**model_params)
        #katz_iters = katz_model.iteration_bunch(num_steps)

        #model_params["seed_set"] = viralrank_seed
        #viralrank_model = model_func(**model_params)
        #viralrank_iters = viralrank_model.iteration_bunch(num_steps)

        model_params["seed_set"] = authority_seed
        authority_model = model_func(**model_params)
        authority_iters = authority_model.iteration_bunch(num_steps)
        
        model_params["seed_set"] = hub_seed
        hub_model = model_func(**model_params)
        hub_iters = hub_model.iteration_bunch(num_steps)  
        
        return degree_iters, pagerank_iters, kcore_iters, nbkcore_iters, hub_iters, authority_iters#, katz_iters#viralrank_iters, 


    model_name = "SIR"  # SIR, LT or IC
    threshold = 0.5 # ex: 0.1 for LT, 0.5 for IC model
    beta = 0.1 # for SIR
    gamma = 0.1 # for SIR
    degree_iters, pagerank_iters, kcore_iters, nbkcore_iters,  hub_iters, authority_iters = construct_model(model_name, threshold, beta, gamma) # viralrank_iters,

    #                         [np.sum([iteration["node_count"][1]+iteration["node_count"][2]]) for iteration in sir_iterations]
    #----------- Plot them
    degree_infected_count = [np.sum([iteration["node_count"].get(inx, 0) for inx in [1, 2]]) for iteration in degree_iters]
    pagerank_infected_count = [np.sum([iteration["node_count"].get(inx, 0) for inx in [1, 2]]) for iteration in pagerank_iters]
    kcore_infected_count = [np.sum([iteration["node_count"].get(inx, 0) for inx in [1, 2]]) for iteration in kcore_iters]
    nbkcore_infected_count = [np.sum([iteration["node_count"].get(inx, 0) for inx in [1, 2]]) for iteration in nbkcore_iters]
    #katz_infected_count = [np.sum([iteration["node_count"].get(inx, 0) for inx in [1, 2]]) for iteration in katz_iters]
    hub_infected_count = [np.sum([iteration["node_count"].get(inx, 0) for inx in [1, 2]]) for iteration in hub_iters]
    authority_infected_count = [np.sum([iteration["node_count"].get(inx, 0) for inx in [1, 2]]) for iteration in authority_iters]

    plt.figure()
    line1, = plt.plot(range(num_steps), degree_infected_count, label="Degree Centrality")
    line2, = plt.plot(range(num_steps), pagerank_infected_count, label="Pagerank")
    line3, = plt.plot(range(num_steps), kcore_infected_count, label="k-core")
    line4, = plt.plot(range(num_steps), nbkcore_infected_count, label="nb-coreness")
    #line5, = plt.plot(range(num_steps), katz_infected_count, label="Katz")
    line6, = plt.plot(range(num_steps), hub_infected_count, label="HITS")
    line7, = plt.plot(range(num_steps), authority_infected_count, label="Authority")
    plt.legend(handles=[line1, line2, line3, line4,  line6, line7])#line5,
    plt.ylabel("Number of infected nodes")
    plt.xlabel("Number of iterations")
    plt.show()

    plt.savefig(save_path +'figure_comparison_metrics.png')


def influence_maximization(G, num_steps):
    from igraph import Graph
    g = Graph([[u,v] for (u,v) in G.edges])

    def simulate_ic_model(graph, ic_threshold=0.1, ic_num_steps=50, ic_seed_set=None):
    
        if ic_seed_set is None:
            raise ValueError("Please set the seed set!")
        
        # Run the model
        ic_model = independent_cascade(graph=graph, threshold=ic_threshold, seed_set=ic_seed_set)
        ic_iterations = ic_model.iteration_bunch(ic_num_steps)

        # Get the number of infected nodes in the last step
        return int(ic_iterations[-1]["node_count"][2])

    def IC(g,S,ic_threshold,ic_num_steps):
        """
        Input:  graph object, set of seed nodes, propagation probability
                and the number of Monte-Carlo simulations
        Output: average number of nodes influenced by the seed nodes
        """
        
        # Loop over the Monte-Carlo Simulations
        spread = []
        for i in range(ic_num_steps):
            
            # Simulate propagation process      
            new_active, A = S[:], S[:]
            while new_active:

                # For each newly active node, find its neighbors that become activated
                new_ones = []
                for node in new_active:
                    
                    # Determine neighbors that become infected
                    np.random.seed(i)
                    success = np.random.uniform(0,1,len(g.neighbors(node,mode="out"))) < ic_threshold
                    new_ones += list(np.extract(success, g.neighbors(node,mode="out")))

                new_active = list(set(new_ones) - set(A))
                
                # Add newly activated nodes to the set of activated nodes
                A += new_active
                
            spread.append(len(A))
            
        return(np.mean(spread))
    def greedy(g,k,p=0.1,mc=1000):
        """
        Input:  graph object, number of seed nodes
        Output: optimal seed set, resulting spread, time for each iteration
        """

        S, spread, timelapse, start_time = [], [], [], time.time()
        
        # Find k nodes with largest marginal gain
        for _ in range(k):

            # Loop over nodes that are not yet in seed set to find biggest marginal gain
            best_spread = 0
            for j in set(range(g.vcount()))-set(S):

                # Get the spread
                s = IC(g,S + [j],p,mc)

                # Update the winning node and spread so far
                if s > best_spread:
                    best_spread, node = s, j

            # Add the selected node to the seed set
            S.append(node)
            
            # Add estimated spread and elapsed time
            spread.append(best_spread)
            timelapse.append(time.time() - start_time)

        return(S,spread,timelapse)
    def celf(g,k,p=0.1,mc=1000):  
        """
        Input:  graph object, number of seed nodes
        Output: optimal seed set, resulting spread, time for each iteration
        """
        
        # --------------------
        # Find the first node with greedy algorithm
        # --------------------
        
        # Calculate the first iteration sorted list
        start_time = time.time() 
        marg_gain = [IC(g,[node],p,mc) for node in range(g.vcount())]

        # Create the sorted list of nodes and their marginal gain 
        Q = sorted(zip(range(g.vcount()),marg_gain), key=lambda x: x[1],reverse=True)

        # Select the first node and remove from candidate list
        S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
        Q, LOOKUPS, timelapse = Q[1:], [g.vcount()], [time.time()-start_time]
        
        # --------------------
        # Find the next k-1 nodes using the list-sorting procedure
        # --------------------
        
        for _ in range(k-1):    

            check, node_lookup = False, 0
            
            while not check:
                
                # Count the number of times the spread is computed
                node_lookup += 1
                
                # Recalculate spread of top node
                current = Q[0][0]
                
                # Evaluate the spread function and store the marginal gain in the list
                Q[0] = (current,IC(g,S+[current],p,mc) - spread)

                # Re-sort the list
                Q = sorted(Q, key = lambda x: x[1], reverse = True)

                # Check if previous top node stayed on top after the sort
                check = (Q[0][0] == current)

            # Select the next node
            spread += Q[0][1]
            S.append(Q[0][0])
            SPREAD.append(spread)
            LOOKUPS.append(node_lookup)
            timelapse.append(time.time() - start_time)

            # Remove the selected node from the list
            Q = Q[1:]

        return(S,SPREAD,timelapse,LOOKUPS)


    # obtain seedset
    greedy_seed = greedy(g, 5, p = 0.5,mc = 2)
    celf_seed = celf(g, 5, p = 0.5,mc = 2)

    plt.figure()

    # Plot settings
    plt.rcParams['figure.figsize'] = (9,6)
    plt.rcParams['lines.linewidth'] = 4
    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['ytick.left'] = False

    # Plot Computation Time
    plt.plot(range(1,len(greedy_seed[2])+1),greedy_seed[2],label="Greedy",color="#FBB4AE")
    plt.plot(range(1,len(celf_seed[2])+1),celf_seed[2],label="CELF",color="#B3CDE3")
    plt.ylabel('Computation Time (Seconds)'); plt.xlabel('Size of Seed Set')
    plt.title('Computation Time'); plt.legend(loc=2)
    plt.savefig(save_path +'figure_computationalTime.png')

    celf_seed_set = list(celf(g, 5, 0.5, 5)[0])
    greedy_seed_set = list(greedy(g, 5, 0.5, 5)[0])

    blue_seed_set = celf_seed_set
    red_seed_set = greedy_seed_set 
    
    visualize_seedsets(graph=G, blue_seed_set=blue_seed_set, red_seed_set=red_seed_set, node_size=30)


    def construct_model_IM(model_name, threshold, beta, gamma):
        if model_name == "SIR":
            model_func = SIR
            model_params = dict(graph=G, 
                        beta=beta, 
                        gamma=gamma)
        elif model_name == "IC":
            model_func = independent_cascade
            model_params = dict(graph=G, 
                        threshold=threshold)
        elif model_name == "LT":
            model_func = linear_threshold
            model_params = dict(graph=G, 
                        threshold=threshold)
        
        model_params["seed_set"] = greedy_seed_set
        greedy_model = model_func(**model_params)
        greedy_iters = greedy_model.iteration_bunch(num_steps)
        
        model_params["seed_set"] = celf_seed_set
        celf_model = model_func(**model_params)
        celf_iters = celf_model.iteration_bunch(num_steps)  
        
        return greedy_iters, celf_iters

    model_name = "IC"  # SIR, LT or IC
    threshold = 0.5 # ex: 0.1 for LT, 0.5 for IC model
    beta = 0.1 # for SIR
    gamma = 0.1 # for SIR
    greedy_iters, celf_iters = construct_model_IM(model_name, threshold, beta, gamma)

    #----------- Plot them
    greedy_infected_count = [np.sum([iteration["node_count"].get(inx, 0) for inx in [1, 2]]) for iteration in greedy_iters]
    celf_infected_count = [np.sum([iteration["node_count"].get(inx, 0) for inx in [1, 2]]) for iteration in celf_iters]

    plt.figure()
    line1, = plt.plot(range(num_steps), greedy_infected_count, label="Greedy")
    line2, = plt.plot(range(num_steps), celf_infected_count, label="CELF")

    plt.legend(handles=[line1, line2])
    plt.ylabel("Number of infected nodes")
    plt.xlabel("Number of iterations")
    plt.show()
    plt.savefig(save_path +'figure_c42.png')


    



def execute_models_comparison(G):
    seed_size=10
    #  Execute SIR
    random_seed=np.random.choice(G.nodes(), seed_size)

    # Number of steps/iterations of the epidemic progression
    sir_num_steps = 50
    # Number of nodes in the seed set
    sir_seed_set_size = 10
    # Determine the seed set
    sir_seed_set = random_seed

    # Determine the model parameters
    sir_gamma = 0.1
    eigval, eigvec = eigh(nx.adjacency_matrix(G).toarray())
    sir_beta = 0.2+1.0/float(eigval[-1])

    # Run the model
    sir_model = SIR(G, sir_beta, sir_gamma, sir_seed_set)
    sir_iterations = sir_model.iteration_bunch(bunch_size=sir_num_steps)


    # Get the number of susceptible(0), infected(1) and the recovered(2) nodes in the last step
    print(sir_iterations[-1]["node_count"])

    print("sir_beta: ",sir_beta )
    # Plot the progression of the number of susceptible, infected and the recovered nodes
    sir_trends = sir_model.build_trends(sir_iterations)
    plt.figure()
    viz = DiffusionTrend(sir_model, sir_trends)
    viz.plot()

    plt.savefig(save_path +'figure_SIC.png')



    #  Execute IC

    # Number of steps/iterations
    ic_num_steps = 50
    # Number of nodes in the seed set
    ic_seed_set_size = 10
    # Determine the seed set
    ic_seed_set = random_seed
    # Determine the model parameter
    ic_threshold = 0.5


    # Run the model
    ic_model = independent_cascade(graph=G, threshold=ic_threshold, seed_set=ic_seed_set)
    ic_iterations = ic_model.iteration_bunch(ic_num_steps)


    # Get the number of susceptible, infected and the recovered nodes 
    # in the last step
    print(ic_iterations[-1]["node_count"])


    #_Plot the progression of the number of susceptible, infected and 
    # the recovered nodes 
    ic_trends = ic_model.build_trends(ic_iterations)
    plt.figure()
    viz = DiffusionTrend(ic_model, ic_trends)
    viz.plot()
    plt.savefig(save_path +'figure_IC.png')


    #  Execute linear_threshold
    # Number of steps/iterations
    lt_num_steps = 50
    # Number of nodes in the seed set
    lt_seed_set_size = 10
    # Determine the seed set
    lt_seed_set = random_seed
    # Determine the model parameter
    lt_threshold = 0.1


    # Run the model
    lt_model = linear_threshold(graph=G, threshold=lt_threshold, seed_set=lt_seed_set)
    lt_iterations = lt_model.iteration_bunch(lt_num_steps)


    # Get the number of susceptible, infected and the recovered nodes 
    # in the last step
    print(lt_iterations[-1]["node_count"])


    #_Plot the progression of the number of susceptible, infected and 
    # the recovered nodes 
    lt_trends = lt_model.build_trends(lt_iterations)
    plt.figure()
    viz = DiffusionTrend(lt_model, lt_trends)
    viz.plot()
    plt.savefig(save_path +'figure_LT.png')

    return sir_iterations, ic_iterations,lt_iterations




















# Put Folder name based on dataset to save the results
folder_name = 'results'

# Directory to save the embeddings and clustering results
save_path = './' + folder_name + "/maximization/"




#create a folder to save the clustering results
def create_folder():
  if not os.path.exists(save_path):
    os.makedirs(save_path)
    print("Folder created")

    return


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

    plt.savefig(os.path.join(save_path + "node_network.png"))

    # plt.show()
    plt.close()


def embeddings(G):
    import numpy as np
    import networkx as nx
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import networkx as nx
    from gensim.models import Word2Vec
    import numpy as np

    # Generate or load your NetworkX graph
    # G = nx.erdos_renyi_graph(n=100, p=0.1)
    node_sequences = [list(map(str, nx.single_source_shortest_path_length(G, node))) for node in G.nodes()]

    # Train Word2Vec model to learn embeddings
    embedding_size = 16
    window_size = 5
    model = Word2Vec(node_sequences, vector_size=embedding_size, window=window_size, min_count=1, sg=1, workers=4)

    # Obtain embeddings for all nodes
    node_embeddings = {str(node): model.wv[str(node)] for node in G.nodes()}

    # Function to extract features from node embeddings
    def extract_features(embeddings):
        # In this example, we use degree centrality and node embeddings as features
        features = []
        for node, embed in embeddings.items():
            degree = G.degree[int(node)]
            feature = list(embed) + [degree]
            features.append(feature)
        return np.array(features)

    # Generate or load your NetworkX graph
    # G = nx.erdos_renyi_graph(n=100, p=0.1)

    # Train a machine learning model to predict node influence
    def train_influence_model(G, embeddings, test_size=0.2):
        # Calculate degree centrality
        degree_centrality = nx.degree_centrality(G)
        
        # Extract features from node embeddings and degree centrality
        features = extract_features(embeddings)
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(features, list(degree_centrality.values()), test_size=test_size, random_state=42)
        
        # Train a RandomForestRegressor model
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        
        # Evaluate the model on the validation set
        val_predictions = model.predict(X_val)
        val_r_squared = model.score(X_val, y_val)
        val_mae = mean_absolute_error(y_val, val_predictions)
        val_mse = mean_squared_error(y_val, val_predictions)
        val_rmse = np.sqrt(val_mse)
        
        print("Validation R^2 Score:", val_r_squared)
        print("Validation MAE:", val_mae)
        print("Validation MSE:", val_mse)
        print("Validation RMSE:", val_rmse)
        
        return model

    # Example usage:
    # Train influence model
    influence_model = train_influence_model(G, node_embeddings)

    # Function to select seed nodes based on node influence predictions
    def select_seed_nodes_influence(model, embeddings, num_seeds):
        # Extract features from node embeddings
        features = extract_features(embeddings)
        
        # Predict node influence using the trained model
        influence_predictions = model.predict(features)
        
        # Select top-ranked nodes as seed nodes
        sorted_nodes = [int(node) for _, node in sorted(zip(influence_predictions, embeddings.keys()), reverse=True)]
        return sorted_nodes[:num_seeds]

    # Example usage:
    ic_seed_set_size = 10  # Number of seed nodes to select

    # Select seed nodes based on node influence predictions
    seed_nodes = select_seed_nodes_influence(influence_model, node_embeddings, ic_seed_set_size)

    print("Selected Seed Nodes:", seed_nodes)


    # Number of steps/iterations
    ic_num_steps = 50
    # Determine the seed set
    ic_seed_set = seed_nodes
    # Determine the model parameter
    ic_threshold = 0.5


    # Run the model
    ic_model = independent_cascade(graph=G, threshold=ic_threshold, seed_set=ic_seed_set)
    ic_iterations = ic_model.iteration_bunch(ic_num_steps)


    # Get the number of susceptible, infected and the recovered nodes 
    # in the last step
    print(ic_iterations[-1]["node_count"])

    #_Plot the progression of the number of susceptible, infected and 
    # the recovered nodes 
    ic_trends = ic_model.build_trends(ic_iterations)
    plt.figure()
    viz = DiffusionTrend(ic_model, ic_trends)
    viz.plot()
    plt.savefig(save_path+'figure_IC_ML.png')




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tool to analyze influence maximization in networks")
    parser.add_argument("file_path", type=str, help="Path to the file containing the network data")
    parser.add_argument("--sample_size", type=int, help="Desired sample size of the network")
    parser.add_argument("--overlapping", action="store_true", default=False, help="Flag to indicate whether to allow overlapping communities")
    parser.add_argument("--k", type=int, default=None, help="Size of the k-cliques to use for identifying communities")
    parser.add_argument("--method", type=str, default="IC", help="Method to use for community detection (palla, louvain or girvan_newman)")
    args = parser.parse_args()

    

    create_folder()

    G=load_graph(file_path=args.file_path)

    # Draw a subset of the large network
    print("Draw")
    #draw_large_network(G)
    print("Execute models")
    sir_iterations, ic_iterations,lt_iterations= execute_models_comparison(G)
    print("Execute comparison")
    comparison_models(sir_iterations, ic_iterations,lt_iterations  )
    print("Execute metrics")
    # Number of steps/iterations
    num_steps = 10
    calculate_metrics(G, num_steps)

    if args.method == "IC" or args.method == "TM" or args.method == "CM":
        print("Influence maximization method:", args.method)
    elif args.method is None:
        print("Influence maximization method: IC(default)") 


    print("Execute influence maximization with selected model")
    #influence_maximization(G, num_steps)
    print("Execute ML model")
    embeddings(G)

    # Description of the graph
    #description = description_of_graph(G)
    #print("Description:", description)

    #print("node range:", args.node_range)
    #print("sample size:", args.sample_size)

    

        
    #if args.method == "palla":
    #    print("Size of k-cliques:", args.k if args.k is not None else "default (3)")
    #    print("Allow overlapping communities:", args.overlapping)

    #print("Analysis:", analysis)





    

    






# Run the script with the following command:
    
# python pallas.py ./content/facebook_combined.txt --node_range 45 70 --sample_size 5 --k 3 --overlapping
    
# python pallas.py ./content/tred.txt --node_range 45 70 --sample_size 5 --k 3 --overlapping
    
    
# The script will analyze the community structure in the network and visualize the identified communities.
# The command-line arguments are used to specify the path to the file containing the network data, the range of nodes to consider, and the desired sample size of the network.
    
# The script will also use test data if the file does not exist.
# The test data is a list of edges that will be used to construct the network if the file does not exist.
# The script will visualize a subset of the large network and identify communities using the Clique Percolation Method.
# The identified communities will be analyzed, and the community structure will be visualized.
    


