import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
import numpy as np
from matplotlib.cm import ScalarMappable
import zipfile
import os


zip_file_path = 'youtube_data.zip'

# # Create paths
# for i in range(17, 28, 2):
#     for j in range(4):
#         globals()[f"txt_{i}_{j}"] = f"{base_path}{i}_{j}.txt"

# # Load data from zip file
# Load data from zip file
dataframes = []
def extract_and_load_data(zip_path):
    global dataframes
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # List all files in the zip archive
        file_names = zip_ref.namelist()
        print(f"Files in the zip archive: {file_names}")

        # Read and load each text file into a DataFrame
        for file_name in file_names:
            if file_name.endswith('.txt'):
                with zip_ref.open(file_name) as file:
                    try:
                        df = pd.read_csv(file, sep='\t', header=None)
                        if len(df.columns) in [9, 29]:  # Expecting either 9 or 29 columns
                            dataframes.append(df)
                        else:
                            print(f"Ignoring file '{file_name}': Expected 9 or 29 columns, found {len(df.columns)} columns")
                    except pd.errors.ParserError as e:
                        print(f"Error parsing '{file_name}': {e}")

# Call the function to load data
extract_and_load_data(zip_file_path)

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

print_combined = print(len(combined_df))
# display(combined_df.head())

#Null values
null_values = combined_df.isnull().sum()


# Display null values
print("Null Values in Each Column:")
print(null_values)

# Numerical and non-numerical meta information columns
numerical_meta_columns = combined_df.columns[[2, 4, 5, 6, 7, 8]]  #  numerical columns
non_numerical_meta_columns = combined_df.columns[[1, 3]]  # non-numerical columns

# Remove rows where all meta information columns are null
meta_columns = combined_df.columns[1:9]
cleaned_df = combined_df.dropna(subset=meta_columns, how='all')

# Fill null values in numerical meta information columns with the mean
cleaned_df[numerical_meta_columns] = cleaned_df[numerical_meta_columns].fillna(cleaned_df[numerical_meta_columns].mean())

# Fill null values in non-numerical meta information columns with 'unknown'
cleaned_df[non_numerical_meta_columns] = cleaned_df[non_numerical_meta_columns].fillna('unknown')

# Fill null values in related video ID columns with 'unknown'
related_video_columns = combined_df.columns[9:]
cleaned_df[related_video_columns] = cleaned_df[related_video_columns].fillna('unknown')

# Verify the cleaning process
print("Null Values after Cleaning:")
print(cleaned_df.isnull().sum())
print(len(cleaned_df))

# Step 1: Construct Undirected Graph with all sources-edges
def construct_graph(data):
    G = nx.Graph()
    for index, row in data.iterrows():
        video_id = row[0]
        related_videos = row[9:]  # Assuming related videos start from column index 9
        for related_video in related_videos:
            if related_video != 'unknown':
                G.add_edge(video_id, related_video)
    return G

# Step 2: Pruning the graph
def prune_graph(G, meta_info):
    G_pruned = G.subgraph(meta_info).copy()  # Create a copy to allow modifications
    return G_pruned

# Step 3: Remove nodes with no edges
def remove_isolated_nodes(G):
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    return G

# Construct the graph
G = construct_graph(cleaned_df)
print("Step 1: Original Graph - Nodes:", len(G.nodes()), "Edges:", len(G.edges()))

# Step 2: Pruning the graph based on meta information (columns 1 to 8)
meta_info = cleaned_df.iloc[:, 0].tolist()  # Assuming the first column contains video IDs
G_pruned = prune_graph(G, meta_info)
print("Step 2: Subgraph Meta info Graph - Nodes:", len(G_pruned.nodes()), "Edges:", len(G_pruned.edges()))

# Step 3: Remove nodes with no edges
G_pruned = remove_isolated_nodes(G_pruned)
print("Step 3: Pruned Graph after removing isolated nodes - Nodes:", len(G_pruned.nodes()), "Edges:", len(G_pruned.edges()))

"""##Degree Distribution##"""

# Histogram for degree distribution
degree_freq = nx.degree_histogram(G_pruned)
degrees = range(len(degree_freq))
plt.figure(figsize=(12, 8))
plt.loglog(degrees[3:], degree_freq[3:],'go-')
plt.xlabel('Degree')
plt.ylabel('Frequency')
# Add horizontal red line at x=20
plt.axvline(x=20, color='red', linestyle='--')
plt.grid(True)
plt.savefig("Degree_Distribution.jpg")
plt.show()

"""##Centrality##"""

# Function to estimate betweenness centrality
def estimate_betweenness_centrality(G, num_samples=100):
    betweenness_centrality = {node: 0 for node in G.nodes()}

    for _ in range(num_samples):
        # Randomly select a node as the source
        source = random.choice(list(G.nodes()))

        # Use networkx's single-source shortest path algorithm to calculate shortest paths from the source
        shortest_paths = nx.single_source_shortest_path(G, source)

        # Increment betweenness centrality for nodes that appear in the shortest paths
        for paths in shortest_paths.values():
            for node in paths:
                if node != source:  # Exclude the source node itself
                    betweenness_centrality[node] += 1

    # Normalize betweenness_centrality by dividing by the number of samples
    for node in G.nodes():
        betweenness_centrality[node] /= num_samples

    return betweenness_centrality

# Example
betweenness_centrality = estimate_betweenness_centrality(G_pruned)

# Create numeric labels for nodes
numeric_labels = {node: i for i, node in enumerate(G_pruned.nodes())}

# Plot the graph with node colors based on betweenness centrality
pos = nx.spring_layout(G_pruned)  # Layout for the graph visualization
node_color = [betweenness_centrality[node] for node in G_pruned.nodes()]  # Node colors based on betweenness centrality

plt.figure(figsize=(12, 8))
nx.draw(G_pruned, pos, node_color=node_color, cmap=plt.cm.Blues, with_labels=True, labels=numeric_labels, node_size=100, font_size=8)

# Create a ScalarMappable object for colorbar
sm = ScalarMappable(cmap=plt.cm.Blues)
sm.set_array([])  # Dummy array for ScalarMappable

# Add colorbar
plt.colorbar(sm, label='Betweenness Centrality')

plt.title('Graph with Node Colors based on Betweenness Centrality')
plt.savefig('Bet_Cen.jpg')
plt.show()

# Closeness Centrality
def estimate_closeness_centrality(G, sample_size=100):
    nodes = list(G.nodes())
    sampled_nodes = random.sample(nodes, sample_size)

    closeness_centrality = {node: 0 for node in nodes}

    for node in sampled_nodes:
        shortest_paths = nx.single_source_shortest_path_length(G, node)
        for target, length in shortest_paths.items():
            if length > 0:
                closeness_centrality[target] += 1 / length

    # Normalize by sample size
    for node in nodes:
        closeness_centrality[node] /= sample_size

    return closeness_centrality

# Estimate closeness centrality
closeness_centrality_approx = estimate_closeness_centrality(G_pruned, sample_size=100)

# Create numeric labels for nodes
numeric_labels = {node: i for i, node in enumerate(G_pruned.nodes())}

# Generate positions for nodes
pos = nx.spring_layout(G_pruned)

# Plot the graph with node colors based on closeness centrality
node_color = [closeness_centrality_approx[node] for node in G_pruned.nodes()]

plt.figure(figsize=(12, 8))
nx.draw(G_pruned, pos, node_color=node_color, cmap=plt.cm.Reds, with_labels=True, labels=numeric_labels, node_size=100, font_size=8)

# Create a ScalarMappable object for colorbar
sm = ScalarMappable(cmap=plt.cm.Reds)
sm.set_array([])  # Dummy array for ScalarMappable

# Add colorbar
plt.colorbar(sm, label='Closeness Centrality (Approx)')

plt.title('Graph with Node Colors based on Closeness Centrality (Approx)')
plt.savefig('Cl_cen.jpg')
plt.show()

# Calculate degree centrality
degree_centrality = nx.degree_centrality(G_pruned)

# Create numeric labels for nodes
numeric_labels = {node: i for i, node in enumerate(G_pruned.nodes())}

# Generate positions for nodes
pos = nx.spring_layout(G_pruned)

# Plot the graph with node colors based on degree centrality
node_color = [degree_centrality[node] for node in G_pruned.nodes()]

plt.figure(figsize=(12, 8))
nx.draw(G_pruned, pos, node_color=node_color, cmap=plt.cm.Purples, with_labels=True, labels=numeric_labels, node_size=100, font_size=8)

# Create a ScalarMappable object for colorbar
sm = ScalarMappable(cmap=plt.cm.Purples, norm=plt.Normalize(vmin=min(node_color), vmax=max(node_color)))
sm.set_array([])  # Dummy array for ScalarMappable

# Add colorbar
plt.colorbar(sm, label='Degree Centrality')

plt.title('Graph with Node Colors based on Degree Centrality')
plt.savefig('D_cen.jpg')
plt.show()

"""##Inside the Density using a Sample##"""

# Function to create a dense sample of the graph
def dense_sample_graph(G, sample_size=2000):
    # Get the nodes sorted by degree (high to low)
    high_degree_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)

    # Start with the highest-degree node
    start_node = high_degree_nodes[0][0]

    # Perform BFS to get a dense subgraph
    bfs_nodes = list(nx.bfs_tree(G, start_node).nodes())[:sample_size]
    G_sample = G.subgraph(bfs_nodes).copy()

    return G_sample

# Create a dense sample of the graph
G_sample = dense_sample_graph(G_pruned, sample_size=2000)

# Calculate centrality measures
betweenness_centrality = nx.betweenness_centrality(G_sample)
closeness_centrality = nx.closeness_centrality(G_sample)
degree_centrality = nx.degree_centrality(G_sample)

# Create a mapping from original node labels to numeric labels
node_mapping = {node: idx for idx, node in enumerate(G_sample.nodes())}
G_sample = nx.relabel_nodes(G_sample, node_mapping)
betweenness_centrality = {node_mapping[node]: bc for node, bc in betweenness_centrality.items()}
closeness_centrality = {node_mapping[node]: cc for node, cc in closeness_centrality.items()}
degree_centrality = {node_mapping[node]: dc for node, dc in degree_centrality.items()}

# Function to draw the graph
def draw_graph_centrality(G, pos, centrality, title, cmap):
    plt.figure(figsize=(12, 8))
    nodes = nx.draw_networkx_nodes(G, pos, node_color=list(centrality.values()), cmap=cmap, node_size=500, edgecolors='black')
    edges = nx.draw_networkx_edges(G, pos, alpha=0.3)
    labels = nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")

    sm = ScalarMappable(cmap=cmap)
    sm.set_array([])  # Dummy array for ScalarMappable

    plt.colorbar(sm, label=title)
    plt.title(f"Dense Sampled Graph with Node Colors based on {title}")
    plt.axis('off')
    plt.show()

# Plot the graph with node colors based on betweenness centrality
pos = nx.spring_layout(G_sample)

plt.figure(figsize=(15, 15))
nodes = nx.draw_networkx_nodes(G_sample, pos, node_color=list(betweenness_centrality.values()), cmap=plt.cm.Blues, node_size=500, edgecolors='black')
edges = nx.draw_networkx_edges(G_sample, pos, alpha=0.3)
labels = nx.draw_networkx_labels(G_sample, pos, font_size=10, font_color="black")

sm = ScalarMappable(cmap=plt.cm.Blues)
sm.set_array([])  # Dummy array for ScalarMappable

plt.colorbar(sm, label="Betweenness Centrality")
plt.title("Dense Sampled Graph with Node Colors based on Betweenness Centrality")
plt.axis('off')
plt.savefig('Sample_Bet_Centrality.jpg')
plt.show()

# Plot the graph with node colors based on betweenness centrality
pos = nx.spring_layout(G_sample)

# Plot the graph with node colors based on closeness centrality
plt.figure(figsize=(16, 16))
nodes = nx.draw_networkx_nodes(G_sample, pos, node_color=list(closeness_centrality.values()), cmap=plt.cm.Greens, node_size=500, edgecolors='black')
edges = nx.draw_networkx_edges(G_sample, pos, alpha=0.3)
labels = nx.draw_networkx_labels(G_sample, pos, font_size=10, font_color="black")

sm = ScalarMappable(cmap=plt.cm.Greens)
sm.set_array([])  # Dummy array for ScalarMappable

plt.colorbar(sm, label="Closeness Centrality")
plt.title("Dense Sampled Graph with Node Colors based on Closeness Centrality")
plt.axis('off')
plt.savefig('Sample_Clos_Centrality.jpg')
plt.show()

# Plot the graph with node colors based on degree centrality
pos = nx.spring_layout(G_sample)
plt.figure(figsize=(15, 15))
nodes = nx.draw_networkx_nodes(G_sample, pos, node_color=list(degree_centrality.values()), cmap=plt.cm.Reds, node_size=500, edgecolors='black')
edges = nx.draw_networkx_edges(G_sample, pos, alpha=0.3)
labels = nx.draw_networkx_labels(G_sample, pos, font_size=10, font_color="black")

sm = ScalarMappable(cmap=plt.cm.Reds)
sm.set_array([])  # Dummy array for ScalarMappable

plt.colorbar(sm, label="Degree Centrality")
plt.title("Dense Sampled Graph with Node Colors based on Degree Centrality")
plt.axis('off')
plt.savefig('Sample_Deg_Centrality.jpg')
plt.show()

# Calculate centrality measures
degree_centrality = nx.degree_centrality(G_pruned)
betweenness_centrality = nx.betweenness_centrality(G_pruned)
closeness_centrality = nx.closeness_centrality(G_pruned)

# Get the top 10 nodes by each centrality measure
top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
top_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]

# Normalize the values
def normalize_centrality(centrality_list):
    values = [x[1] for x in centrality_list]
    min_val, max_val = min(values), max(values)
    return [(node, (value - min_val) / (max_val - min_val)) for node, value in centrality_list]

normalized_top_degree = normalize_centrality(top_degree)
normalized_top_betweenness = normalize_centrality(top_betweenness)
normalized_top_closeness = normalize_centrality(top_closeness)

# Print the results
print("Top 10 nodes by normalized Degree Centrality:")
for node, centrality in normalized_top_degree:
    print(f"Node: {node}, Normalized Degree Centrality: {centrality:.4f}")

print("\nTop 10 nodes by normalized Betweenness Centrality:")
for node, centrality in normalized_top_betweenness:
    print(f"Node: {node}, Normalized Betweenness Centrality: {centrality:.4f}")

print("\nTop 10 nodes by normalized Closeness Centrality:")
for node, centrality in normalized_top_closeness:
    print(f"Node: {node}, Normalized Closeness Centrality: {centrality:.4f}")

"""##Shortest path distribution (shortest path vs number of node pairs)##"""

# Function to approximate shortest path length distribution
def approximate_shortest_path_distribution(G, num_samples=10000):
    nodes = list(G.nodes())
    path_lengths = []

    for _ in range(num_samples):
        source = random.choice(nodes)
        target = random.choice(nodes)
        if source != target and nx.has_path(G, source, target):
            length = nx.shortest_path_length(G, source=source, target=target)
            path_lengths.append(length)

    return path_lengths

# Approximate shortest path length distribution
num_samples = 10000
path_lengths = approximate_shortest_path_distribution(G_pruned, num_samples)

# Create a histogram of shortest path lengths
plt.figure(figsize=(10, 6))
plt.hist(path_lengths, bins=np.arange(min(path_lengths), max(path_lengths) + 1, 1), alpha=0.7, color='b', edgecolor='black')
plt.title('Shortest Path Length Distribution (Approximation)')
plt.xlabel('Shortest Path Length')
plt.ylabel('Number of Node Pairs')
plt.grid(True)
plt.savefig('Shortest_Path.jpg')
plt.show()

"""##Link Prediction##

## Simple Random Walk vs Pixie Random Walk  using train-test-eval split 80-10-10##
"""

# set seed in order to reproduce results
random.seed(150)

# 80-10-10 split
def split_edges(G, eval_fraction=0.10, test_fraction=0.10):
    G_train = G.copy()
    #list  eval,test links
    eval_edges = []
    test_edges = []

    # debugging edges that cannot be removed
    unable_to_remove_eval = []
    unable_to_remove_test = []

    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) < 2:
            continue

        num_eval = int(len(neighbors) * eval_fraction)
        num_test = int(len(neighbors) * test_fraction)

        random.shuffle(neighbors)

        eval_neighbors = neighbors[:num_eval]
        test_neighbors = neighbors[num_eval:num_eval + num_test]

        for neighbor in eval_neighbors:
            try:
                eval_edges.append((node, neighbor))
                G_train.remove_edge(node, neighbor)
            except nx.NetworkXError:
                unable_to_remove_eval.append((node, neighbor))

        for neighbor in test_neighbors:
            try:
                test_edges.append((node, neighbor))
                G_train.remove_edge(node, neighbor)
            except nx.NetworkXError:
                unable_to_remove_test.append((node, neighbor))

    print("Unable to remove evaluation edges:", unable_to_remove_eval)
    print("Unable to remove test edges:", unable_to_remove_test)

    return G_train, eval_edges, test_edges

# Split akmes
G_train, eval_edges, test_edges = split_edges(G_pruned)

print("Training Graph - Nodes:", len(G_train.nodes()), "Edges:", len(G_train.edges()))
print("Evaluation Edges:", len(eval_edges))
print("Test Edges:", len(test_edges))

# Simple random walk
def simple_random_walk(G, start_node, num_steps):
    current_node = start_node
    walk = [current_node]

    for _ in range(num_steps):
        neighbors = list(G.neighbors(current_node))
        if not neighbors:
            break
        current_node = random.choice(neighbors)
        walk.append(current_node)

    return walk

# Pixie Random Walk - the difference is that we have weights based on the neighbors and put also biases as extra
def pixie_random_walk(G, start_node, num_steps, alpha=0.2, biases=None):
    current_node = start_node
    walk = [current_node]

    for _ in range(num_steps):
        neighbors = list(G.neighbors(current_node))
        if not neighbors:
            break

        if biases:
            weights = np.ones(len(neighbors))
            if biases.get("uploader"):
                weights = [1.5 if G.nodes[neighbor].get('uploader') == G.nodes[current_node].get('uploader') else 1.0 for neighbor in neighbors]
            if biases.get("genre"):
                weights = [1.5 if G.nodes[neighbor].get('genre') == G.nodes[current_node].get('genre') else 1.0 for neighbor in neighbors]
            if biases.get("combined"):
                weights = [1.75 if (G.nodes[neighbor].get('uploader') == G.nodes[current_node].get('uploader') and G.nodes[neighbor].get('genre') == G.nodes[current_node].get('genre')) else weight for weight, neighbor in zip(weights, neighbors)]
            probs = np.array(weights) / sum(weights)
            next_node = np.random.choice(neighbors, p=probs)
        else:
            next_node = random.choice(neighbors)

        walk.append(next_node)
        current_node = next_node

    return walk

# Perfom the Walks
def perform_random_walks(G, num_walks, num_steps, walk_type="simple", biases=None):
    all_walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):
        start_node = random.choice(nodes)
        if walk_type == "simple":
            walk = simple_random_walk(G, start_node, num_steps)
        elif walk_type == "pixie":
            walk = pixie_random_walk(G, start_node, num_steps, biases=biases)
        all_walks.append(walk)
    return all_walks

# Parameters to try
num_walks = 250
num_steps = 2000

# Random walks
simple_walks = perform_random_walks(G_train, num_walks, num_steps, walk_type="simple")
pixie_walks = perform_random_walks(G_train, num_walks, num_steps, walk_type="pixie")
pixie_walks_uploader = perform_random_walks(G_train, num_walks, num_steps, walk_type="pixie", biases={"uploader": True})
pixie_walks_genre = perform_random_walks(G_train, num_walks, num_steps, walk_type="pixie", biases={"genre": True})
pixie_walks_combined = perform_random_walks(G_train, num_walks, num_steps, walk_type="pixie", biases={"combined": True})

# Visit counts all together
def aggregate_visit_counts(walks):
    visit_counts = {}
    for walk in walks:
        for node in walk:
            if node not in visit_counts:
                visit_counts[node] = 0
            visit_counts[node] += 1
    return visit_counts

simple_visit_counts = aggregate_visit_counts(simple_walks)
pixie_visit_counts = aggregate_visit_counts(pixie_walks)
pixie_visit_counts_uploader = aggregate_visit_counts(pixie_walks_uploader)
pixie_visit_counts_genre = aggregate_visit_counts(pixie_walks_genre)
pixie_visit_counts_combined = aggregate_visit_counts(pixie_walks_combined)

# Sort nodes by visit counts
sorted_simple = sorted(simple_visit_counts.items(), key=lambda x: x[1], reverse=True)
sorted_pixie = sorted(pixie_visit_counts.items(), key=lambda x: x[1], reverse=True)
sorted_pixie_uploader = sorted(pixie_visit_counts_uploader.items(),key=lambda x: x[1], reverse=True)
sorted_pixie_genre = sorted(pixie_visit_counts_genre.items(),key=lambda x: x[1], reverse=True)
sorted_pixie_combined =  sorted(pixie_visit_counts_combined.items(),key=lambda x: x[1], reverse=True)

# top 10 influential nodes for each random walk type
top_10_simple = sorted_simple[:10]
top_10_pixie = sorted_pixie[:10]
top_10_pixie_uploader = sorted_pixie_uploader[:10]
top_10_pixie_genre = sorted_pixie_genre[:10]
top_10_pixie_combined = sorted_pixie_combined[:10]

# Dataframe the top 10 inflential nodes for each type of walk
# Create a DataFrame to hold the top 10 influential nodes
data = {
    "Simple Random Walk": [node[0] for node in top_10_simple],
    "Simple RW Score": [node[1] for node in top_10_simple],
    "Pixie Random Walk": [node[0] for node in top_10_pixie],
    "Pixie RW Score": [node[1] for node in top_10_pixie],
    "Pixie RW Uploader Bias": [node[0] for node in top_10_pixie_uploader],
    "Uploader RW Score": [node[1] for node in top_10_pixie_uploader],
    "Pixie RW Genre Bias": [node[0] for node in top_10_pixie_genre],
    "Genre RW Score": [node[1] for node in top_10_pixie_genre],
    "Pixie RW Combined": [node[0] for node in top_10_pixie_combined],
    "Combined RW Score": [node[1] for node in top_10_pixie_combined]
}

df_top_10_influential = pd.DataFrame(data)

# Display the DataFrame
display(df_top_10_influential)

# Rank visited nodes
def rank_visited_nodes(visit_counts):
    sorted_nodes = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)
    ranked_nodes = [node for node, count in sorted_nodes]
    return ranked_nodes

ranked_nodes_simple = rank_visited_nodes(simple_visit_counts)
ranked_nodes_pixie = rank_visited_nodes(pixie_visit_counts)
ranked_nodes_pixie_uploader = rank_visited_nodes(pixie_visit_counts_uploader)
ranked_nodes_pixie_genre = rank_visited_nodes(pixie_visit_counts_genre)
ranked_nodes_pixie_combined = rank_visited_nodes(pixie_visit_counts_combined)

# Evaluate Visit Counts
def evaluate_visit_counts(G, visit_counts, eval_edges, test_edges):
    sorted_nodes = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)
    ranked_nodes = [node for node, count in sorted_nodes]

    true_positive_eval = len([edge for edge in eval_edges if edge[0] in ranked_nodes and edge[1] in ranked_nodes])
    false_positive_eval = len(ranked_nodes) - true_positive_eval
    false_negative_eval = len([edge for edge in eval_edges if edge[0] not in ranked_nodes or edge[1] not in ranked_nodes])

    true_positive_test = len([edge for edge in test_edges if edge[0] in ranked_nodes and edge[1] in ranked_nodes])
    false_positive_test = len(ranked_nodes) - true_positive_test
    false_negative_test = len([edge for edge in test_edges if edge[0] not in ranked_nodes or edge[1] not in ranked_nodes])

    precision_eval = true_positive_eval / (true_positive_eval + false_positive_eval) if true_positive_eval + false_positive_eval > 0 else 0
    recall_eval = true_positive_eval / (true_positive_eval + false_negative_eval) if true_positive_eval + false_negative_eval > 0 else 0
    f1_score_eval = 2 * (precision_eval * recall_eval) / (precision_eval + recall_eval) if precision_eval + recall_eval > 0 else 0
    accuracy_eval = true_positive_eval / len(eval_edges) if len(eval_edges) > 0 else 0

    precision_test = true_positive_test / (true_positive_test + false_positive_test) if true_positive_test + false_positive_test > 0 else 0
    recall_test = true_positive_test / (true_positive_test + false_negative_test) if true_positive_test + false_negative_test > 0 else 0
    f1_score_test = 2 * (precision_test * recall_test) / (precision_test + recall_test) if precision_test + recall_test > 0 else 0
    accuracy_test = true_positive_test / len(test_edges) if len(test_edges) > 0 else 0

    results_eval = {
        'precision': precision_eval,
        'recall': recall_eval,
        'f1_score': f1_score_eval,
        'accuracy': accuracy_eval
    }

    results_test = {
        'precision': precision_test,
        'recall': recall_test,
        'f1_score': f1_score_test,
        'accuracy': accuracy_test
    }

    return results_eval, results_test

# Evaluate visit counts
results_simple_eval, results_simple_test = evaluate_visit_counts(G_train, simple_visit_counts, eval_edges, test_edges)
results_pixie_eval, results_pixie_test = evaluate_visit_counts(G_train, pixie_visit_counts, eval_edges, test_edges)

# Print results
print("Evaluation Results - Simple Random Walk:", results_simple_eval)
print("Test Results - Simple Random Walk:", results_simple_test)
print("Evaluation Results - Pixie Random Walk:", results_pixie_eval)
print("Test Results - Pixie Random Walk:", results_pixie_test)

# Evaluate visit counts
results_simple_eval, results_simple_test = evaluate_visit_counts(G_train, simple_visit_counts, eval_edges, test_edges)
results_pixie_uploader_eval, results_pixie_uploader_test = evaluate_visit_counts(G_train, pixie_visit_counts_uploader, eval_edges, test_edges)
results_pixie_genre_eval, results_pixie_genre_test = evaluate_visit_counts(G_train, pixie_visit_counts_genre, eval_edges, test_edges)
results_pixie_comb_eval, results_pixie_comb_test = evaluate_visit_counts(G_train, pixie_visit_counts_combined,eval_edges, test_edges)


# Function to convert results to DataFrame
def results_to_dataframe(results_eval, results_test, walk_type):
    data = [
        ['Evaluation', results_eval['precision'], results_eval['recall'], results_eval['f1_score'], results_eval['accuracy']],
        ['Test', results_test['precision'], results_test['recall'], results_test['f1_score'], results_test['accuracy']]
    ]
    df = pd.DataFrame(data, columns=['Set', 'Precision', 'Recall', 'F1-Score', 'Accuracy'])
    df['Walk Type'] = walk_type
    return df

# Results to DataFrames
df_simple = results_to_dataframe(results_simple_eval, results_simple_test, "Simple Random Walk")
df_pixie_uploader = results_to_dataframe(results_pixie_uploader_eval, results_pixie_uploader_test, "Pixie Random Walk (Uploader Bias)")
df_pixie_genre = results_to_dataframe(results_pixie_genre_eval, results_pixie_genre_test, "Pixie Random Walk (Genre Bias)")
df_pixie_combined = results_to_dataframe(results_pixie_comb_eval, results_pixie_comb_test, "Pixie Random Walk (Combined Bias)")

# Combine all results into a single DataFrame
df_results = pd.concat([df_simple, df_pixie_uploader, df_pixie_genre,df_pixie_combined], ignore_index=True)

# Display the result
display(df_results)


if __name__ == "__main__":
    extract_and_load_data(zip_file_path)
    print(print_combined)