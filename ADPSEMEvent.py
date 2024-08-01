import networkx as nx
from SE import SE
from itertools import combinations, chain
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering

def make_symmetric(matrix):
    return np.triu(matrix) + np.triu(matrix, 1).T

def search_stable_points(embeddings, epsilon, path, max_num_neighbors = 200):
    print("size_of_embeddings",len(embeddings))
    corr_matrix = np.corrcoef(embeddings)  
    np.fill_diagonal(corr_matrix, 0)

    print("epsilon=",epsilon)
    s = -1
    if epsilon != None:
        max_ = np.max(corr_matrix)
        min_ = np.min(corr_matrix)
        print("Local Sensitivity:",(max_- min_))
        # delta = 10e-6  
        delta = 1 / len(embeddings)**2  
        beta = epsilon / (2 * np.log(2/delta))
        S = np.exp(-beta) * (max_- min_) * 2
        print("Smooth Sensitivity:", S)
        if S < 2:
            s = S
        else:
            s = 2

        print("Sensitivity=",s)
        corr_matrix = [[i+np.random.laplace(loc=0, scale=s/epsilon) for i in corr_matrix_] for corr_matrix_ in corr_matrix]
        corr_matrix = np.array(corr_matrix)
        corr_matrix = make_symmetric(corr_matrix)

    np.fill_diagonal(corr_matrix, 0)
    print(f"{path}"+f'corr_matrix_{epsilon}.npy')
    np.save(f"{path}"+f'corr_matrix_{epsilon}.npy', corr_matrix)
    corr_matrix_sorted_indices = np.argsort(corr_matrix)
    
    all_1dSEs = []
    seg = None
    for i in range(max_num_neighbors):
        dst_ids = corr_matrix_sorted_indices[:, -(i+1)]
        knn_edges = [(s+1, d+1, corr_matrix[s, d]) \
            for s, d in enumerate(dst_ids) if corr_matrix[s, d] > 0] # (s+1, d+1): +1 as node indexing starts from 1 instead of 0
        if i == 0:
            g = nx.Graph()
            g.add_weighted_edges_from(knn_edges)
            seg = SE(g)
            all_1dSEs.append(seg.calc_1dSE())
        else:
            all_1dSEs.append(seg.update_1dSE(all_1dSEs[-1], knn_edges))
    
    #print('all_1dSEs: ', all_1dSEs)
    stable_indices = []
    for i in range(1, len(all_1dSEs) - 1):
        if all_1dSEs[i] < all_1dSEs[i - 1] and all_1dSEs[i] < all_1dSEs[i + 1]:
            stable_indices.append(i)
    if len(stable_indices) == 0:
        print('No stable points found after checking k = 1 to ', max_num_neighbors)
        return 0, 0, s
    else:
        stable_SEs = [all_1dSEs[index] for index in stable_indices]
        index = stable_indices[stable_SEs.index(min(stable_SEs))]
        print('stable_indices: ', stable_indices)
        print('stable_SEs: ', stable_SEs)
        print('First stable point: k = ', stable_indices[0]+1, ', correspoding 1dSE: ', stable_SEs[0]) # n_neighbors should be index + 1
        print('Global stable point within the searching range: k = ', index + 1, \
            ', correspoding 1dSE: ', all_1dSEs[index]) # n_neighbors should be index + 1

    return stable_indices[0]+1, index + 1, s # first stable point, global stable point


def get_graph_edges(attributes):
    attr_nodes_dict = {}
    for i, l in enumerate(attributes):
        for attr in l:
            if attr not in attr_nodes_dict:
                attr_nodes_dict[attr] = [i+1] # node indexing starts from 1
            else:
                attr_nodes_dict[attr].append(i+1)

    for attr in attr_nodes_dict.keys():
        attr_nodes_dict[attr].sort()

    graph_edges = []
    for l in attr_nodes_dict.values():
        graph_edges += list(combinations(l, 2))
    return list(set(graph_edges))


def get_knn_edges(epsilon, path, default_num_neighbors):
    # corr_matrix = np.corrcoef(embeddings)
    # np.fill_diagonal(corr_matrix, 0)
    corr_matrix = np.load(f"{path}"+f'corr_matrix_{epsilon}.npy')
    corr_matrix_sorted_indices = np.argsort(corr_matrix)
    knn_edges = []
    for i in range(default_num_neighbors):
        dst_ids = corr_matrix_sorted_indices[:, -(i+1)]
        knn_edges += [(s+1, d+1) if s < d else (d+1, s+1) \
            for s, d in enumerate(dst_ids) if corr_matrix[s, d] > 0] # (s+1, d+1): +1 as node indexing starts from 1 instead of 0
    return list(set(knn_edges))

def get_global_edges(attributes, epsilon, folder, default_num_neighbors, e_a = True, e_s = True):
    graph_edges, knn_edges = [], []
    if e_a == True:
        graph_edges = get_graph_edges(attributes)
    if e_s == True:
        knn_edges = get_knn_edges(epsilon, folder, default_num_neighbors)
    return list(set(knn_edges + graph_edges))

def get_subgraphs_edges(clusters, graph_splits, weighted_global_edges):
    '''
    get the edges of each subgraph

    clusters: a list containing the current clusters, each cluster is a list of nodes of the original graph
    graph_splits: a list of (start_index, end_index) pairs, each (start_index, end_index) pair indicates a subset of clusters, 
        which will serve as the nodes of a new subgraph
    weighted_global_edges: a list of (start node, end node, edge weight) tuples, each tuple is an edge in the original graph

    return: all_subgraphs_edges: a list containing the edges of all subgraphs
    '''
    all_subgraphs_edges = []
    for split in graph_splits:
        subgraph_clusters = clusters[split[0]:split[1]]
        subgraph_nodes = list(chain(*subgraph_clusters))
        subgraph_edges = [edge for edge in weighted_global_edges if edge[0] in subgraph_nodes and edge[1] in subgraph_nodes]
        all_subgraphs_edges.append(subgraph_edges)
    return all_subgraphs_edges


def get_best_egde(adj_matrix_, subgraphs_, all_subgraphs):
    adj_matrix = adj_matrix_.copy()
    
    mask_nodes = list(set(all_subgraphs+subgraphs_))  
    if len(mask_nodes) >0:
        adj_matrix[mask_nodes, :] = 0
        adj_matrix[:, mask_nodes] = 0

    flat_index = np.argmax(adj_matrix)
    egde = np.unravel_index(flat_index, adj_matrix.shape)
    weight = adj_matrix[egde]
    if weight > 0:
        return list(egde), weight
    else:
        print("There is no egdes in current G")
        return -1, -1

def get_best_node(adj_matrix_, subgraphs_, all_subgraphs):
    adj_matrix = adj_matrix_.copy()

    mask_nodes = list(set(all_subgraphs+subgraphs_))  
    nodes_to_modify = np.array(mask_nodes)
    adj_matrix[np.ix_(nodes_to_modify, nodes_to_modify)] = 0

    distance = adj_matrix[subgraphs_].sum(axis=0)
    distance_sort_arg = np.argsort(distance)[::-1]
    distance_sort = np.sort(distance)[::-1]
    avg = np.mean(distance[distance>0])
    indices = distance_sort[distance_sort>avg]

    if len(indices) > 0:
        return distance_sort_arg[:len(indices)].tolist(), distance_sort[:len(indices)].tolist()
    else:
        print("There are no edges connected to the current subgraph")
        return -1, -1


def get_subgraphs(adj_matrix, division, n, k_max):
    merged_rows_matrix = np.vstack([ adj_matrix[np.array(ls_)-1].sum(axis=0).tolist() for ls_ in division ])
    final_sum = np.array([ merged_rows_matrix[:, np.array(ls_)-1].sum(axis=1).tolist() for ls_ in division ] )
    np.fill_diagonal(final_sum, 0)
    G = nx.from_numpy_matrix(final_sum)
    
    subgraphs = []
    all_subgraphs = [] 
    for k in range(k_max):
        subgraphs_ = []
        if len(final_sum) - len(all_subgraphs)<= n: 
            G.remove_nodes_from(all_subgraphs)
            subgraphs_ = list(G.nodes)
            subgraphs.append(subgraphs_)
            print(len(subgraphs_), subgraphs_)
            break

        max_edge_or_node, max_weight = get_best_egde(final_sum, subgraphs_, all_subgraphs)
        subgraphs_.extend(max_edge_or_node)
        all_subgraphs.extend(max_edge_or_node)
        while True:
            if len(subgraphs_) >= n:
                break
            node_, weight_ = get_best_node(final_sum, subgraphs_, all_subgraphs)
            if node_ == -1:
                max_edge_or_node, max_weight = get_best_egde(final_sum, subgraphs_, all_subgraphs)
                subgraphs_.extend(max_edge_or_node)
                all_subgraphs.extend(max_edge_or_node)
                continue
            else:
                if len(subgraphs_) + len(node_) > n:
                    index_ = n - len(subgraphs_)
                    subgraphs_.extend(node_[:index_])
                    all_subgraphs.extend(node_[:index_])
                else:
                    subgraphs_.extend(node_)
                    all_subgraphs.extend(node_)
        subgraphs.append(subgraphs_)
        # print(len(subgraphs_), subgraphs_)

    # subgraphs = [[element + 1 for element in row] for row in subgraphs]

    new_division = []
    for subgraphs_index in subgraphs:
        new_division_ = []
        for index in subgraphs_index:
            new_division_.append(division[index])
        new_division.append(new_division_)
        
    return new_division


def hier_2D_SE_mini(weighted_global_edges, n_messages, n = 100):
    '''
    hierarchical 2D SE minimization
    '''
    ite = 0
    # initially, each node (message) is in its own cluster
    # node encoding starts from 1

    G = nx.Graph()
    G.add_weighted_edges_from(weighted_global_edges)
    adj_matrix = nx.to_numpy_array(G)

    clusters = [[i] for i in list(G.nodes)]
    while True:
        ite += 1
        print('\n=========Iteration ', str(ite), '=========')
        n_clusters = len(clusters)
        graph_splits = [(s, min(s+n, n_clusters)) for s in range(0, n_clusters, n)] # [s, e)
        # all_subgraphs_edges = get_subgraphs_edges(clusters, graph_splits, weighted_global_edges)

        if 1:
            subgraphs = get_subgraphs(adj_matrix, clusters, n, len(graph_splits))

            all_subgraphs_edges = []
            for subgraph_nodes in subgraphs:
                subgraph_nodes = [str(item) for sublist in subgraph_nodes for item in sublist]
                subgraph_edges = [(int(edge[0]),int(edge[1]),edge[2]) for edge in weighted_global_edges 
                                  if str(edge[0]) in subgraph_nodes and str(edge[1]) in subgraph_nodes]
                all_subgraphs_edges.append(subgraph_edges)

        else:
            all_subgraphs_edges = get_subgraphs_edges(clusters, graph_splits, weighted_global_edges)


        last_clusters = clusters
        print(f"the number of clusters: {len(last_clusters)}")
        clusters = []
        for i, subgraph_edges in enumerate(all_subgraphs_edges):
            print('\tSubgraph ', str(i+1))

            g = nx.Graph()
            g.add_weighted_edges_from(subgraph_edges)
            seg = SE(g)
            if 1:
                seg.division = {j: cluster for j, cluster in enumerate(subgraphs[i]) }
                # print({j: cluster for j, cluster in enumerate(subgraphs[i]) })
            else:
                seg.division = {j: cluster for j, cluster in enumerate(last_clusters[graph_splits[i][0]:graph_splits[i][1]])}
                # print(seg.division)
            seg.add_isolates()
            for k in seg.division.keys():
                for node in seg.division[k]:
                    seg.graph.nodes[node]['comm'] = k
            seg.update_struc_data()
            seg.update_struc_data_2d()
            seg.update_division_MinSE()

            print(f"size of subgraph{str(i+1)}: {len(subgraphs[i])} to {len(list(seg.division.values()))}")

            clusters += list(seg.division.values())

        if len(graph_splits) == 1:
            break
        if clusters == last_clusters:
            n *= 2
    return clusters