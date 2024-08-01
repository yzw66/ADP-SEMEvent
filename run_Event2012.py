import sys
from ADPSEMEvent import hier_2D_SE_mini, get_global_edges, search_stable_points
from utils import evaluate, decode
from datetime import datetime
import math
import numpy as np
import pickle
import pandas as pd
import os
from os.path import exists
import time
import multiprocessing

def get_stable_point(path, if_updata, epsilon):
    stable_point_path = path + f'stable_point_{epsilon}.pkl'
    if not exists(stable_point_path) or if_updata == True:
        embeddings_path = path + 'SBERT_embeddings.pkl'
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        first_stable_point, global_stable_point, Sensitivity = search_stable_points(embeddings, epsilon, path)
        stable_points = {'first': first_stable_point, 'global': global_stable_point}
        with open(stable_point_path, 'wb') as fp:
            pickle.dump(stable_points, fp)
        print('stable points stored.')

    with open(stable_point_path, 'rb') as f:
        stable_points = pickle.load(f)
    print('stable points loaded.')
    return stable_points, Sensitivity

def run_hier_2D_SE_mini_Event2012_open_set(n = 400, e_a = True, e_s = True, test_with_one_block = True, epsilon = 0.2):
    save_path = 'data/Event2012/open_set/'
  
    if test_with_one_block:
        blocks = [16]
    else:
        blocks = [i+1 for i in range(20) if i+1>=1]
    for block in blocks:
        print('\n\n====================================================')
        print('block: ', block)
        print(datetime.now().strftime("%H:%M:%S"))

        folder = f'{save_path}{block}/'
        
        # load message embeddings
        embeddings_path = folder + 'SBERT_embeddings.pkl'
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        
        df_np = np.load(f'{folder}{block}.npy', allow_pickle=True)
        df = pd.DataFrame(data=df_np, columns=["original_index", "event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",\
                "place_type", "place_full_name", "place_country_code", "hashtags", "user_mentions", "image_urls", "entities", 
                "words", "filtered_words", "sampled_words", "date"])
        all_node_features = [[str(u)] + \
            [str(each) for each in um] + \
            [h.lower() for h in hs] + \
            e \
            for u, um, hs, e in \
            zip(df['user_id'], df['user_mentions'],  df['hashtags'], df['entities'])]
        
        start_time = time.time()
        stable_points, Sensitivity = get_stable_point(folder, if_updata=True, epsilon=epsilon)
        if e_a == False: # only rely on e_s (semantic-similarity-based edges)
            default_num_neighbors = stable_points['global']
        else:
            default_num_neighbors = stable_points['first']
        if default_num_neighbors == 0: 
            default_num_neighbors = math.ceil((len(embeddings)/1000)*10)
        
        global_edges = get_global_edges(all_node_features, epsilon, folder, default_num_neighbors, e_a = e_a, e_s = e_s)

        # corr_matrix = np.corrcoef(embeddings)
        # np.fill_diagonal(corr_matrix, 0)
        corr_matrix = np.load(f"{folder}"+f'corr_matrix_{epsilon}.npy')
        weighted_global_edges = [(edge[0], edge[1], corr_matrix[edge[0]-1, edge[1]-1]) for edge in global_edges \
            if corr_matrix[edge[0]-1, edge[1]-1] > 0] # node encoding starts from 1
        
        np.save(f"{folder}"+f'weighted_global_edges_{epsilon}.npy', np.array(weighted_global_edges))
        # sys.exit()
        
        division = hier_2D_SE_mini(weighted_global_edges, len(embeddings), n = n)
        # print(division)
        print(datetime.now().strftime("%H:%M:%S"))

        prediction = decode(division)

        labels_true = df['event_id'].tolist()
        n_clusters = len(list(set(labels_true)))
        print('n_clusters gt: ', n_clusters)

        nmi, ami, ari = evaluate(labels_true, prediction)
        print('n_clusters pred: ', len(division))
        print('nmi: ', nmi)
        print('ami: ', ami)
        print('ari: ', ari)
        with open("Event2018_open_set_"+f'{epsilon}.txt', 'a') as f:
            f.write("block:" + str(block) + '\n')
            f.write("division:"+str(division)+ '\n')
            f.write('Runtime: ' + str(time.time() - start_time) + " Seconds" + '\n')
            f.write('n_clusters gt: '+ str(len(list(set(labels_true))))+ '\n')
            f.write('n_clusters pred: ' + str(len(division)) + '\n')
            f.write('epsilon: ' + str(epsilon) + '\n')
            f.write('n: ' + str(n) + '\n')
            f.write('Sensitivity: ' + str(Sensitivity) + '\n')
            f.write('nmi: ' + str(nmi) + '\n')
            f.write('ami: ' + str(ami) + '\n')
            f.write('ari: ' + str(ari) + '\n' + '\n')
        
    return

def run_hier_2D_SE_mini_Event2012_closed_set(n = 300, e_a = True, e_s = True, epsilon = None):
    save_path = 'data/Event2012/closed_set/'

    #load test_set_df
    test_set_df_np_path = save_path + 'test_set.npy'
    test_df_np = np.load(test_set_df_np_path, allow_pickle=True)
    test_df = pd.DataFrame(data=test_df_np, columns=["event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",\
            "place_type", "place_full_name", "place_country_code", "hashtags", "user_mentions", "image_urls", "entities", 
            "words", "filtered_words", "sampled_words"])
    print("Dataframe loaded.")
    all_node_features = [[str(u)] + \
        [str(each) for each in um] + \
        [h.lower() for h in hs] + \
        e \
        for u, um, hs, e in \
        zip(test_df['user_id'], test_df['user_mentions'],  test_df['hashtags'], test_df['entities'])]

    # load embeddings of the test set messages
    with open(f'{save_path}/SBERT_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    start_time = time.time()
    stable_points, Sensitivity = get_stable_point(save_path, if_updata=True, epsilon=epsilon)
    default_num_neighbors = stable_points['first']

    global_edges = get_global_edges(all_node_features, epsilon, default_num_neighbors, e_a = e_a, e_s = e_s)
    corr_matrix = np.load(f"{save_path}"+f'corr_matrix_{epsilon}.npy')
    # corr_matrix = np.corrcoef(embeddings)
    # np.fill_diagonal(corr_matrix, 0)
    weighted_global_edges = [(edge[0], edge[1], corr_matrix[edge[0]-1, edge[1]-1]) for edge in global_edges \
        if corr_matrix[edge[0]-1, edge[1]-1] > 0] # node encoding starts from 1

    division = hier_2D_SE_mini(weighted_global_edges, len(embeddings), n = n)
    prediction = decode(division)

    labels_true = test_df['event_id'].tolist()
    n_clusters = len(list(set(labels_true)))
    print('n_clusters gt: ', n_clusters)

    nmi, ami, ari = evaluate(labels_true, prediction)
    print('n_clusters pred: ', len(division))
    print('nmi: ', nmi)
    print('ami: ', ami)
    print('ari: ', ari)
    with open("Event2012_closed_set_"+f'{n}_{epsilon}.txt', 'a') as f:
        f.write("division:"+str(division)+ '\n')
        f.write('Runtime: ' + str(time.time() - start_time) + " Seconds" + '\n')
        f.write('n_clusters gt: '+ str(len(list(set(labels_true))))+ '\n')
        f.write('n_clusters pred: ' + str(len(division)) + '\n')
        f.write('epsilon: ' + str(epsilon) + '\n')
        f.write('Sensitivity: ' + str(Sensitivity) + '\n')
        f.write('nmi: ' + str(nmi) + '\n')
        f.write('ami: ' + str(ami) + '\n')
        f.write('ari: ' + str(ari) + '\n' + '\n')
    return

def run_hier_2D_SE_mini_Event2018_open_set(n = 300, e_a = True, e_s = True, test_with_one_block = True, epsilon = None):
    save_path = 'data/Event2018/open_set/'
  
    if test_with_one_block:
        blocks = [16]
    else:
        blocks = [i+1 for i in range(16)]
    for block in blocks:
        print('\n\n====================================================')
        print('block: ', block)
        print(datetime.now().strftime("%H:%M:%S"))

        folder = f'{save_path}{block}/'
        
        # load message embeddings
        embeddings_path = folder + 'SBERT_embeddings.pkl'
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        
        df_np = np.load(f'{folder}{block}.npy', allow_pickle=True)
        df = pd.DataFrame(data=df_np, columns=["original_index", "tweet_id", "user_name", "text", "time", "event_id", "user_mentions", \
                "hashtags", "urls", "words", "created_at", "filtered_words", "entities", "sampled_words", "date"])
        all_node_features = [list(set([str(u)] + \
            [str(each) for each in um] + \
            [h.lower() for h in hs] + \
            e)) \
            for u, um, hs, e in \
            zip(df['user_name'], df['user_mentions'],  df['hashtags'], df['entities'])]
        
        start_time = time.time()
        stable_points, Sensitivity = get_stable_point(folder, if_updata=True, epsilon=epsilon)        
        # stable_points = get_stable_point(folder)
        if e_a == False: # only rely on e_s (semantic-similarity-based edges)
            default_num_neighbors = stable_points['global']
        else:
            default_num_neighbors = stable_points['first']
        if default_num_neighbors == 0: 
            default_num_neighbors = math.ceil((len(embeddings)/1000)*10)
        
        global_edges = get_global_edges(all_node_features, epsilon, folder, default_num_neighbors, e_a = e_a, e_s = e_s)

        # corr_matrix = np.corrcoef(embeddings)
        # np.fill_diagonal(corr_matrix, 0)
        corr_matrix = np.load(f"{folder}"+f'corr_matrix_{epsilon}.npy')
        weighted_global_edges = [(edge[0], edge[1], corr_matrix[edge[0]-1, edge[1]-1]) for edge in global_edges \
            if corr_matrix[edge[0]-1, edge[1]-1] > 0] # node encoding starts from 1
        
        division = hier_2D_SE_mini(weighted_global_edges, len(embeddings), n = n)
        print(datetime.now().strftime("%H:%M:%S"))

        prediction = decode(division)

        labels_true = df['event_id'].tolist()
        n_clusters = len(list(set(labels_true)))
        print('n_clusters gt: ', n_clusters)

        nmi, ami, ari = evaluate(labels_true, prediction)
        print('n_clusters pred: ', len(division))
        print('nmi: ', nmi)
        print('ami: ', ami)
        print('ari: ', ari)
        
        with open("Event2018_open_set_"+f'{n}_{epsilon}.txt', 'a') as f:
            f.write("block:" + str(block) + '\n')
            f.write("division:"+str(division)+ '\n')
            f.write('Runtime: ' + str(time.time() - start_time) + " Seconds" + '\n')
            f.write('n_clusters gt: '+ str(len(list(set(labels_true))))+ '\n')
            f.write('n_clusters pred: ' + str(len(division)) + '\n')
            f.write('epsilon: ' + str(epsilon) + '\n')
            f.write('Sensitivity: ' + str(Sensitivity) + '\n')
            f.write('nmi: ' + str(nmi) + '\n')
            f.write('ami: ' + str(ami) + '\n')
            f.write('ari: ' + str(ari) + '\n' + '\n')       
    return

def run_hier_2D_SE_mini_Event2018_closed_set(n = 800, e_a = True, e_s = True, epsilon = None):
    save_path = 'data/Event2018/closed_set/'

    #load test_set_df
    test_set_df_np_path = save_path + 'test_set.npy'
    test_df_np = np.load(test_set_df_np_path, allow_pickle=True)
    test_df = pd.DataFrame(data=test_df_np, columns=["tweet_id", "user_name", "text", "time", "event_id", "user_mentions", \
            "hashtags", "urls", "words", "created_at", "filtered_words", "entities", "sampled_words"])
    print("Dataframe loaded.")
    all_node_features = [list(set([str(u)] + \
        [str(each) for each in um] + \
        [h.lower() for h in hs] + \
        e)) \
        for u, um, hs, e in \
        zip(test_df['user_name'], test_df['user_mentions'],  test_df['hashtags'], test_df['entities'])]

    # load embeddings of the test set messages
    with open(f'{save_path}/SBERT_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    start_time = time.time()
    stable_points, Sensitivity = get_stable_point(save_path, if_updata=True, epsilon=epsilon) 
    default_num_neighbors = stable_points['first']

    global_edges = get_global_edges(all_node_features, epsilon, default_num_neighbors, e_a = e_a, e_s = e_s)
    # corr_matrix = np.corrcoef(embeddings)
    # np.fill_diagonal(corr_matrix, 0)
    corr_matrix = np.load(f"{save_path}"+f'corr_matrix_{epsilon}.npy')
    weighted_global_edges = [(edge[0], edge[1], corr_matrix[edge[0]-1, edge[1]-1]) for edge in global_edges \
        if corr_matrix[edge[0]-1, edge[1]-1] > 0] # node encoding starts from 1

    division = hier_2D_SE_mini(weighted_global_edges, len(embeddings), n = n)
    prediction = decode(division)

    labels_true = test_df['event_id'].tolist()
    n_clusters = len(list(set(labels_true)))
    print('n_clusters gt: ', n_clusters)

    nmi, ami, ari = evaluate(labels_true, prediction)
    print('n_clusters pred: ', len(division))
    print('nmi: ', nmi)
    print('ami: ', ami)
    print('ari: ', ari)

    with open("Event2018_closed_set_"+ f'{n}_{epsilon}.txt' , 'a') as f:
        f.write("division:"+str(division)+ '\n')
        f.write('Runtime: ' + str(time.time() - start_time) + " Seconds" + '\n')
        f.write('n_clusters gt: '+ str(len(list(set(labels_true))))+ '\n')
        f.write('n_clusters pred: ' + str(len(division)) + '\n')
        f.write('epsilon: ' + str(epsilon) + '\n')
        f.write('Sensitivity: ' + str(Sensitivity) + '\n')
        f.write('nmi: ' + str(nmi) + '\n')
        f.write('ami: ' + str(ami) + '\n')
        f.write('ari: ' + str(ari) + '\n' + '\n')  
    return


# =====================================================================================================
def create_process_Event2012_open_set(epsilon):
    p = multiprocessing.Process(target=run_hier_2D_SE_mini_Event2012_open_set, 
                                kwargs={"n": 100,
                                        "e_a":True, "e_s":True, "test_with_one_block":True,
                                        "epsilon": epsilon})
    p.start()
    return p

def create_process_Event2012_closed_set(epsilon):
    p = multiprocessing.Process(target=run_hier_2D_SE_mini_Event2012_closed_set, 
                                kwargs={"n": 300,
                                        "e_a":True, "e_s":True,
                                        "epsilon": epsilon})
    p.start()
    return p

def create_process_Event2018_open_set(epsilon):
    p = multiprocessing.Process(target=run_hier_2D_SE_mini_Event2018_open_set, 
                                kwargs={"n": 800,
                                        "e_a":True, "e_s":True, "test_with_one_block":False,
                                        "epsilon": epsilon})
    p.start()
    return p

def create_process_Event2018_closed_set(epsilon):
    p = multiprocessing.Process(target=run_hier_2D_SE_mini_Event2018_closed_set, 
                                kwargs={"n": 300,
                                        "e_a":True, "e_s":True,
                                        "epsilon": epsilon})
    p.start()
    return p

if __name__ == "__main__":
    # Privacy budget parameters in DP
    epsilons = [15]
    # epsilons = [10, 15]
    processes = [create_process_Event2012_open_set(epsilon) for epsilon in epsilons]
    for process in processes:
        process.join()
    print("All processes have completed their tasks.")

    
