import numpy as np
import pykitti
import json
import os
from tqdm import tqdm
import pickle
from sklearn.neighbors import KDTree
import pandas as pd

def get_dataset(sequence_id='00', basedir= '/home/yanbhliu/data/kitti/'):
    return pykitti.odometry(basedir, sequence_id)


def p_dist(pose1, pose2, threshold=3, print_bool=False):
    # dist = np.linalg.norm(pose1[:,-1]-pose2[:,-1])    # xyz
    dist = np.linalg.norm(pose1[0::2, -1] - pose2[0::2, -1])  # xz in cam0
    if print_bool==True:
        print(dist)
    if abs(dist) <= threshold:
        return True
    else:
        return False

def t_dist(t1, t2, threshold=10):
    if abs((t1-t2).total_seconds()) >= threshold:
        return True
    else:
        return False


def train_test_split(train_nums=5, on_framenum=True,
                     ids=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']):
    ids = np.array(ids)
    if on_framenum:
        # choose training dataset with fewer frame nums
        framenums = [len(get_dataset(id).timestamps) for id in ids]
        print(framenums)

        train_ids = list(ids[np.argsort(framenums)[:train_nums]])
        test_ids = list(ids[np.argsort(framenums)[train_nums:]])
        return train_ids, test_ids

    else:
        train_ids = list(ids[:train_nums])
        test_ids = list(ids[train_nums:])
        return train_ids, test_ids


def get_evaluate_dict(ids, output_dir):
    evaluate = {}
    database_sets = [] #database
    query_sets = [] #query
    database_trees=[]
    query_trees = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count_sec = 1
    base_path = "sequences"

    for id in tqdm(ids):
        database = {} #database
        query = {} #query

        dataset = get_dataset(id)
        initial_time = dataset.timestamps[0]
        # print('timestamps:', (dataset.timestamps[12]- dataset.timestamps[0]).total_seconds())
        
        df_database= pd.DataFrame(columns=['query_velo','x','z'])
        df_query = pd.DataFrame(columns=['query_velo','x','z']) # df_test
        prev_time = -0.6
        count_db = 0
        if id not in evaluate:
            print("generate test set on:", id)
            evaluate[id] = {}

            for t1 in range(len(dataset.timestamps)):
                # print('pose:', dataset.poses[t1][0::2, -1][0])
                time_diff = (dataset.timestamps[t1] - initial_time).total_seconds() #current time diff
                
                # print('time_diff:', time_diff, 'type:',type(time_diff))
                if (time_diff - prev_time> 0.2):
                    query_bin = os.path.join(base_path, id, 'velodyne', '%06d' % int(t1)+".bin")
                    # if ( (time_diff < 100) or (time_diff > 259 and time_diff<=264 ) ):  #seq 08
                    if (time_diff < 170):  #seq 00
                        print('pose:',dataset.poses[t1][0::2, -1])
                        database[len(database.keys())] = {'query_velo': query_bin, 'x': dataset.poses[t1][0::2, -1][0], 'z': dataset.poses[t1][0::2, -1][1]}
                        
                        df_database.loc[len(df_database.index)] = [query_bin, dataset.poses[t1][0::2, -1][0], dataset.poses[t1][0::2, -1][1]]
                       
                        count_db = count_db + 1
                        print("count_db:", count_db)
                    else:
                        query[len(query.keys())] = {'query_velo': query_bin, 'x': dataset.poses[t1][0::2, -1][0], 'z': dataset.poses[t1][0::2, -1][1]}
                        
                        df_query.loc[len(df_database.index)] = [query_bin, dataset.poses[t1][0::2, -1][0], dataset.poses[t1][0::2, -1][1]]
                    prev_time = time_diff
   
        
        
        database_tree = KDTree(df_database[['x','z']])
        test_tree = KDTree(df_query[['x','z']])
        database_trees.append(database_tree)
        query_trees.append(test_tree)

        query_sets.append(query)
        database_sets.append(database)
    
    for i in range(len(database_sets)):
        tree=database_trees[i]
        for j in range(len(query_sets)):
            # if(i==j):
            #     continue
            for key in range(len(query_sets[j].keys())):
                coor=np.array([[query_sets[j][key]["x"],query_sets[j][key]["z"]]])
                index = tree.query_radius(coor, r=25)
                #indices of the positive matches in database i of each query (key) in test set j
                query_sets[j][key][i]=index[0].tolist()
    
    with open(os.path.join(output_dir,'kitti_{}_database_evaluate_new.pickle'.format(sequences[0])), 'wb') as handle:
            pickle.dump(database_sets, handle, protocol=pickle.HIGHEST_PROTOCOL)	
    with open(os.path.join(output_dir,'kitti_{}_queries_evaluate_new.pickle'.format(sequences[0])), 'wb') as handle:
            pickle.dump(query_sets, handle, protocol=pickle.HIGHEST_PROTOCOL)



def get_positive_dict(ids, output_dir, d_thresh, t_thresh):
    positive = {}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for id in tqdm(ids):
        dataset = get_dataset(id)

        if id not in positive:
            positive[id] = {}

        for t1 in range(len(dataset.timestamps)):
            # for t2 in range(t1, len(dataset.timestamps)):
            for t2 in range(len(dataset.timestamps)):
                if p_dist(dataset.poses[t1], dataset.poses[t2], d_thresh) & t_dist(dataset.timestamps[t1], dataset.timestamps[t2], t_thresh):
                    if t1 not in positive[id]:
                        positive[id][t1] = []
                    positive[id][t1].append(t2)

    with open('{}/positive_sequence_D-{}_T-{}.json'.format(output_dir, d_thresh, t_thresh), 'w') as f:
        json.dump(positive, f)

    return positive



if __name__ == '__main__':
    sequences = ['00']
    get_evaluate_dict(sequences, 'KITTI_evaluate')

