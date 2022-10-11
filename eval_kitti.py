"""
Code taken from https://github.com/cattaneod/PointNetVlad-Pytorch/blob/master/evaluate.py
"""

from SPConvNets.options import opt as opt_oxford
from importlib import import_module
import torch.nn as nn
import numpy as np
import pickle
import torch
import os
import sys
from tqdm import tqdm

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 13})
import config as cfg


def get_sets_dict(filename):
	#[key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}},key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}}, ...}
	with open(filename, 'rb') as handle:
		trajectories = pickle.load(handle)
		print("Trajectories Loaded.")
		return trajectories

def load_pc_file(filename):
    dataset_folder = cfg.DATASET_FOLDER
    comp_name = os.path.join(dataset_folder,filename)
    pc = np.fromfile(comp_name, dtype=np.float32).reshape(-1,4)[:,:3] # xyz

    l = 25
    ind = np.argwhere(pc[:, 0] <= l).reshape(-1)
    pc = pc[ind]
    ind = np.argwhere(pc[:, 0] >= -l).reshape(-1)
    pc = pc[ind]
    ind = np.argwhere(pc[:, 1] <= l).reshape(-1)
    pc = pc[ind]
    ind = np.argwhere(pc[:, 1] >= -l).reshape(-1)
    pc = pc[ind]
    ind = np.argwhere(pc[:, 2] <= l).reshape(-1)
    pc = pc[ind]
    ind = np.argwhere(pc[:, 2] >= -l).reshape(-1)
    pc = pc[ind]
    # sample to cfg.NUM_POINTS #4096
    if pc.shape[0] >= cfg.NUM_POINTS:
        ind = np.random.choice(pc.shape[0], cfg.NUM_POINTS, replace=False)
        pc = pc[ind, :]
    else:
        ind = np.random.choice(pc.shape[0], cfg.NUM_POINTS, replace=True)
        pc = pc[ind, :]
    # rescale to [-1,1] with zero mean
    mean = np.mean(pc, axis=0)
    pc = pc - mean
    scale = np.max(abs(pc))
    pc = pc/scale

    return pc

def load_pc_files(filenames, opt):
	pcs=[]
	for filename in filenames:
		#print(filename)
		pc=load_pc_file(os.path.join(cfg.DATASET_FOLDER, filename))
		if(pc.shape[0]!=cfg.NUM_POINTS):
			continue
		pcs.append(pc)
	pcs=np.array(pcs)
	return pcs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt_oxford.device = device

    # number of rotation anchors. 60 for EPN, 12 for E2PN
    opt_oxford.model.kanchor = 60

    # build model
    if cfg.EVAL_MODEL == 'epn_netvlad':
        from SPConvNets.models.epn_netvlad import EPNNetVLAD
        model = EPNNetVLAD(opt_oxford)
    elif cfg.EVAL_MODEL == 'epn_gem':
        from SPConvNets.models.epn_gem import EPNGeM
        model = EPNGeM(opt_oxford)
    elif cfg.EVAL_MODEL == 'atten_epn_netvlad':
        from SPConvNets.models.atten_epn_netvlad import Atten_EPN_NetVLAD
        model = Atten_EPN_NetVLAD(opt_oxford)
    else:
        print('Model not available')
        exit(0)
        
    # load pretrained file
    if cfg.RESUME_FILENAME.split('.')[1] == 'pth':
        saved_state_dict = torch.load(cfg.RESUME_FILENAME)
    elif cfg.RESUME_FILENAME.split('.')[1] == 'ckpt':
        checkpoint = torch.load(cfg.RESUME_FILENAME)
        saved_state_dict = checkpoint['state_dict']

    model.load_state_dict(saved_state_dict)
    model = nn.DataParallel(model)

    print('Number of Parameters:', count_parameters(model))


    print('average one percent recall', evaluate_model(model, opt_oxford))


def evaluate_model(model, opt):
    # obtain evaluation dataset and query
    DATABASE_SETS = get_sets_dict(cfg.EVAL_DATABASE_FILE)
    QUERY_SETS = get_sets_dict(cfg.EVAL_QUERY_FILE)

    if not os.path.exists(cfg.RESULTS_FOLDER):
        os.mkdir(cfg.RESULTS_FOLDER)

    recall = np.zeros(25)
    count = 0
    similarity = []
    one_percent_recall = []

    DATABASE_VECTORS = []
    QUERY_VECTORS = []

    try:
        # load from file if already evaluated
        DATABASE_VECTORS = np.load(os.path.join(cfg.RESULTS_FOLDER,'database_vectors.npy'), allow_pickle=True)
        QUERY_VECTORS = np.load(os.path.join(cfg.RESULTS_FOLDER, 'query_vectors.npy'), allow_pickle=True)
    except:
        # generate descriptors from input point clouds
        print('Generating descriptors from database sets...')
        for i in tqdm(range(len(DATABASE_SETS))):
            DATABASE_VECTORS.append(get_latent_vectors(model, DATABASE_SETS[i], opt))

        print('Generating descriptors from query sets...')
        for j in tqdm(range(len(QUERY_SETS))):
            QUERY_VECTORS.append(get_latent_vectors(model, QUERY_SETS[j], opt))

        # save descriptors to folder
        np.save(os.path.join(cfg.RESULTS_FOLDER,'database_vectors.npy'), np.array(DATABASE_VECTORS))
        np.save(os.path.join(cfg.RESULTS_FOLDER, 'query_vectors.npy'), np.array(QUERY_VECTORS))

    print('Calculating average recall...')
    for m in tqdm(range(len(DATABASE_SETS))):
        for n in range(len(QUERY_SETS)):
            pair_recall, pair_similarity, pair_opr = get_recall(
                m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            for x in pair_similarity:
                similarity.append(x)

    ave_recall = 0
    if count > 0:
        ave_recall = recall / count

    average_similarity = np.mean(similarity)

    ave_one_percent_recall = np.mean(one_percent_recall)

    with open(cfg.OUTPUT_FILE, "w") as output:
        output.write("Average Recall @N:\n")
        output.write(str(ave_recall))
        output.write("\n\n")
        output.write("Average Similarity:\n")
        output.write(str(average_similarity))
        output.write("\n\n")
        output.write("Average Top 1% Recall:\n")
        output.write(str(ave_one_percent_recall))

    print('ave_recall\n', ave_recall)

    return ave_one_percent_recall


def get_latent_vectors(model, dict_to_process, opt):

    model.eval()
    is_training = False
    eval_file_idxs = np.arange(0, len(dict_to_process.keys()))

    batch_num = cfg.EVAL_BATCH_SIZE * \
        (1 + cfg.EVAL_POSITIVES_PER_QUERY + cfg.EVAL_NEGATIVES_PER_QUERY)
    q_output = []
    total_time = 0
    timer_count = 0
    for q_index in range(len(eval_file_idxs)//batch_num):
        # initialize timer
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        file_indices = eval_file_idxs[q_index *
                                       batch_num:(q_index+1)*(batch_num)]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query_velo"])
        queries = load_pc_files(file_names, opt)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.to(opt.device)
            # inference time
            starter.record()
            out, _ = model(feed_tensor)
            # inference time
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)/1000
            total_time += curr_time
            timer_count += 1

        out = out.detach().cpu().numpy()
        out = np.squeeze(out)

        q_output.append(out)

    q_output = np.array(q_output)
    if(len(q_output) != 0):
        q_output = q_output.reshape(-1, q_output.shape[-1])

    # handle edge case
    index_edge = len(eval_file_idxs) // batch_num * batch_num
    if index_edge < len(dict_to_process.keys()):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        file_indices = eval_file_idxs[index_edge:len(dict_to_process.keys())]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query_velo"])
        queries = load_pc_files(file_names, opt)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.to(opt.device)
            starter.record()
            o1, _ = model(feed_tensor)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)/1000
            total_time += curr_time
            timer_count += 1

        output = o1.detach().cpu().numpy()
        output = np.squeeze(output)
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    print('average inference time = ', total_time, '/', timer_count, '=', total_time/timer_count)
    return q_output


def get_recall(m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS):

    database_output = DATABASE_VECTORS[m]
    queries_output = QUERY_VECTORS[n]

    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        true_neighbors = QUERY_SETS[n][i][m]
        if(len(true_neighbors) == 0):
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(
            np.array([queries_output[i]]),k=num_neighbors)
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if(j == 0):
                    similarity = np.dot(
                        queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))
    recall = (np.cumsum(recall)/float(num_evaluated))

    return recall, top1_similarity_score, one_percent_recall


if __name__ == "__main__":
    evaluate()