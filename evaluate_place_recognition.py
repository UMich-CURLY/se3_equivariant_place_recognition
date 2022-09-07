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
	#returns Nx3 matrix
	pc=np.fromfile(filename, dtype=np.float64)

	if(pc.shape[0]!= cfg.NUM_POINTS*3):
		print("Error in pointcloud shape")
		return np.array([])

	pc=np.reshape(pc,(pc.shape[0]//3,3))
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

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt_oxford.device = device

    # try different number of anchor
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

    print('Number of Parameters', sum(p.numel() for p in model.parameters()))

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
            if (m == n):
                continue
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

    plot_average_recall_curve(ave_recall, ave_one_percent_recall, opt)

    # precision-recall curve
    get_precision_recall_curve(QUERY_SETS, QUERY_VECTORS, DATABASE_VECTORS, opt, ave_one_percent_recall)
    get_f1_recall_curve(opt)

    return ave_one_percent_recall


def plot_average_recall_curve(ave_recall, ave_one_percent_recall, opt):
    index = np.arange(1, 26)
    plt.figure()
    if cfg.EVAL_MODEL == 'epn_netvlad':
        if opt.model.kanchor == 20:
            plt.plot(index, ave_recall, 'b', label='EPN-NetVLAD 20 anchors, AR@1%%=%.2f' % (ave_one_percent_recall))
        else:
            plt.plot(index, ave_recall, 'b', label='EPN-NetVLAD, AR@1%%=%.2f' % (ave_one_percent_recall))
    elif cfg.EVAL_MODEL == 'atten_epn_netvlad':
        if opt.model.kanchor == 20:
            plt.plot(index, ave_recall, 'b', label='EPN-NetVLAD 20 anchors, AR@1%%=%.2f' % (ave_one_percent_recall))
        else:
            plt.plot(index, ave_recall, 'b', label='EPN-NetVLAD Attentive Downsample, AR@1%%=%.2f' % (ave_one_percent_recall))

    else:
        print('Model not available')

    # plot average recall curve for baselines, optional
    try:
        for baseline_result_folder, baseline_name, plot_style in zip([cfg.BASELINE_RESULT_FOLDER, \
                                                                      cfg.POINTNETVLAD_RESULT_FOLDER, \
                                                                      cfg.SCANCONTEXT_RESULT_FOLDER, \
                                                                      cfg.M2DP_RESULT_FOLDER \
                                                                     ], 
                                                                      ['EPN-NetVLAD 60 anchors', \
                                                                       'PointNetVLAD', 'Scan Context', 'M2DP'],
                                                                      ['c', 'k--', 'm-.', 'g:']):

            ave_one_percent_recall_baseline = None
            with open(os.path.join(baseline_result_folder, 'results.txt'), "r") as baseline_result_file:
                ave_one_percent_recall_baseline = float(baseline_result_file.readlines()[-1])
            ave_recall_baseline = ''
            with open(os.path.join(baseline_result_folder, 'results.txt'), "r") as baseline_result_file:
                ave_recall_baseline_temp = baseline_result_file.readlines()[1:6]
                for i in range(len(ave_recall_baseline_temp)):
                    ave_recall_baseline_temp[i] = ave_recall_baseline_temp[i].replace('[', '')
                    ave_recall_baseline_temp[i] = ave_recall_baseline_temp[i].replace(']', '')
                    ave_recall_baseline_temp[i] = ave_recall_baseline_temp[i].replace('\n', '')
                    ave_recall_baseline = ave_recall_baseline + ave_recall_baseline_temp[i]
                ave_recall_baseline = np.array(ave_recall_baseline.split())
                ave_recall_baseline = np.asarray(ave_recall_baseline, dtype = float)
            plt.plot(index, ave_recall_baseline, plot_style, label=baseline_name+', AR@1%%=%.2f' % (ave_one_percent_recall_baseline))
    except:
        print('error plotting baselines curve')
    
    plt.title("Average recall @N Curve")
    plt.xlabel('in top N')
    plt.ylabel('Average recall @N')
    plt.xlim(-1,26)
    plt.ylim(0,1.1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.RESULTS_FOLDER, "average_recall_curve.png"))
    print('Average recall curve is saved at:', os.path.join(cfg.RESULTS_FOLDER, "average_recall_curve.png"))


def get_precision_recall_curve(QUERY_SETS, QUERY_VECTORS, DATABASE_VECTORS, opt, ave_one_percent_recall):
    try:
        precision = np.load(os.path.join(cfg.RESULTS_FOLDER, 'precision.npy'))
        recall = np.load(os.path.join(cfg.RESULTS_FOLDER, 'recall.npy'))
    except:
        y_true = []
        y_predicted = []

        for q in range(len(QUERY_SETS)):
            for d in range(len(QUERY_SETS)):
                if (q==d):
                    continue

                database_nbrs = KDTree(DATABASE_VECTORS[d])

                for i in range(len(QUERY_SETS[q])):
                    true_neighbors = QUERY_SETS[q][i][d]
                    if(len(true_neighbors)==0):
                        continue
                    distances, indices = database_nbrs.query(np.array([QUERY_VECTORS[q][i]]))
                    current_y_true = 0
                    current_y_predicted = 0
                    for j in range(len(indices[0])):
                        if indices[0][j] in true_neighbors:
                            # predicted neighbor is correct
                            current_y_true = 1
                        current_y_predicted_temp = np.dot(QUERY_VECTORS[q][i], DATABASE_VECTORS[d][indices[0][j]]) / \
                                                        (np.linalg.norm(QUERY_VECTORS[q][i]) * np.linalg.norm(DATABASE_VECTORS[d][indices[0][j]]))
                        # take prediction similarity that is the highest amoung neighbors
                        if current_y_predicted_temp > current_y_predicted:
                            current_y_predicted = current_y_predicted_temp
                    # loop or not
                    y_true.append(current_y_true)

                    # similarity
                    y_predicted.append(current_y_predicted)
        
        np.set_printoptions(threshold=sys.maxsize)

        precision, recall, thresholds = precision_recall_curve(y_true, y_predicted)

        np.set_printoptions(threshold=1000)

        np.save(os.path.join(cfg.RESULTS_FOLDER, 'precision.npy'), np.array(precision))
        np.save(os.path.join(cfg.RESULTS_FOLDER, 'recall.npy'), np.array(recall))

    # Plot Precision-recall curve
    plt.figure()
    if cfg.EVAL_MODEL =='epn_netvlad':
        if opt.model.kanchor == 20:
            plt.plot(recall, precision, 'b', label='EPN-NetVLAD 20 anchors')
        else:
            plt.plot(recall, precision, 'b', label='EPN-NetVLAD')
    elif cfg.EVAL_MODEL =='atten_epn_netvlad':
        if opt.model.kanchor == 20:
            plt.plot(recall, precision, 'b', label='EPN-NetVLAD Attentive Downsample 20 anchors')
        else:
            plt.plot(recall, precision, 'b', label='EPN-NetVLAD Attentive Downsample')
    else:
        print('Model unavailable')

    # plot precision-recall curve for baselines, optional
    try:
        for baseline_result_folder, baseline_name, plot_style in zip([cfg.BASELINE_RESULT_FOLDER, \
                                                                      cfg.POINTNETVLAD_RESULT_FOLDER, \
                                                                      cfg.SCANCONTEXT_RESULT_FOLDER, \
                                                                      cfg.M2DP_RESULT_FOLDER \
                                                                     ], 
                                                                      ['EPN-NetVLAD Attentive Downsampling 60 anchors', \
                                                                       'PointNetVLAD', 'Scan Context', 'M2DP'],
                                                                      ['c', 'k--', 'm-.', 'g:']):

            precision_baseline = np.load(os.path.join(baseline_result_folder, 'precision.npy'))
            recall_baseline = np.load(os.path.join(baseline_result_folder, 'recall.npy'))
            plt.plot(recall_baseline, precision_baseline, plot_style, label=baseline_name)
    except:
        print('error plotting baselines curve')
    
    plt.title("Precision-recall Curve")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0,1.1)
    plt.ylim(0,1.1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.RESULTS_FOLDER, "precision_recall_oxford.png"))
    print('Precision-recall curve is saved at:', os.path.join(cfg.RESULTS_FOLDER, "precision_recall_oxford.png"))


def get_f1_recall_curve(opt):
    precision = np.load(os.path.join(cfg.RESULTS_FOLDER, 'precision.npy'))
    recall = np.load(os.path.join(cfg.RESULTS_FOLDER, 'recall.npy'))
    f1 = 2 * precision * recall / (precision + recall)
    np.save(os.path.join(cfg.RESULTS_FOLDER, 'f1.npy'), np.array(f1))

    # Plot F1-recall curve
    plt.figure()
    if cfg.EVAL_MODEL == 'epn_netvlad':
        if opt.model.kanchor == 20:
            plt.plot(recall, f1, 'b', label='EPN-NetVLAD 20 anchors')
        else:
            plt.plot(recall, f1, 'b', label='EPN-NetVLAD')
    elif cfg.EVAL_MODEL =='atten_epn_netvlad':
        if opt.model.kanchor == 20:
            plt.plot(recall, f1, 'b', label='EPN-NetVLAD Attentive Downsampling 20 anchors')
        else:
            plt.plot(recall, f1, 'b', label='EPN-NetVLAD Attentive Downsampling')

    # plot f1-recall curve for baselines, optional
    try:
        for baseline_result_folder, baseline_name, plot_style in zip([cfg.BASELINE_RESULT_FOLDER, \
                                                                      cfg.POINTNETVLAD_RESULT_FOLDER, \
                                                                      cfg.SCANCONTEXT_RESULT_FOLDER, \
                                                                      cfg.M2DP_RESULT_FOLDER \
                                                                     ], 
                                                                      ['EPN-NetVLAD Attentive Downsampling 60 anchors', \
                                                                       'PointNetVLAD', 'Scan Context', 'M2DP'],
                                                                      ['c', 'k--', 'm-.', 'g:']):

            f1_baseline = np.load(os.path.join(baseline_result_folder, 'f1.npy'))
            recall_baseline = np.load(os.path.join(baseline_result_folder, 'recall.npy'))
            plt.plot(recall_baseline, f1_baseline, plot_style, label=baseline_name)
    except:
        print('error plotting baselines curve')

    plt.title("F1-recall Curve")
    plt.xlabel('Recall')
    plt.ylabel('F1 Score')
    plt.xlim(0,1.1)
    plt.ylim(0,1.1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.RESULTS_FOLDER, "f1_recall_oxford.png"))
    print('F1-recall curve is saved at:', os.path.join(cfg.RESULTS_FOLDER, "f1_recall_oxford.png"))


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
            file_names.append(dict_to_process[index]["query"])
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
            file_names.append(dict_to_process[index]["query"])
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