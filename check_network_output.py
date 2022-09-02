"""
Test output feature from the network
"""
# Local E2PN package
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'vgtk') )

import numpy as np
import socket
import importlib
import os
import sys

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from SPConvNets.options import opt as opt_oxford
from importlib import import_module
import config as cfg


def test_global_features():
    def load_pcd_file(filename):
        pc=np.load(filename)
        return pc

    def load_pc_file(filename):
        #returns Nx3 matrix
        pc=np.fromfile(filename, dtype=np.float64)
        pc=np.reshape(pc,(pc.shape[0]//3,3))
        return pc

    def load_model(opt):
        # build model
        if cfg.EVAL_MODEL == 'e2pn_netvlad': 
            from SPConvNets.models.e2pn_netvlad import E2PNNetVLAD
            model = E2PNNetVLAD(opt_oxford)
        elif cfg.EVAL_MODEL == 'e2pn_gem': 
            from SPConvNets.models.e2pn_gem import E2PNGeM
            model = E2PNGeM(opt_oxford)
        
        # load pretrained weight
        if opt.resume_path.split('.')[1] == 'pth':
            saved_state_dict = torch.load(opt.resume_path)
        elif opt.resume_path.split('.')[1] == 'ckpt':
            checkpoint = torch.load(opt.resume_path)
            saved_state_dict = checkpoint['state_dict']    
        model.load_state_dict(saved_state_dict)
        model = nn.DataParallel(model)

        return model
    
    def get_global_descriptor(model, network_input, opt):
        network_input = network_input.reshape((1, network_input.shape[0], network_input.shape[1]))
        network_input = torch.Tensor(network_input).float().cuda()

        # get output features from the model
        model = model.eval()
        network_output, frontend_output = model(network_input)
        
        frontend_output = frontend_output.detach().cpu().numpy() #[:, :, 0].reshape((1024,))
        print('frontend_output', frontend_output.shape)

        frontend_output = frontend_output[:,0,:]
        frontend_output = frontend_output.reshape((-1,))

        # tensor to numpy
        network_output = network_output.detach().cpu().numpy()[0, :]
        network_output = network_output.astype(np.double)
        frontend_output = frontend_output.astype(np.double)
        
        print('network_output', network_output.shape)
        return network_output, frontend_output

    opt_oxford.device = torch.device('cuda')

    # IO
    opt_oxford.model.input_num = cfg.NUM_POINTS #4096

    # place recognition
    opt_oxford.num_points = opt_oxford.model.input_num
    opt_oxford.pos_per_query = 1
    opt_oxford.neg_per_query = 1

    # param tuning
    # opt_oxford.model.search_radius = 0.35 #0.4
    # opt_oxford.model.initial_radius_ratio = 0.2 #0.2
    # opt_oxford.model.sampling_ratio = 0.8 #0.8

    # pretrained weight
    # opt_oxford.resume_path = 'pretrained_model/epn_netvlad_seq567_64_maxlinear1024.ckpt'
    opt_oxford.resume_path = 'pretrained_model/e2pn_netvlad_32_64_trainall.ckpt'
    
    # input file
    input_folder = 'results/test_network_output/'
    input_filename = os.path.join(input_folder, '0_anchor.npy')
    input_pointcloud = load_pcd_file(os.path.join(input_filename))
    print('input_pointcloud', input_pointcloud.shape)
    print('input_pointcloud', input_pointcloud[0,:])

    # rotate and translate
    rotated_pointcloud = load_pcd_file('results/test_network_output/0_rotated.npy')
    translated_pointcloud = load_pcd_file('results/test_network_output/0_translated.npy')
    rotated_translated_pointcloud = load_pcd_file('results/test_network_output/0_rotated_translated.npy')
    translated_rotated_pointcloud = load_pcd_file('results/test_network_output/0_translated_rotated.npy')
    
    # anchor, positive, negative
    positive_pointcloud = load_pcd_file('results/test_network_output/0_positive.npy')
    negative_pointcloud = load_pcd_file('results/test_network_output/0_negative.npy')
    # input_pointcloud = load_pc_file('/home/cel/data/benchmark_datasets/oxford/2014-11-18-13-20-12/pointcloud_20m/1416316967238675.bin')
    # positive_pointcloud = load_pc_file('/home/cel/data/benchmark_datasets/oxford/2014-11-18-13-20-12/pointcloud_20m_tran_1/1416316967238675.bin')
    # negative_pointcloud = load_pc_file('/home/cel/data/benchmark_datasets/oxford/2014-11-18-13-20-12/pointcloud_20m/1416317406139905.bin')

    model = load_model(opt_oxford)
    
    with torch.no_grad():          
        # generate descriptors from point clouds
        output_descriptor, output_pointnet = get_global_descriptor(model, input_pointcloud, opt_oxford)
        rotated_descriptor, rotated_pointnet = get_global_descriptor(model, rotated_pointcloud, opt_oxford)
        translated_descriptor, translated_pointnet = get_global_descriptor(model, translated_pointcloud, opt_oxford)
        rotated_translated_descriptor, rotated_translated_pointnet = get_global_descriptor(model, rotated_translated_pointcloud, opt_oxford)
        translated_rotated_descriptor, translated_rotated_pointnet = get_global_descriptor(model, translated_rotated_pointcloud, opt_oxford)
        # positives and negatives
        positive_descriptor, positive_frontend = get_global_descriptor(model, positive_pointcloud, opt_oxford)
        negative_descriptor, negative_frontend = get_global_descriptor(model, negative_pointcloud, opt_oxford)

    # calculate similarity
    similarity_rotate = np.absolute(np.dot(output_descriptor, rotated_descriptor)/(np.linalg.norm(output_descriptor)*np.linalg.norm(rotated_descriptor)))
    similarity_translate = np.absolute(np.dot(output_descriptor, translated_descriptor)/(np.linalg.norm(output_descriptor)*np.linalg.norm(translated_descriptor)))
    similarity_rotated_translated = np.absolute(np.dot(output_descriptor, rotated_translated_descriptor)/(np.linalg.norm(output_descriptor)*np.linalg.norm(rotated_translated_descriptor)))
    similarity_translated_rotated = np.absolute(np.dot(output_descriptor, translated_rotated_descriptor)/(np.linalg.norm(output_descriptor)*np.linalg.norm(translated_rotated_descriptor)))
    similarity_positive = np.absolute(np.dot(output_descriptor, positive_descriptor)/(np.linalg.norm(output_descriptor)*np.linalg.norm(positive_descriptor)))
    similarity_negative = np.absolute(np.dot(output_descriptor, negative_descriptor)/(np.linalg.norm(output_descriptor)*np.linalg.norm(negative_descriptor)))
    
    # calculate similarity
    similarity_pointnet_rotate = np.absolute(np.dot(output_pointnet, rotated_pointnet)/(np.linalg.norm(output_pointnet)*np.linalg.norm(rotated_pointnet)))
    similarity_pointnet_translate = np.absolute(np.dot(output_pointnet, translated_pointnet)/(np.linalg.norm(output_pointnet)*np.linalg.norm(translated_pointnet)))
    similarity_pointnet_rotated_translated = np.absolute(np.dot(output_pointnet, rotated_translated_pointnet)/(np.linalg.norm(output_pointnet)*np.linalg.norm(rotated_translated_pointnet)))
    similarity_pointnet_translated_rotated = np.absolute(np.dot(output_pointnet, translated_rotated_pointnet)/(np.linalg.norm(output_pointnet)*np.linalg.norm(translated_rotated_pointnet)))
    similarity_frontend_positive = np.absolute(np.dot(output_pointnet, positive_frontend)/(np.linalg.norm(output_pointnet)*np.linalg.norm(positive_frontend)))
    similarity_frontend_negative = np.absolute(np.dot(output_pointnet, negative_frontend)/(np.linalg.norm(output_pointnet)*np.linalg.norm(negative_frontend)))
    
    # visualize input point clouds
    fig = plt.figure()
    
    ax = fig.add_subplot(2, 3, 1, projection='3d')
    ax.scatter(input_pointcloud[:, 0], input_pointcloud[:, 1], input_pointcloud[:, 2], c='C0', label='original point cloud')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('original')

    ax = fig.add_subplot(2, 3, 2, projection='3d')
    ax.scatter(rotated_pointcloud[:, 0], rotated_pointcloud[:, 1], rotated_pointcloud[:, 2], c='C1', label='rotated point cloud')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('rotated')

    ax = fig.add_subplot(2, 3, 3, projection='3d')
    ax.scatter(translated_pointcloud[:, 0], translated_pointcloud[:, 1], translated_pointcloud[:, 2], c='C2', label='translated point cloud')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('translated')

    ax = fig.add_subplot(2, 3, 4, projection='3d')
    ax.scatter(rotated_translated_pointcloud[:, 0], rotated_translated_pointcloud[:, 1], rotated_translated_pointcloud[:, 2], c='C4', label='translated point cloud')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('rotated then translated')

    ax = fig.add_subplot(2, 3, 6, projection='3d')
    ax.scatter(translated_rotated_pointcloud[:, 0], translated_rotated_pointcloud[:, 1], translated_rotated_pointcloud[:, 2], c='C5', label='translated point cloud')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('translated then rotated')

    fig.suptitle('Input Point Clouds')
    plt.savefig(os.path.join(input_folder, '0_pointclouds_transform.png'))

    # visualize descriptors (rotated, translated)
    x_index = np.arange(output_descriptor.size)
    plt.figure()
    plt.plot(x_index, output_descriptor, label='original')
    plt.plot(x_index, rotated_descriptor, '--', label='rotated, similarity=%.2f' % (similarity_rotate))
    plt.plot(x_index, translated_descriptor, ':', label='translated, similarity=%.2f' % (similarity_translate))
    plt.plot(x_index, rotated_translated_descriptor, '-.', c='C4', label='rotated then translated, similarity=%.2f' % (similarity_rotated_translated))
    plt.plot(x_index, translated_rotated_descriptor, '-.', c='C5', label='translated then rotated, similarity=%.2f' % (similarity_translated_rotated))
    plt.title('Global Descriptors')
    plt.xlabel('index')
    plt.ylabel('feature value')
    plt.legend()
    plt.savefig(os.path.join(input_folder, '0_'+cfg.EVAL_MODEL+'_global_descriptor_transform.png'))

    # visualize frontend features of first point
    x_index2 = np.arange(output_pointnet.size)
    plt.figure()
    plt.plot(x_index2, output_pointnet, label='original')
    plt.plot(x_index2, rotated_pointnet, '--', label='rotated, similarity=%.2f' % (similarity_pointnet_rotate))
    plt.plot(x_index2, translated_pointnet, ':', label='translated, similarity=%.2f' % (similarity_pointnet_translate))
    plt.plot(x_index2, rotated_translated_pointnet, '-.', c='C4', label='rotated then translated, similarity=%.2f' % (similarity_pointnet_rotated_translated))
    plt.plot(x_index2, translated_rotated_pointnet, '-.', c='C5', label='translated then rotated, similarity=%.2f' % (similarity_pointnet_translated_rotated))
    plt.xlabel('index')
    plt.ylabel('feature value')
    plt.legend()
    plt.title('Local Features')
    plt.savefig(os.path.join(input_folder, '0_'+cfg.EVAL_MODEL+'_loacl_features_tranform.png'))

    # visualize descriptors (positive, negative)
    plt.figure()
    plt.plot(x_index, output_descriptor, label='anchor')
    plt.plot(x_index, positive_descriptor, '--', label='positive, similarity=%.2f' % (similarity_positive))
    plt.plot(x_index, negative_descriptor, ':', label='negative, similarity=%.2f' % (similarity_negative))
    plt.title('Global Descriptors')
    plt.xlabel('index')
    plt.ylabel('feature value')
    plt.legend()
    plt.savefig(os.path.join(input_folder, '0_'+cfg.EVAL_MODEL+'_global_descriptor_posneg.png'))

    # visualize features of positives and negatives
    plt.figure()
    plt.plot(x_index2, output_pointnet, label='original')
    plt.plot(x_index2, positive_frontend, '--', label='positive, similarity=%.2f' % (similarity_frontend_positive))
    plt.plot(x_index2, negative_frontend, ':', label='negative, similarity=%.2f' % (similarity_frontend_negative))
    plt.xlabel('index')
    plt.ylabel('feature value')
    plt.legend()
    plt.title('Local Features')
    plt.savefig(os.path.join(input_folder, '0_'+cfg.EVAL_MODEL+'_loacl_features_posneg.png'))


if __name__ == "__main__":
    test_global_features()

