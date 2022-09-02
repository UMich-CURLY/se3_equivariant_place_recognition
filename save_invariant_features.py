"""
Save output features from the network to pcd
"""
# Local E2PN package
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'vgtk') )


def equivariant_features_to_numpy():
    import numpy as np
    import socket
    import importlib
    import os
    import sys

    import matplotlib.pyplot as plt

    import torch
    import torch.nn as nn

    from SPConvNets.options import opt as opt_oxford
    from importlib import import_module
    import config as cfg

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
        from SPConvNets.models.e2pn_netvlad import E2PNNetVLAD
        model = E2PNNetVLAD(opt_oxford)
        
        # load pretrained weight
        if opt.resume_path.split('.')[1] == 'pth':
            saved_state_dict = torch.load(opt.resume_path)
        elif opt.resume_path.split('.')[1] == 'ckpt':
            checkpoint = torch.load(opt.resume_path)
            saved_state_dict = checkpoint['state_dict']    
        model.load_state_dict(saved_state_dict)
        model = nn.DataParallel(model)

        return model
    
    def get_se3_equivariant_local_feature(model, network_input):
        network_input = network_input.reshape((1, network_input.shape[0], network_input.shape[1]))
        network_input = torch.Tensor(network_input).float().cuda()

        # get output features from the model
        model = model.eval()
        _, invariant_feature = model(network_input)
        
        # downsampled_points = downsampled_points.detach().cpu().numpy()[0]

        invariant_feature = invariant_feature.detach().cpu().numpy()[0]
        
        return invariant_feature

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt_oxford.device = device

    # try different number of anchor
    opt_oxford.model.kanchor = 12

    # place recognition
    opt_oxford.pos_per_query = 1
    opt_oxford.neg_per_query = 1


    # pretrained weight
    opt_oxford.resume_path = 'pretrained_model/e2pn_netvlad_32_64_lf256_3seq.ckpt'
    
    # input file
    input_folder = 'results/test_network_output/'
    input_filename = os.path.join(input_folder, '0_anchor.npy')
    input_pointcloud = load_pcd_file(os.path.join(input_filename))
    print('input_pointcloud', input_pointcloud.shape)

    # transformation
    transformation_matrix = np.array([[0.2919266, -0.4546487,  0.8414710, 0.5], \
                                      [0.8372224, -0.3038967, -0.4546487, -0.2], \
                                      [0.4624257,  0.8372224,  0.2919266, 0.1], \
                                      [0, 0, 0, 1]])
    transformed_pointcloud = transformation_matrix @ np.vstack((input_pointcloud.T, np.ones((1, input_pointcloud.shape[0]))))
    transformed_pointcloud = transformed_pointcloud[:3, :].T
    print('transformed_pointcloud', transformed_pointcloud.shape)


    model = load_model(opt_oxford)
    
    with torch.no_grad():          
        # generate local features from point clouds
        invariant_feature = get_se3_equivariant_local_feature(model, input_pointcloud)
        transformed_invariant_feature = get_se3_equivariant_local_feature(model, transformed_pointcloud)
    
        # downsmapled_pointcloud, invariant_feature = get_se3_equivariant_local_feature(model, input_pointcloud)
        # downsmapled_transformed_pointcloud, transformed_invariant_feature = get_se3_equivariant_local_feature(model, transformed_pointcloud)
    
    # print('downsmapled_pointcloud', downsmapled_pointcloud.shape)
    # print('downsmapled_transformed_pointcloud', downsmapled_transformed_pointcloud.shape)
        
    # save point clouds as numpy file
    print('input_pointcloud', input_pointcloud.shape, input_pointcloud.dtype)
    print('transformed_pointcloud', transformed_pointcloud.shape, transformed_pointcloud.dtype)
    input_pointcloud = input_pointcloud.astype(np.float64)
    transformed_pointcloud = transformed_pointcloud.astype(np.float64)
    np.save("results/test_network_output/cvo_original_pointcloud_points.npy", input_pointcloud)
    np.save("results/test_network_output/cvo_transformed_pointcloud_points.npy", transformed_pointcloud)
    transformed_pointcloud.astype(np.float64).tofile("results/test_network_output/cvo_transformed_pointcloud_points.bin")


    # save local feartures as numpy file
    print('invariant_feature', invariant_feature.shape)
    print('transformed_invariant_feature', transformed_invariant_feature.shape)
    # np.save("results/test_network_output/cvo_original_pointcloud_points.npy", downsmapled_pointcloud)
    np.save("results/test_network_output/cvo_orginal_pointcloud_invariant_feature.npy", invariant_feature)
    # np.save("results/test_network_output/cvo_transformed_pointcloud_points.npy", downsmapled_transformed_pointcloud)
    np.save("results/test_network_output/cvo_transformed_pointcloud_invariant_feature.npy", transformed_invariant_feature)

    # save as .bin files
    invariant_feature_float = np.array(invariant_feature,'float32')
    invariant_file = open("results/test_network_output/cvo_original_pointcloud_invariant_feature.bin", 'wb')
    invariant_feature_float.tofile(invariant_file)
    invariant_file.close()

    transformed_invariant_feature_float = np.array(transformed_invariant_feature,'float32')
    transformed_invariant_file = open("results/test_network_output/cvo_transformed_pointcloud_invariant_feature.bin", 'wb')
    transformed_invariant_feature_float.tofile(transformed_invariant_file)
    transformed_invariant_file.close()

    # save as .txt files
    np.savetxt("results/test_network_output/cvo_original_pointcloud_invariant_feature.txt", invariant_feature, delimiter=',')
    np.savetxt("results/test_network_output/cvo_transformed_pointcloud_invariant_feature.txt", transformed_invariant_feature, delimiter=',')

def numpy_to_pcd():
    import numpy as np
    from pypcd import pypcd

    input_pointcloud = np.load("results/test_network_output/cvo_original_pointcloud_points.npy")
    print('input_pointcloud', input_pointcloud.shape, input_pointcloud.dtype)
    print(input_pointcloud)
    input_pcd = pypcd.make_xyz_point_cloud(input_pointcloud)
    input_pcd.save_pcd("results/test_network_output/cvo_original_pointcloud_points.pcd")

    transformed_pointcloud = np.fromfile("results/test_network_output/cvo_transformed_pointcloud_points.bin", dtype=np.float64)
    transformed_pointcloud = np.reshape(transformed_pointcloud,(transformed_pointcloud.shape[0]//3,3))
    # transformed_pointcloud = np.load("results/test_network_output/cvo_transformed_pointcloud_points.npy")
    print('transformed_pointcloud', transformed_pointcloud.shape, transformed_pointcloud.dtype)
    print(transformed_pointcloud)
    transformed_pcd = pypcd.make_xyz_point_cloud(transformed_pointcloud)
    transformed_pcd.save_pcd("results/test_network_output/cvo_transformed_pointcloud_points.pcd")


if __name__ == "__main__":
    # equivariant_features_to_numpy()
    numpy_to_pcd()

