from cProfile import label
import numpy as np

def load_pc_file(filename):
    #returns Nx3 matrix
    pc=np.fromfile(filename, dtype=np.float64)
    pc=np.reshape(pc,(pc.shape[0]//3,3))
    return pc

def save_transformed_pointcloud():
    from scipy.spatial.transform import Rotation as R
    from pypcd import pypcd

    # IO setting
    filename = "/home/cel/DockerFolder/data/benchmark_datasets/oxford/2014-11-14-16-34-33/pointcloud_20m/1415985064199113.bin"
    output_folder = "/home/cel/DockerFolder/code/EPN-NetVLAD/results/invariant_test/"
    
    # load point cloud
    original_pointcloud = load_pc_file(filename)
    # print('original_pointcloud', original_pointcloud.shape)

    # save this one
    original_pcd = pypcd.make_xyz_point_cloud(original_pointcloud)
    original_pcd.save_pcd(output_folder+"pcd_pointclouds/"+"original_pointcloud_seq5_3.pcd")
    np.save(output_folder+"npy_pointclouds/"+"original_pointcloud_seq5_3.npy", original_pointcloud)

    # make it to homogeneous
    original_pointcloud_homogeneous = np.hstack((original_pointcloud, np.ones((original_pointcloud.shape[0], 1))))
    # print('original_pointcloud_homogeneous', original_pointcloud_homogeneous.shape)

    # transformation
    axis_list = ['x', 'y', 'z']
    for axis in range(3): # rotating in different axis (x, y, z)
        for angle in range(0, 365, 5): # different angle in step of 5 degrees
            # rotation matrix
            r = R.from_euler(axis_list[axis], angle, degrees=True)
            rotation_matrix = np.vstack((np.hstack((r.as_matrix(), np.zeros((3, 1)))), np.array([0., 0., 0., 1.]))).astype(np.float64)

            # rotate point cloud
            rotated_pointcloud_homogeneous = rotation_matrix @ original_pointcloud_homogeneous.T
            rotated_pointcloud = rotated_pointcloud_homogeneous[:3, :].T
            rotated_pointcloud = rotated_pointcloud.astype(np.float64)

            # save rotated point cloud
            rotated_pointcloud.astype(np.float64).tofile(output_folder+"bin_pointclouds/"+"rotated_"+axis_list[axis]+"_"+str(angle)+".bin")
            rotated_pointcloud = np.fromfile(output_folder+"bin_pointclouds/"+"rotated_"+axis_list[axis]+"_"+str(angle)+".bin", dtype=np.float64)
            rotated_pointcloud = np.reshape(rotated_pointcloud,(rotated_pointcloud.shape[0]//3,3))
            rotated_pcd = pypcd.make_xyz_point_cloud(rotated_pointcloud)
            rotated_pcd.save_pcd(output_folder+"pcd_pointclouds/"+"rotated_"+axis_list[axis]+"_"+str(angle)+".pcd")
            np.save(output_folder+"npy_pointclouds/"+"rotated_"+axis_list[axis]+"_"+str(angle)+".npy", rotated_pointcloud)
        
        for trans in np.arange(-100, 105, 5): # different value for translation
            # add translation
            trans_pointcloud = np.zeros_like(original_pointcloud)
            trans_pointcloud[:, axis] += trans/100
            translated_pointcloud = original_pointcloud + trans_pointcloud

            # save translated point cloud 
            translated_pcd = pypcd.make_xyz_point_cloud(translated_pointcloud)
            translated_pcd.save_pcd(output_folder+"pcd_pointclouds/"+"translated_"+axis_list[axis]+"_"+str(trans/100)+".pcd")
            np.save(output_folder+"npy_pointclouds/"+"translated_"+axis_list[axis]+"_"+str(trans/100)+".npy", translated_pointcloud)

def visualization_o3d():
    import open3d as o3d
    import time

    folder_path = "/home/cel/DockerFolder/code/EPN-NetVLAD/results/invariant_test/"

    axis_list = ['x', 'y', 'z']
    for axis in range(3): # rotating in different axis (x, y, z)
        for angle in range(0, 365, 5): # different angle in step of 5 degrees
            filename = folder_path+"npy_pointclouds/"+"rotated_"+axis_list[axis]+"_"+str(angle)+".npy"
            pointcloud_xyz = np.load(filename)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud_xyz)

            # save screenshot individually
            vis = o3d.visualization.Visualizer()
            # vis.create_window(visible=False)
            vis.create_window()
            vis.add_geometry(pcd)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(folder_path+"png_pointclouds/"+"rotated_"+axis_list[axis]+"_"+str(angle)+".png")
            vis.destroy_window()
            time.sleep(5)


def visualization_matplotlib():
    import matplotlib.pyplot as plt

    folder_path = "/home/cel/DockerFolder/code/EPN-NetVLAD/results/invariant_test/"

    original_pointcloud = np.load(folder_path+"npy_pointclouds/"+"original_pointcloud_seq5_3.npy", )


    axis_list = ['x', 'y', 'z']
    for axis in range(3): # rotating in different axis (x, y, z)
        for angle in range(0, 365, 5): # different angle in step of 5 degrees
            rotated_filename = folder_path+"npy_pointclouds/"+"rotated_"+axis_list[axis]+"_"+str(angle)+".npy"
            rotated_pointcloud = np.load(rotated_filename)
            rotated_png_savename = folder_path+"png_pointclouds_withref/"+"rotated_"+axis_list[axis]+"_"+str(angle)+".png"

            # visualize input point clouds
            fig = plt.figure()
        
            ax = plt.axes(projection='3d')
            ax.grid(False)
            ax.scatter(original_pointcloud[:, 0], original_pointcloud[:, 1], original_pointcloud[:, 2], c='#C5C9C7', marker=".", alpha=0.3)
            ax.scatter(rotated_pointcloud[:, 0], rotated_pointcloud[:, 1], rotated_pointcloud[:, 2], c=rotated_pointcloud[:, 2], marker=".")
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_title('point cloud rotated around %s-axis for %3d degrees' % (axis_list[axis], angle))

            plt.savefig(rotated_png_savename)
            plt.close()

        for trans in np.arange(-100, 105, 5): # different value for translation
            translated_filename = folder_path+"npy_pointclouds/"+"translated_"+axis_list[axis]+"_"+str(trans/100)+".npy"
            translated_pointcloud = np.load(translated_filename)
            translated_png_savename = folder_path+"png_pointclouds_withref/"+"translated_"+axis_list[axis]+"_"+str(trans/100)+".png"

            # visualize input point clouds
            fig = plt.figure()
        
            ax = plt.axes(projection='3d')
            ax.grid(False)
            ax.scatter(original_pointcloud[:, 0], original_pointcloud[:, 1], original_pointcloud[:, 2], c='#C5C9C7', marker=".", alpha=0.3)
            ax.scatter(translated_pointcloud[:, 0], translated_pointcloud[:, 1], translated_pointcloud[:, 2], c=translated_pointcloud[:, 2], marker=".")
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_title('point cloud translated in %s-axis for %.2f' % (axis_list[axis], trans/100))

            plt.savefig(translated_png_savename)
            plt.close()



def generate_descriptors(model_name):
    from SPConvNets.options import opt as opt_oxford
    import torch
    import torch.nn as nn
    import os
    from tqdm import tqdm

    def load_model(EVAL_MODEL, opt):
        # build model
        if EVAL_MODEL == 'epn_netvlad':
            from SPConvNets.models.epn_netvlad import EPNNetVLAD
            model = EPNNetVLAD(opt)
        elif EVAL_MODEL == 'epn_gem':
            from SPConvNets.models.epn_gem import EPNGeM
            model = EPNGeM(opt)
        elif EVAL_MODEL == 'atten_epn_netvlad':
            from SPConvNets.models.atten_epn_netvlad import Atten_EPN_NetVLAD
            model = Atten_EPN_NetVLAD(opt)
        
        # load pretrained weight
        if opt.resume_path.split('.')[1] == 'pth':
            saved_state_dict = torch.load(opt.resume_path)
        elif opt.resume_path.split('.')[1] == 'ckpt':
            checkpoint = torch.load(opt.resume_path)
            saved_state_dict = checkpoint['state_dict']    
        model.load_state_dict(saved_state_dict)
        model = nn.DataParallel(model)

        return model

    def get_global_descriptor(model, network_input):
        with torch.no_grad():    
            network_input = network_input.reshape((1, network_input.shape[0], network_input.shape[1]))
            network_input = torch.Tensor(network_input).float().cuda()

            # get output features from the model
            model = model.eval()
            network_output, _ = model(network_input)

            # tensor to numpy
            network_output = network_output.detach().cpu().numpy().reshape(-1)
            network_output = network_output.astype(np.double)
        
        return network_output


    opt_oxford.device = torch.device('cuda')

    opt_oxford.pos_per_query = 1
    opt_oxford.neg_per_query = 1

    # pretrained weight
    opt_oxford.resume_path = 'pretrained_model/epn_gem_train3seq.ckpt'
    model = load_model(model_name, opt_oxford)

    # input file
    folder_path = "/home/cel/code/EPN-NetVLAD/results/invariant_test/"
    if not os.path.exists(folder_path+model_name+"_descriptors/"):
        os.mkdir(folder_path+model_name+"_descriptors/")
          
    # generate descriptors from original point clouds
    original_pointcloud = np.load(folder_path+"npy_pointclouds/"+"original_pointcloud_seq5_3.npy", )
    original_descriptor = get_global_descriptor(model, original_pointcloud)
    np.save(folder_path+model_name+"_descriptors/original_pointcloud_seq5_3.npy", original_descriptor)

    axis_list = ['x', 'y', 'z']
    for axis in range(3): # rotating in different axis (x, y, z)
        for angle in tqdm(range(0, 365, 5)): # different angle in step of 5 degrees
            rotated_filename = folder_path+"npy_pointclouds/"+"rotated_"+axis_list[axis]+"_"+str(angle)+".npy"
            rotated_pointcloud = np.load(rotated_filename)

            # generate descriptors from point clouds
            rotated_descriptor = get_global_descriptor(model, rotated_pointcloud)
            np.save(folder_path+model_name+"_descriptors/rotated_"+axis_list[axis]+"_"+str(angle)+".npy", rotated_descriptor)

        for trans in np.arange(-100, 105, 5): # different value for translation
            translated_filename = folder_path+"npy_pointclouds/"+"translated_"+axis_list[axis]+"_"+str(trans/100)+".npy"
            translated_pointcloud = np.load(translated_filename)

            # generate descriptors from point clouds
            translated_descriptor = get_global_descriptor(model, translated_pointcloud)
            np.save(folder_path+model_name+"_descriptors/translated_"+axis_list[axis]+"_"+str(trans/100)+".npy", translated_descriptor)


def plot_feature(model_name):
    import os
    import matplotlib.pyplot as plt

    folder_path = "/home/cel/DockerFolder/code/EPN-NetVLAD/results/invariant_test/"
    if not os.path.exists(folder_path+model_name+"_descriptors_png/"):
        os.mkdir(folder_path+model_name+"_descriptors_png/")

    original_descriptor = np.load(folder_path+model_name+"_descriptors/original_pointcloud_seq5_3.npy")

    axis_list = ['x', 'y', 'z']
    for axis in range(3): # rotating in different axis (x, y, z)
        rotated_similarity_list = []
        for angle in range(0, 365, 5): # different angle in step of 5 degrees
            rotated_descriptor = np.load(folder_path+model_name+"_descriptors/rotated_"+axis_list[axis]+"_"+str(angle)+".npy")
            similarity_rotated = np.absolute(np.dot(original_descriptor, rotated_descriptor)/(np.linalg.norm(original_descriptor)*np.linalg.norm(rotated_descriptor)))
            rotated_similarity_list.append(similarity_rotated)
            rotated_png_savename = folder_path+model_name+"_descriptors_png/"+"rotated_"+axis_list[axis]+"_"+str(angle)+".png"

            x_index = np.arange(rotated_descriptor.size)
            plt.figure()
            plt.plot(x_index, original_descriptor, label='original', color='#929591')
            plt.plot(x_index, rotated_descriptor, label='rotated around %s-axis for %3d degrees, similarity=%.2f' % (axis_list[axis], angle, similarity_rotated))
            plt.xlim(-5, 260)
            plt.ylim(-1, 4)
            plt.title('E$^2$PN-GeM Global Descriptor')
            plt.xlabel('descriptor index')
            plt.ylabel('descriptor value')
            plt.legend(loc='lower left')
            plt.savefig(rotated_png_savename)
            plt.close()

        plt.figure()
        plt.plot(np.arange(0, 365, 5), rotated_similarity_list)
        plt.xlim(-5, 365)
        plt.ylim(0, 1.1)
        plt.title('E$^2$PN-GeM Descriptor Similarity Under Rotation Around %s-axis' % (axis_list[axis]))
        plt.xlabel('rotatation angle around %s-axis' % (str(axis_list[axis])))
        plt.ylabel('descriptor similarity')
        plt.savefig(folder_path+"similarity/"+model_name+"_rotate_"+axis_list[axis]+".png")
        plt.close()

        np.save(folder_path+"similarity/"+model_name+"_rotate_"+axis_list[axis]+".npy", np.array(rotated_similarity_list))

        with open(folder_path+"similarity/"+model_name+"_rotate_"+axis_list[axis]+".txt", "w") as output:
            output.write(str(rotated_similarity_list))

        translated_similarity_list = []
        for trans in np.arange(-100, 105, 5): # different value for translation
            translated_descriptor = np.load(folder_path+model_name+"_descriptors/translated_"+axis_list[axis]+"_"+str(trans/100)+".npy")
            similarity_translated = np.absolute(np.dot(original_descriptor, translated_descriptor)/(np.linalg.norm(original_descriptor)*np.linalg.norm(translated_descriptor)))
            translated_similarity_list.append(similarity_translated)
            translated_png_savename = folder_path+model_name+"_descriptors_png/"+"translated_"+axis_list[axis]+"_"+str(trans/100)+".png"

            x_index = np.arange(translated_descriptor.size)
            plt.figure()
            plt.plot(x_index, original_descriptor, label='original', color='#929591')
            plt.plot(x_index, translated_descriptor, label='translated in %s-axis for %.2f, similarity=%.2f' % (axis_list[axis], trans/100, similarity_translated))
            plt.xlim(-5, 260)
            plt.ylim(-1, 4)
            plt.title('E$^2$PN-GeM Global Descriptor')
            plt.xlabel('descriptor index')
            plt.ylabel('descriptor value')
            plt.legend(loc='lower left')
            plt.savefig(translated_png_savename)
            plt.close()
        
        plt.figure()
        plt.plot(np.arange(-100, 105, 5)/100, translated_similarity_list)
        plt.xlim(-1.1, 1.1)
        plt.ylim(0, 1.1)
        plt.title('E$^2$PN-GeM Descriptor Similarity Under Translation In %s-axis' % (axis_list[axis]))
        plt.xlabel('translation in %s-axis' % (str(axis_list[axis])))
        plt.ylabel('descriptor similarity')
        plt.savefig(folder_path+"similarity/"+model_name+"_translate_"+axis_list[axis]+".png")
        plt.close()

        np.save(folder_path+"similarity/"+model_name+"_translate_"+axis_list[axis]+".npy", np.array(translated_similarity_list))

        with open(folder_path+"similarity/"+model_name+"_translate_"+axis_list[axis]+".txt", "w") as output:
            output.write(str(translated_similarity_list))



def generate_video():
    import imageio.v2 as imageio
    folder_path = "/home/cel/DockerFolder/code/EPN-NetVLAD/results/invariant_test/"
    axis_list = ['x', 'y', 'z']

    # with imageio.get_writer(folder_path+"pointcloud_transformation_withref.gif", mode='I', duration=0.05) as writer:  
    #     for axis in range(3): # rotating in different axis (x, y, z)
    #         for angle in range(0, 365, 5): # different angle in step of 5 degrees
    #             rotated_png_filename = folder_path+"png_pointclouds_withref/"+"rotated_"+axis_list[axis]+"_"+str(angle)+".png"
    #             image = imageio.imread(rotated_png_filename)
    #             writer.append_data(image)

    #     for axis in range(3):
    #         for trans in np.arange(-100, 105, 5): # different value for translation
    #             translated_png_savename = folder_path+"png_pointclouds_withref/"+"translated_"+axis_list[axis]+"_"+str(trans/100)+".png"
    #             image = imageio.imread(translated_png_savename)
    #             writer.append_data(image)

    # with imageio.get_writer(folder_path+"atten_epn_netvlad_descriptors.gif", mode='I', duration=0.05) as writer:  
    #     for axis in range(3): # rotating in different axis (x, y, z)
    #         for angle in range(0, 365, 5): # different angle in step of 5 degrees
    #             rotated_png_filename = folder_path+"atten_epn_netvlad_descriptors_png/"+"rotated_"+axis_list[axis]+"_"+str(angle)+".png"
    #             image = imageio.imread(rotated_png_filename)
    #             writer.append_data(image)

    #     for axis in range(3):
    #         for trans in np.arange(-100, 105, 5): # different value for translation
    #             translated_png_savename = folder_path+"atten_epn_netvlad_descriptors_png/"+"translated_"+axis_list[axis]+"_"+str(trans/100)+".png"
    #             image = imageio.imread(translated_png_savename)
    #             writer.append_data(image)

    # with imageio.get_writer(folder_path+"e2pn_netvlad_descriptors.gif", mode='I', duration=0.05) as writer:  
    #     for axis in range(3): # rotating in different axis (x, y, z)
    #         for angle in range(0, 365, 5): # different angle in step of 5 degrees
    #             rotated_png_filename = folder_path+"e2pn_netvlad_descriptors_png/"+"rotated_"+axis_list[axis]+"_"+str(angle)+".png"
    #             image = imageio.imread(rotated_png_filename)
    #             writer.append_data(image)

    #     for axis in range(3):
    #         for trans in np.arange(-100, 105, 5): # different value for translation
    #             translated_png_savename = folder_path+"e2pn_netvlad_descriptors_png/"+"translated_"+axis_list[axis]+"_"+str(trans/100)+".png"
    #             image = imageio.imread(translated_png_savename)
    #             writer.append_data(image)
   
    # with imageio.get_writer(folder_path+"epn_netvlad_descriptors.gif", mode='I', duration=0.05) as writer:  
    #     for axis in range(3): # rotating in different axis (x, y, z)
    #         for angle in range(0, 365, 5): # different angle in step of 5 degrees
    #             rotated_png_filename = folder_path+"epn_netvlad_descriptors_png/"+"rotated_"+axis_list[axis]+"_"+str(angle)+".png"
    #             image = imageio.imread(rotated_png_filename)
    #             writer.append_data(image)

    #     for axis in range(3):
    #         for trans in np.arange(-100, 105, 5): # different value for translation
    #             translated_png_savename = folder_path+"epn_netvlad_descriptors_png/"+"translated_"+axis_list[axis]+"_"+str(trans/100)+".png"
    #             image = imageio.imread(translated_png_savename)
    #             writer.append_data(image)

    with imageio.get_writer(folder_path+"epn_gem_descriptors.gif", mode='I', duration=0.05) as writer:  
        for axis in range(3): # rotating in different axis (x, y, z)
            for angle in range(0, 365, 5): # different angle in step of 5 degrees
                rotated_png_filename = folder_path+"epn_gem_descriptors_png/"+"rotated_"+axis_list[axis]+"_"+str(angle)+".png"
                image = imageio.imread(rotated_png_filename)
                writer.append_data(image)

        for axis in range(3):
            for trans in np.arange(-100, 105, 5): # different value for translation
                translated_png_savename = folder_path+"epn_gem_descriptors_png/"+"translated_"+axis_list[axis]+"_"+str(trans/100)+".png"
                image = imageio.imread(translated_png_savename)
                writer.append_data(image)

    with imageio.get_writer(folder_path+"e2pn_gem_descriptors.gif", mode='I', duration=0.05) as writer:  
        for axis in range(3): # rotating in different axis (x, y, z)
            for angle in range(0, 365, 5): # different angle in step of 5 degrees
                rotated_png_filename = folder_path+"e2pn_gem_descriptors_png/"+"rotated_"+axis_list[axis]+"_"+str(angle)+".png"
                image = imageio.imread(rotated_png_filename)
                writer.append_data(image)

        for axis in range(3):
            for trans in np.arange(-100, 105, 5): # different value for translation
                translated_png_savename = folder_path+"e2pn_gem_descriptors_png/"+"translated_"+axis_list[axis]+"_"+str(trans/100)+".png"
                image = imageio.imread(translated_png_savename)
                writer.append_data(image)

def plot_similarity(model_name):
    import matplotlib.pyplot as plt

    folder_path = "/home/cel/DockerFolder/code/EPN-NetVLAD/results/invariant_test/"
    axis_list = ['x', 'y', 'z']

    # visualize descriptor similarity under transformation
    fig = plt.figure()
    
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_xlim(-5, 365)
    ax1.set_ylim(0, 1.1)
    ax1.set_xlabel('rotatation angle')
    ax1.set_ylabel('descriptor similarity')
    ax1.set_title('Rotation')

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(0, 1.1)
    ax2.set_xlabel('translation')
    ax2.set_ylabel('descriptor similarity')
    ax2.set_title('Translation')

    for axis in range(3): # rotating in different axis (x, y, z)
        rotated_similarity = np.load(folder_path+"similarity/"+model_name+"_rotate_"+axis_list[axis]+".npy")
        ax1.plot(np.arange(0, 365, 5), rotated_similarity, label='rotation around %s-axis' % (axis_list[axis]))
        

        translated_similarity = np.load(folder_path+"similarity/"+model_name+"_translate_"+axis_list[axis]+".npy")
        ax2.plot(np.arange(-100, 105, 5)/100, translated_similarity, label='translation in %s-axis' % (axis_list[axis]))

    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')

    fig.suptitle('E$^2$PN-NetVLAD Descriptor Similarity Under Transformation')
    plt.subplots_adjust(hspace=0.6)
    plt.savefig(folder_path+"similarity/"+model_name+"_comparison.png")

if __name__ == "__main__":
    # save_transformed_pointcloud()
    # visualization_matplotlib()
    # generate_descriptors('epn_gem')
    # plot_feature('epn_gem')
    # plot_similarity('epn_gem')
    plot_similarity('e2pn_netvlad')
    # plot_feature('e2pn_gem')
    # plot_similarity('e2pn_gem')
    # generate_video()