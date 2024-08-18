import sys
sys.path.insert(0, '../')

import os
import os.path as osp
import numpy as np
import torch
import torch.utils.data as data
from .load_image import load_image
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from torchvision import transforms
import pytorch3d.transforms as p3t

class SevenScenes(data.Dataset):
    def __init__(self, scene:str, data_path:str, seq_assign:int=None, train:bool=True,
                 transform:transforms.Compose=None,
                 target_transform=None, clip:int=2, seed:int=7):
        """_summary_

        Args:
            scene (str): _description_
            data_path (str): _description_
            train (bool, optional): _description_. Defaults to True.
            transform (transforms.Compose, optional): _description_. Defaults to None.
            target_transform (_type_, optional): _description_. Defaults to None.
            clip (int, optional): _description_. Defaults to 2.
            seed (int, optional): _description_. Defaults to 22.
        """
        self.transform = transform
        self.target_transform = target_transform
        self.clip = clip
        self.train = train
        np.random.seed(seed)

        if scene == 'all':
            # directories
            scene_list = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']
            seqs_dir = []
            for scene in scene_list:
                base_dir = osp.join(osp.expanduser(data_path), scene)
                if train:
                    file_folder_list = osp.join(base_dir, 'TrainSplit.txt')
                else:
                    file_folder_list = osp.join(base_dir, 'TestSplit.txt')
                
                with open(file_folder_list, 'r') as f:
                    seq_dir = [osp.join(base_dir, 'seq-{:02d}'.format(int(l.split('sequence')[-1])))
                            for l in f if not l.startswith('#')]
                seqs_dir.extend(seq_dir)
        else:
            # directories
            base_dir = osp.join(osp.expanduser(data_path), scene)
            
            # decide which sequence
            if train:
                file_folder_list = osp.join(base_dir, 'TrainSplit.txt')
            else:
                file_folder_list = osp.join(base_dir, 'TestSplit.txt')
            
            with open(file_folder_list, 'r') as f:
                
                seqs_dir = [osp.join(base_dir, 'seq-{:02d}'.format(int(l.split('sequence')[-1])))
                        for l in f if not l.startswith('#')]
                
            if seq_assign is not None:
                seqs_dir = [osp.join(base_dir, 'seq-{:02d}'.format(int(i))) 
                            for i in seq_assign]
        
        # read pose normalization term
        pose_stats_filename = osp.join(base_dir, 'pose_stats.txt')
        mean_t, std_t = np.loadtxt(pose_stats_filename)
        self.mean_t = mean_t
        self.std_t = std_t
        
        self.first_index_of_seqs = []
        self.seqs_len = len(seqs_dir)    
        self.imgs = []
        self.poses = []

        for seq_dir in seqs_dir:
            self.first_index_of_seqs.append(len(self.imgs))
            print(f"loading {seq_dir} data ...")
            
            # pose file is writen as T matrix in 4x4, camera-to-world
            pose_filenames = [n for n in os.listdir(osp.join(seq_dir, '.')) if
                           n.find('pose') >= 0]
            frame_idx = np.array(range(len(pose_filenames)), dtype=int)
            pss = [np.loadtxt(osp.join(seq_dir, 'frame-{:06d}.pose.txt'.
                                       format(i))) for i in frame_idx]
            abosulte_pose = [self.matrix_to_tq(p) for p in pss]
            # normalize pose
            for i in range(len(abosulte_pose)):
                abosulte_pose[i][:3] = (abosulte_pose[i][:3] - mean_t) / std_t
            self.poses.extend(abosulte_pose)
            
            c_imgs = [osp.join(seq_dir, 'frame-{:06d}.color.png'.format(i))
                      for i in frame_idx]
            self.imgs.extend(c_imgs)
            
            # visualization
            if 0:
                all_x = [p[4] for p in abosulte_pose]
                all_y = [p[5] for p in abosulte_pose]
                axis_range = [min(all_x)-0.1, max(all_x)+0.1, 
                            min(all_y)-0.1, max(all_y)+0.1]
                self._visualize_initial(axis_range)
                for i, image_path in enumerate(c_imgs):
                    self._visualize_update(image_path, abosulte_pose[i], axis_range)

        # consider first several clips
        if self.clip > 2:
            tmp_list = self.first_index_of_seqs.copy()
            for c in range(self.clip-2):
                self.first_index_of_seqs.extend([index+c+1 for index in tmp_list])
        
        self.create_window()        
        
    def __getitem__(self, index:int) -> list[tuple]:
        """_summary_

        Args:
            index (int): _description_

        Returns:
            dict: 
        """
        # to ensure all images is in a same sequence
        imgs = []
        poses = []
        # get filtered index
        filtered_index = self.window[index]
        if self.train :
            random_skip = np.random.randint(0, 6) # generate a random skip path from [0, 6)
        else:
            random_skip = 1

        for t in range(self.clip):
            # Ensure that the image exists
            if filtered_index-t*random_skip in self.window:
                pose = self.poses[filtered_index-t*random_skip]
                img = load_image(self.imgs[filtered_index-t*random_skip])
            else:
                pose = self.poses[filtered_index-t]
                img = load_image(self.imgs[filtered_index-t])

            if self.target_transform is not None:
                pose = self.target_transform(pose)
                pose = pose.unsqueeze(0)
            if self.transform is not None:
                img = self.transform(img)
                img = img.unsqueeze(0)
                
            imgs.append(img)
            poses.append(pose)
        
        imgs.reverse()
        poses.reverse()
        imgs = np.concatenate(imgs, axis=0)
        imgs = np.asarray(imgs)
        # # transpose to 3 x clips
       
        poses = np.concatenate(poses, axis=0)
        poses = np.asarray(poses)
        return imgs, poses        
        
    def __len__(self) -> int:
        return len(self.poses) - (self.clip-1)*self.seqs_len
    
    def qlog(self, q):
        if all(q[1:] == 0):
            q = np.zeros(3)
        else:
            q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
        return q
    
    # to create a selected index set
    def create_window(self):
        self.window = [i for i in range(len(self.poses)) 
                       if i not in self.first_index_of_seqs]
    
    def compute_relative_pose(self, T_0:np.ndarray[float], T_1:np.ndarray[float]) -> np.ndarray[float]:
        """_summary_

        Args:
            T_0 (np.ndarray[float]): The rigid transformation matrix of T_0 
            T_1 (np.ndarray[float]): The rigid transformation matrix of T_1 

        Returns:
            np.ndarray[float]: The relative pose transformation matrix T_{0,1}
        """
        return np.linalg.inv(T_0)*T_1
         
    def matrix_to_tq(self, T:np.ndarray[float]) -> np.ndarray:
        """_summary_

        Args:
            T (list[list]): pose file is writen as T matrix in 4x4, camera-to-world

        Returns:
            np.ndarray[float]: 7-dof with [t, q]
        """
        so3_r = p3t.so3_log_map(torch.from_numpy(np.expand_dims(T[:3, :3], 0))).squeeze().numpy()
        t = T[:3, 3]
        return np.concatenate([t, so3_r])
        
    def pose_denormalize(self, pose):
        pose[:, :3] = pose[:, :3] * self.std_t +self.mean_t
        return pose
    
        

if __name__ =='__main__':
    
    import torch
    from torch.utils.data import DataLoader
    from time import time
    from tqdm import tqdm
    image_transform = transforms.Compose([
        transforms.Resize((640, 480)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.34721234, 0.36705238, 0.36066107],
            std=[0.30737526, 0.31515116, 0.32020183]),
    ])
    target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())
    dataset = SevenScenes('fire', '/mnt/f/code/VKFPos/datasets/7Scenes', clip=3, 
                          transform=image_transform, target_transform=target_transform)