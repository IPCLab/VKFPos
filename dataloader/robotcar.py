import dataloader.robotcar_sdk.image as image
import dataloader.robotcar_sdk.camera_model as camera_model
from dataloader.robotcar_sdk.interpolate_poses import interpolate_ins_poses
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
import pytorch3d.transforms as p3t
from .load_image import load_image


class RobotCar(data.Dataset):
    def __init__(self, scene:str, data_path:str, seq_assign:int=None, train:bool=False,
                 transform:transforms.Compose=None,
                 target_transform=None, clip:int=2, seed:int=7) -> None:
        super().__init__()
        
        self.transform = transform
        self.target_transform = target_transform
        self.clip = clip
        self.train = train
        np.random.seed(seed)
        
        base_dir = osp.expanduser(osp.join(data_path, scene))
        data_dir = osp.join('datasets', 'robotcar', scene)
        if train:
            split_filename = osp.join(base_dir, 'TrainSplit.txt')
        else:
            split_filename = osp.join(base_dir, 'TestSplit.txt')
            
        with open(split_filename, 'r') as f:
            seqs = [l.rstrip() for l in f if not l.startswith('#')]
        if seq_assign is not None:
            seqs = [seq_assign]
        self.seqs_len = len(seqs) 
        
        # read pose normalization term
        pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')
        mean_t, std_t = np.loadtxt(pose_stats_filename)
        self.mean_t = mean_t
        self.std_t = std_t
            
        self.poses = []
        ts = {}
        self.imgs = []
        self.first_index_of_seqs = []
        for seq in seqs:
            
            self.first_index_of_seqs.append(len(self.imgs))
            seq_dir = osp.join(base_dir, seq)
            print(f"loading {seq_dir} data ...")
            
            # read the image timestamps
            ts_filename = osp.join(seq_dir, 'stereo.timestamps')
            with open(ts_filename, 'r') as f:
                ts[seq] = [int(l.rstrip().split(' ')[0]) for l in f]
            
            # read groundtruth
            pose_filename = osp.join(seq_dir, 'gps', 'ins.csv')
            pss = np.asarray(interpolate_ins_poses(pose_filename, ts[seq], ts[seq][0]))

            abosulte_pose = [self.matrix_to_tq(p) for p in pss]
            for i in range(len(abosulte_pose)):
                abosulte_pose[i][:3] = (abosulte_pose[i][:3] - mean_t) / std_t
            self.poses.extend(abosulte_pose)
            # read image
            c_imgs = [osp.join(seq_dir, 'stereo', 'centre_processed', '{:d}.png'.
                                format(t)) for t in ts[seq]]
            self.imgs.extend(c_imgs)
            
        if self.clip > 2:
            tmp_list = self.first_index_of_seqs.copy()
            for c in range(self.clip-2):
                self.first_index_of_seqs.extend([index+c+1 for index in tmp_list])
        
        self.create_window()   
        self.camera_model = camera_model.CameraModel(osp.join('dataloader', 'robotcar_camera_models'),
                               osp.join('stereo', 'centre'))
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
                pose = np.float32(self.poses[filtered_index-t*random_skip])
                img = load_image(self.imgs[filtered_index-t*random_skip])
            else:
                pose = np.float32(self.poses[filtered_index-t])
                img = load_image(self.imgs[filtered_index-t])
                
            # image transform
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
        # transpose to 3 x clips
        
        poses = np.concatenate(poses, axis=0)
        poses = np.asarray(poses)
        return imgs, poses   
    
    
    def __len__(self) -> int:
        return len(self.poses) - (self.clip-1)*self.seqs_len
    
    
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
    
    def create_window(self):
        self.window = [i for i in range(len(self.poses)) 
                       if i not in self.first_index_of_seqs]
        
    def pose_denormalize(self, pose):
        pose[:, :3] = pose[:, :3] * self.std_t 
        return pose
    
if __name__ == '__main__':
    robotcar = RobotCar('loop', data_path= 'datasets/robotcar', seq_assign='2014-06-23-15-41-25',
                        train=False)
    for i in range(3355):
        item = robotcar.__getitem__(i)
    print(item)
    print(robotcar.__len__())