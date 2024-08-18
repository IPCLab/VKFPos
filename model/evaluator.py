import sys
from tqdm import tqdm
sys.path.insert(0, '../')

import torch
import torch.utils.data as data
import numpy as np
import time, csv
import matplotlib.pyplot as plt

from .model import VKFPosOdom, VKFPosBoth
from .criterion import Criterion
from utils.filter import ErrorStateKalmanFilterTensor
from utils.transformation import apply_vo, angular_error_np



class Evaluator(object):
    def __init__(self, test_dataset:data.Dataset, config):
        self.config = config
        torch.manual_seed(config.random_seed)
        
        if config.model == 'VKFPosOdom':
            self.model = VKFPosOdom()
        elif config.model == 'VKFPosBoth':
            self.model = VKFPosBoth(training=False, share_levels_n=3)
            
                
        if config.is_cuda:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                torch.cuda.manual_seed(config.random_seed)
            else:
                assert("Detect no gpu or cuda, check setup")
                
        self.test_dataset = test_dataset
        
        if test_dataset is not None:
            self.dataloader = data.DataLoader(test_dataset, batch_size=config.batch_size,
                                                shuffle=False, num_workers=config.num_workers)
            self.data_iter = iter(self.dataloader)
        # set criterion
        self.criterion = Criterion(self.config.test_task, batchsize=config.batch_size, requires_grad=False)
        self.t_criterion = lambda t_pred, t_gt: torch.norm(t_pred - t_gt, dim=-1)
        self.q_criterion = angular_error_np
        
        self.vo_pose = None
        # to cuda
        if config.is_cuda:
            self.model.to(self.device)
            self.criterion.to(self.device)
        
        # initial and give label
        with open('export/rpr_sigma.csv', 'w', newline='') as rpr_sigma_file:
            writer = csv.writer(rpr_sigma_file)
            writer.writerow(['tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
        with open('export/apr_sigma.csv', 'w', newline='') as apr_sigma_file:
            writer = csv.writer(apr_sigma_file)
            writer.writerow(['tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
        
        
        self.load_state_dict()
        
        
        # initial filter if needed
        if self.config.mode == "EKF":
            self.filter = ErrorStateKalmanFilterTensor(device='cpu')
            self.u = torch.zeros((1, 6), device=torch.device('cpu'))
            self.x_prior = None
            self.P_hist = []
        
    def load_state_dict(self):
        loc_func = lambda storage, loc: storage.cuda(0) if self.config.is_cuda\
            else torch.device('cpu')
        checkpoint = torch.load(
            self.config.checkpoint_file, map_location=loc_func)
        
        model_names = [n for n, _ in self.model.named_parameters()]
        state_names = [n for n in checkpoint['model_state_dict'].keys()]
        
        # find prefix for the model and state dicts from the first param name
        if model_names[0].find(state_names[0]) >= 0:
            model_prefix = model_names[0].replace(state_names[0], '')
            state_prefix = None
        elif state_names[0].find(model_names[0]) >= 0:
            state_prefix = state_names[0].replace(model_names[0], '')
            model_prefix = None
        else:
            print('Could not find the correct prefixes between {:s} and {:s}'.
                format(model_names[0], state_names[0]))
            raise KeyError
        
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            if state_prefix is None:
                k = model_prefix + k
            else:
                k = k.replace(state_prefix, '')
            new_state_dict[k] = v
            
        self.model.load_state_dict(new_state_dict)
        print('Loaded checkpoint {:s} epoch {:d}'.format(self.config.checkpoint_file,
                                                         checkpoint['epoch']))
       
    @torch.no_grad()
    def predict_next(self, reset=False):
        self.model.eval()
        if reset:
            self.filter = ErrorStateKalmanFilterTensor(device='cpu')
        images, gt = next(self.data_iter)
        if self.config.is_cuda:
            images_cuda = images.to(self.device)
        start_t = time.time()
        estimated_relative_pose, estimated_absolute_pose = self.model(images_cuda.float())
        end_t = time.time()
        inference_time = end_t - start_t
        
        images = images[:, -1].squeeze().numpy()
        gt_np = gt[:, -1].squeeze().numpy() # 6-dim
        
        # using VO
        if self.config.mode == 'vo':
            if self.vo_pose is None:
                self.vo_pose = gt[:, -1].clone()
            else:
                self.vo_pose = apply_vo(self.vo_pose, estimated_relative_pose[:, :6].cpu())
            return_pose = self.vo_pose.squeeze().numpy().copy() # 6-dim
            return images, return_pose, gt_np, inference_time
        elif self.config.mode == 'global': 
            estimated_absolute_pose = estimated_absolute_pose[:, -1, :6].cpu().squeeze().numpy() # 6-dim 
            return images, estimated_absolute_pose, gt_np, inference_time  
        
        elif self.config.mode == "EKF":
            estimated_absolute_pose = estimated_absolute_pose.cpu()
            estimated_relative_pose = estimated_relative_pose.cpu()
            self.x_prior = self.filter.x
            if not self.filter.x_init :
                estimated_absolute_pose = torch.mean(estimated_absolute_pose, dim=1)
                self.filter.x = estimated_absolute_pose[:, :6]
                self.filter.x_init = True
                x_abosulote_uncer = torch.exp(estimated_absolute_pose[:, 6:9])
                q_abosulote_uncer = torch.exp(estimated_absolute_pose[:, 9:12])
                self.filter.P = torch.block_diag(torch.eye(3)*x_abosulote_uncer,
                                                 torch.eye(3)*q_abosulote_uncer).unsqueeze(0)
                self.filter.x_post = self.filter.x.clone()
                self.filter.P_post = self.filter.P.clone()
            
            else:
                x_relative_uncer = torch.exp(estimated_relative_pose[:, 6:9])
                q_relative_uncer = torch.exp(estimated_relative_pose[:, 9:12])
                self.filter.W = torch.block_diag(torch.eye(3)*x_relative_uncer,
                                                 torch.eye(3)*q_relative_uncer).unsqueeze(0)
    
                self.filter.predict(estimated_relative_pose[:, :6])
                with open('export/rpr_sigma.csv', 'a', newline='') as rpr_sigma_file:
                    writer = csv.writer(rpr_sigma_file)
                    writer.writerow(np.concatenate([x_relative_uncer.squeeze().numpy(),
                                                    q_relative_uncer.squeeze().numpy()]))
    
                
                predictor_count = estimated_absolute_pose.shape[1]
                for c in range(predictor_count):
                    x_abosulote_uncer = torch.exp(estimated_absolute_pose[:, c, 6:9])
                    q_abosulote_uncer = torch.exp(estimated_absolute_pose[:, c, 9:12])
                    self.filter.R = torch.block_diag(torch.eye(3)*x_abosulote_uncer,
                                                    torch.eye(3)*q_abosulote_uncer).unsqueeze(0)

                    self.filter.update(estimated_absolute_pose[:, c, :6]) 

                    with open('export/apr_sigma.csv', 'a', newline='') as apr_sigma_file:
                        writer = csv.writer(apr_sigma_file)
                        writer.writerow(np.concatenate([x_abosulote_uncer.squeeze().numpy(),
                                                        q_abosulote_uncer.squeeze().numpy()]))

            x_post_np = self.filter.x.squeeze().numpy()
            P_post_np = self.filter.P.squeeze().numpy()
            return images, x_post_np, np.diag(P_post_np), gt_np, inference_time
        
    
        
    @torch.no_grad()
    def test(self):
        self.model.eval()
        
        pred_poses = []  
        targ_poses = []  
        total_inference_time = 0
        with tqdm(self.dataloader, unit="frame") as tepoch:
            tepoch.set_description(f"Test")
            for images, gt in tepoch:
                # initial state
                targ_poses.append(gt[:, -1])      
                
                if self.config.is_cuda:
                    images, gt = images.to(self.device), gt.to(self.device)
                
                if self.config.test_task == 'both':
                    start_time = time.time()  
                    estimated_relative_pose, estimated_absolute_pose = self.model(images.float())
                    
                    Inference_Time = (time.time()-start_time) *1000
                    tepoch.set_postfix(Inference_Time=f"{Inference_Time:.1f}")
                    total_inference_time += Inference_Time
                    
                    if self.config.mode == 'vo':
                        # evaluate vo pose
                        # using VO
                        if self.vo_pose is None:
                            self.vo_pose = gt[:, -1]
                        else:
                            self.vo_pose = apply_vo(self.vo_pose, estimated_relative_pose[:, :6].cpu()).squeeze().numpy().copy() # 6-dim
                        pred_poses.append(self.vo_pose)
                    elif self.config.mode == 'global':
                        # evaluate global pose
                        pred_poses.append(estimated_absolute_pose)
                    
                    
                elif self.config.test_task == 'odom':
                    start_time = time.time()  
                    estimated_relative_pose = self.model(images.float())
                    Inference_Time = (time.time()-start_time) *1000
                    tepoch.set_postfix(Inference_Time=f"{Inference_Time:.1f}")
                    total_inference_time += Inference_Time
                    
                    estimated_relative_pose = estimated_relative_pose.cpu().data.numpy().squeeze()
                    abs_pose = self.transformation_mult(pred_poses[-1],
                                                        estimated_relative_pose)
                    pred_poses.append(abs_pose)
                    
                elif self.config.test_task == 'apr':
                    start_time = time.time()  
                    estimated_absolute_pose = self.model(images.float())
                    Inference_Time = (time.time()-start_time) *1000
                    tepoch.set_postfix(Inference_Time=f"{Inference_Time:.1f}")
                    total_inference_time += Inference_Time
                    pred_poses.append(estimated_absolute_pose.cpu().data.numpy().squeeze())
        
        targ_poses = torch.cat(targ_poses)
        pred_poses = torch.cat(pred_poses).cpu().data
        
        # compute error    
        t_error_list = self.t_criterion(pred_poses[:, :3], targ_poses[:, :3])
        q_error_list = self.q_criterion(pred_poses[:, 3:], targ_poses[:, 3:])
        
        avg_inference_time = total_inference_time / self.dataloader.__len__()
        print(f"average Inference Time: {avg_inference_time:.3f} ms")
        errs = {
            "Error in translation(median)": "{:5.3f}".format(torch.median(t_error_list).item()),
            "Error in translation(mean)": "{:5.3f}".format(torch.mean(t_error_list).item()),
            "Error in rotation(median)": "{:5.3f}".format(torch.median(q_error_list).item()),
            "Error in rotation(mean)": "{:5.3f}".format(torch.mean(q_error_list).item()),
        }
        import pandas as pd
        df = pd.DataFrame(errs, index=[0])
        print(df)
     
        self.visualize(pred_poses_log = pred_poses[:999].numpy(), targ_poses_log = targ_poses[:999].numpy())
        
                
                
                
    def visualize(self, pred_poses_log, targ_poses_log):
         
        pred_poses = np.concatenate([pred_poses_log[:, :3], pred_poses_log[:, 3:]], axis=1)
        targ_poses = np.concatenate([targ_poses_log[:, :3], targ_poses_log[:, 3:]], axis=1)
        
        # create figure object          
        fig = plt.figure()
        if self.config.dataset != '7Scenes':
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
        
        ss = 1
        # scatter the points and draw connecting line
        x = np.vstack((pred_poses[::ss, 0].T, targ_poses[::ss, 0].T))
        y = np.vstack((pred_poses[::ss, 1].T, targ_poses[::ss, 1].T))
        
        if self.config.dataset != '7Scenes':  # 2D drawing
            ax.plot(x, y, c='b')
            ax.scatter(x[0, :], y[0, :], c='r')
            ax.scatter(x[1, :], y[1, :], c='g')
        else:
            z = np.vstack((pred_poses[::ss, 2].T, targ_poses[::ss, 2].T))
            for xx, yy, zz in zip(x.T, y.T, z.T):
                ax.plot(xx, yy, zs=zz, c='b', linewidth=0.5)
                
            # for predicted pose(red)
            ax.scatter(x[0, :], y[0, :], zs=z[0, :],
                       c='r', depthshade=0, s=0.8)
            # for groundtruth pose(green)
            ax.scatter(x[1, :], y[1, :], zs=z[1, :],
                       c='g', depthshade=0, s=0.8)
            ax.view_init(azim=119, elev=13)
            
        # fig.savefig
        plt.show()
        return fig
        