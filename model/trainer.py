from collections import OrderedDict
import sys
from matplotlib import pyplot as plt
import numpy as np
sys.path.insert(0, '../')

import time
import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate
import pandas as pd

from .criterion import Criterion
from .model import VKFPosOdom, VKFPosBoth
from .optimizer import Optimizer

from tqdm import tqdm
import os.path as osp
from utils.filter import ErrorStateKalmanFilterTensor
from utils.transformation import angular_error, compute_vo


def safe_collate(batch):
    """
    Collate function for DataLoader that filters out None's
    :param batch: minibatch
    :return: minibatch filtered for None's
    """
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)

class Trainer(object):
    def __init__(self, train_dataset:data.Dataset, 
                 val_dataset:data.Dataset, 
                 config):
        self.config = config
        torch.manual_seed(config.random_seed)
        
        if config.model == 'VKFPosOdom':
            self.model = VKFPosOdom()
        elif config.model == 'VKFPosBoth':
            self.model = VKFPosBoth(share_levels_n=3, dropout=config.dropout)
            
        if config.is_cuda:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                torch.cuda.manual_seed(config.random_seed)
            else:
                assert("Detect no gpu or cuda, check setup")
                
        if config.data_set == 'starirs':
            self.seq_len = 499
        elif config.data_set == 'heads':
            self.seq_len = 999
        else :
            self.seq_len = val_dataset.__len__()
            
        self.train_dataset = train_dataset
        self.epochs = config.epochs
        self.lr = self.config.learning_rate
        self.t_error_best, self.q_error_best = [np.inf, np.inf]
        
        self.scale = torch.tensor([config.t_scale, config.t_scale, config.t_scale,
                                   config.r_scale, config.r_scale, config.r_scale]).to(self.device)
        # dataloader
        self.train_loader = data.DataLoader(train_dataset, batch_size=self.config.batch_size,
                                            shuffle=True,num_workers=config.num_workers,
                                            collate_fn=safe_collate)
        if self.config.valid :
            self.val_loader = data.DataLoader(val_dataset, batch_size=1,
                                            shuffle=False,num_workers=4,
                                            collate_fn=safe_collate)
        
        # set criterion
        self.criterion = Criterion(self.config.train_task, batchsize=self.config.batch_size, requires_grad=False)
        self.t_criterion = lambda t_pred, t_gt: torch.median(torch.norm(t_pred - t_gt, dim=-1))
        self.q_criterion = angular_error
        
        # set optimizer
        if config.train_task == 'odom':
            for param in list(self.model.parameters()):
                param.requires_grad = False
            param_list = [param for param_group in self.model.odom_param_list for param in param_group]
            for param in param_list:
                param.requires_grad = True
            self.optimizer = Optimizer(params=param_list, config=self.config)
            
        elif config.train_task == 'apr':
            for param in list(self.model.parameters()):
                param.requires_grad = False
            param_list = [param for param_group in self.model.global_param_list for param in param_group]
            for param in param_list:
                param.requires_grad = True
            self.optimizer = Optimizer(params=param_list, config=self.config)
        
        elif config.train_task == 'both':
            if self.config.sep_training:
                odom_param_list = [param for param_group in self.model.odom_param_list for param in param_group]
                global_param_list = [param for param_group in self.model.global_param_list for param in param_group]
                odom_param_list.extend(self.criterion.odom_param_list)
                global_param_list.extend(self.criterion.global_param_list)
                self.odom_optimizer = Optimizer(params=odom_param_list, config=self.config)
                self.global_optimizer = Optimizer(params=global_param_list, config=self.config)
            else:
                # for apr and odom
                param_list = list(self.model.parameters())
                self.optimizer = Optimizer(params=param_list, config=self.config)
        
        elif config.train_task == 'EKF':
            # only uncertainty
            param_list = list(self.model.parameters())
            
            param_list = [param for param_group in self.model.global_param_list for param in param_group]
            self.optimizer = Optimizer(params=param_list, config=self.config)
        
        # load previous weight if resuming
        if config.pretrain_file is not None:
            self.load_pretrain()
            
        elif config.resume_training:
        #    load previous weight 
            self.load_state_dict()
        else:
            self.start_epoch = 0

        
        # initial log file    
        self.log_initial()
            
        # to cuda
        if config.is_cuda:
            self.model.to(self.device)
            self.criterion.to(self.device)
        

    def safe_collate(self, batch):
        """
        Collate function for DataLoader that filters out None's
        :param batch: minibatch
        :return: minibatch filtered for None's
        """
        # batch := [(img, pose), (img, pose), ...]
        # to filter out whether img is None
        real_batch = torch.cat([item for item in batch if item[0] is not None])
        return real_batch  
    
    def load_pretrain(self):
        print(f"Loading pretrain from {self.config.pretrain_file}...")
        loc_func = lambda storage, loc: storage.cuda(0) if self.config.is_cuda else torch.device('cpu')
        checkpoint = torch.load(self.config.pretrain_file, map_location=loc_func)

        model_names = [n for n, _ in self.model.named_parameters()]
        state_names = [n for n in checkpoint['model_state_dict'].keys()]
        
        if model_names[0].find(state_names[0]) >= 0:
            model_prefix = model_names[0].replace(state_names[0], '')
            state_prefix = None
        elif state_names[0].find(model_names[0]) >= 0:
            state_prefix = state_names[0].replace(model_names[0], '')
            model_prefix = None
        else:
            print('Could not find the correct prefixes between {:s} and {:s}'.format(model_names[0], state_names[0]))
            raise KeyError
        
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            if state_prefix is None:
                k = model_prefix + k
            else:
                k = k.replace(state_prefix, '')
            new_state_dict[k] = v
        self.start_epoch = 0 
        self.model.load_state_dict(new_state_dict)
        
        
    def load_state_dict(self):
        loc_func = lambda storage, loc: storage.cuda(0) if self.config.is_cuda else torch.device('cpu')
        checkpoint = torch.load(self.config.checkpoint_file, map_location=loc_func)
        
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
            print('Could not find the correct prefixes between {:s} and {:s}'.format(model_names[0], state_names[0]))
            raise KeyError
        
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            if state_prefix is None:
                k = model_prefix + k
            else:
                k = k.replace(state_prefix, '')
            new_state_dict[k] = v
        self.start_epoch = checkpoint['epoch'] + 1 
        self.t_error_best =  checkpoint['t_error_best']
        self.q_error_best =  checkpoint['q_error_best']
        self.model.load_state_dict(new_state_dict)
        
        if not self.config.sep_training:
            # Move optimizer state dictionary to the correct device before loading
            for param in self.optimizer.learner.param_groups[0]['params']:
                param.data = param.data.to(self.device)
            
            # Load optimizer checkpoint
            self.optimizer.learner.load_state_dict(checkpoint['optim_state_dict'])
        else:
            for param in self.odom_optimizer.learner.param_groups[0]['params']:
                param.data = param.data.to(self.device)

            for param in self.global_optimizer.learner.param_groups[0]['params']:
                param.data = param.data.to(self.device)
                
            # Load optimizer checkpoint
            self.odom_optimizer.learner.load_state_dict(checkpoint['odom_optim_state_dict'])
            self.global_optimizer.learner.load_state_dict(checkpoint['global_optim_state_dict'])
            
        # Load criterion checkpoint
        self.criterion.load_state_dict(checkpoint['criterion_state_dict'])
        
        print('Loaded checkpoint {:s} epoch {:d}'.format(self.config.checkpoint_file, checkpoint['epoch']))

    
    def save_checkpoint(self, epoch):
        
        if not osp.exists(osp.join(self.config.logdir, self.config.scene)):
            osp.mkdir(osp.join(self.config.logdir, self.config.scene))
            
        if self.config.sep_training:
            odom_optim_state_dict = self.odom_optimizer.learner.state_dict()
            global_optim_state_dict = self.global_optimizer.learner.state_dict()
            filename = osp.join(self.config.logdir, 
                                self.config.scene, self.config.train_task +'_sep_epoch_{:03d}.pth.tar'.format(epoch))
            checkpoint_dict = {'epoch': epoch,
                                't_error_best': self.t_error_best,
                                'q_error_best': self.q_error_best,
                                'model_state_dict': self.model.state_dict(),
                                'odom_optim_state_dict': odom_optim_state_dict,
                                'global_optim_state_dict': global_optim_state_dict,
                                'criterion_state_dict': self.criterion.state_dict()}
        else:
            optim_state_dict = self.optimizer.learner.state_dict()
            filename = osp.join(self.config.logdir, 
                                self.config.scene, self.config.train_task +'_epoch_{:03d}.pth.tar'.format(epoch))
            checkpoint_dict = {'epoch': epoch, 
                               't_error_best': self.t_error_best,
                                'q_error_best': self.q_error_best,
                                'model_state_dict': self.model.state_dict(),
                                'optim_state_dict': optim_state_dict,
                                'criterion_state_dict': self.criterion.state_dict()}
            
        torch.save(checkpoint_dict, filename)
        print('Epoch {:d} checkpoint saved to {:s}'.format(epoch, filename))
    
    def log_initial(self):
        filename = osp.join(self.config.hisdir, self.config.hispath)
        if self.config.resume_training and osp.exists(filename):
            print(f"Load log file from {filename} ...")
            df = pd.read_csv(filename)
            self.train_log = df.to_dict(orient="list")
                
        else:
            self.train_log = {'Epoch': [None],
                          'Loss': [None],
                          't_error': [None],
                          'q_error': [None],
                          'val_t_error': [None],
                          'val_q_error': [None],
                          'learning_rate': [self.lr],
                          'sx_abs': [self.criterion.sx_abs.cpu().item()],
                          'sq_abs': [self.criterion.sq_abs.cpu().item()],
                          'sx_rel': [self.criterion.sx_rel.cpu().item()],
                          'sq_rel': [self.criterion.sq_rel.cpu().item()],
                          'sx_vo': [self.criterion.sx_vo.cpu().item()],
                          'sq_vo': [self.criterion.sq_vo.cpu().item()]
                          } 
        
    def log_update(self, epoch, loss, t_error, q_error, val_t_error, val_q_error):
        self.train_log['Epoch'].append(epoch)
        self.train_log['Loss'].append(loss)
        self.train_log['t_error'].append(t_error)
        self.train_log['q_error'].append(q_error)
        self.train_log['val_t_error'].append(val_t_error)
        self.train_log['val_q_error'].append(val_q_error)
        self.train_log['learning_rate'].append(self.lr)
        
        self.train_log['sx_abs'].append(self.criterion.sx_abs.cpu().item())
        self.train_log['sq_abs'].append(self.criterion.sq_abs.cpu().item())
        self.train_log['sx_rel'].append(self.criterion.sx_rel.cpu().item())
        self.train_log['sq_rel'].append(self.criterion.sq_rel.cpu().item())
        self.train_log['sx_vo'].append(self.criterion.sx_vo.cpu().item())
        self.train_log['sq_vo'].append(self.criterion.sq_vo.cpu().item())
        
    def log_writer(self):
        df = pd.DataFrame(self.train_log)
        filename = osp.join(self.config.hisdir, self.config.hispath)    
        df.to_csv(filename, index=False)

    def train_epoch(self, epoch):
        # the routine of train step
        # batch := (B x clips x 3 x H x W, B x clips x 6)
        self.model.train()
        epoch_loss = 0
        t_error = 0
        q_error = 0
        train_loss = 0
        iter = (epoch - 1) * len(self.train_loader) + 1
        with tqdm(self.train_loader, unit="batch") as tepoch:                
            for batch_counter, (images, gt) in enumerate(tepoch):
                if images.shape[0] == 1:
                    break
                if torch.isnan(images).any():
                    print("The images get nan")
                tepoch.set_description(f"Epoch {epoch}")
                if self.config.is_cuda and self.config.train_task != 'EKF':
                    images, gt = images.to(self.device), gt.to(self.device)  

                if self.config.train_task == 'both':                    
                    estimated_relative_pose, estimated_absolute_pose = self.model(images.float())
                    pose_loss, odom_loss = self.criterion(xq_odom=estimated_relative_pose,
                                                xq_global=estimated_absolute_pose, 
                                                xq_gt=gt)
                    train_loss = pose_loss + odom_loss
                    
                    t_error += self.t_criterion(estimated_absolute_pose[:, -1, -1, :3], gt[:, -1, :3])
                    q_error += torch.mean(self.q_criterion(estimated_absolute_pose[:, -1, -1, 3:6], gt[:, -1, 3:]))
                
                elif self.config.train_task == 'EKF': 
                    # initial EKF
                    tensor_filter = ErrorStateKalmanFilterTensor(batch_size=self.config.batch_size)
                    images = images.permute(1, 0, 2, 3, 4) # to clipsxBx3xHxW
                    clips = images.shape[0]-1
                    gt = gt.permute(1, 0, 2) # to clipsxbx6
                    if self.config.is_cuda:
                        gt = gt.to(self.device) 
                    batch_loss = 0
                    pose_vec = []
                    gt_vec = []
                    for idx in range(clips):
                        clip_images = images[idx:idx+2]
                        clip_gt = gt[idx: idx+2]
                        if self.config.is_cuda:
                            clip_images = clip_images.to(self.device)
                        clip_images = clip_images.permute(1, 0, 2, 3, 4) # to Bxclipsx3xHxW
                        clip_gt = clip_gt.permute(1, 0, 2) # to Bxclipsx6
                        estimated_relative_pose, estimated_absolute_pose = self.model(clip_images.float())

                        if not tensor_filter.x_init:                            
                            tensor_filter.x = estimated_absolute_pose[:, -1, -2, :6]
                            tensor_filter.x_init = True
                            # define uncertainty matrix
                            tensor_filter.P = torch.diag_embed(torch.exp(estimated_absolute_pose[:, -1, -2, 6:12]))
                            covariance_list = torch.stack([torch.diag(p) for p in tensor_filter.P])
                            
                            gt_vec.append(clip_gt[:, -2].unsqueeze(0))
                            pose_vec.append(torch.cat([tensor_filter.x, covariance_list], dim=-1).unsqueeze(0))

                            
                            
                        # define uncertainty matrix
                        if torch.isnan(estimated_relative_pose).any() or torch.isnan(estimated_absolute_pose[:, -1, -1, 6:12]).any():
                            import os
                            print("Nan")
                            os._exit(0)
                        tensor_filter.W = torch.diag_embed(torch.exp(estimated_relative_pose[:, 6:12]) )
                        tensor_filter.R = torch.diag_embed(torch.exp(estimated_absolute_pose[:, -1, -1, 6:12]))



                        tensor_filter.predict(estimated_relative_pose[:, :6])
                        tensor_filter.update(estimated_absolute_pose[:, -1, -1, :6])
                        covariance_list = torch.stack([torch.diag(p) for p in tensor_filter.P])
                        gt_vec.append(clip_gt[:, -1].unsqueeze(0))
                        pose_vec.append(torch.cat([tensor_filter.x, covariance_list], dim=-1).unsqueeze(0))
                        

                        t_error += self.t_criterion(tensor_filter.x[:, :3].detach(), clip_gt[:, -1, :3]) / clips
                        q_error += torch.median(self.q_criterion(tensor_filter.x[:, 3:6].detach(), clip_gt[:, -1, 3:])) /clips

                    gt_vec = torch.cat(gt_vec, dim=0)
                    pose_vec = torch.cat(pose_vec, dim=0)

                    pose_loss, odom_loss = self.criterion(xq_odom=None,
                                                            xq_global=pose_vec, 
                                                            xq_gt=gt_vec, train_uncer=True)
                    batch_loss = pose_loss 
                    batch_loss.backward()
                    self.optimizer.learner.step()
                    self.optimizer.learner.zero_grad()

                    train_loss = batch_loss/clips
                       
                             
                elif self.config.train_task == 'odom':
                    estimated_relative_pose, estimated_absolute_pose = self.model(images.float())
                    pose_loss, odom_loss = self.criterion(xq_odom=estimated_relative_pose,
                                                xq_global=estimated_absolute_pose, 
                                                xq_gt=gt, transform_scale=self.scale)
                    train_loss = odom_loss
                    # recover sclae
                    vo_gt = compute_vo(gt[:, -2], gt[:, -1])
                    estimated_relative_pose[:, :6] = estimated_relative_pose[:, :6]/self.scale
                    
                    t_error += self.t_criterion(estimated_relative_pose[:, :3].detach(), vo_gt[:,  :3])
                    q_error += torch.mean(self.q_criterion(estimated_relative_pose[:, 3:6].detach(), vo_gt[:, 3:]))
                elif self.config.train_task == 'apr':
                    estimated_relative_pose, estimated_absolute_pose = self.model(images.float())
                    pose_loss, odom_loss = self.criterion(xq_odom=estimated_relative_pose,
                                                xq_global=estimated_absolute_pose, 
                                                xq_gt=gt)
                    train_loss = pose_loss
                    
                    t_error += self.t_criterion(estimated_absolute_pose[:, -1, -1, :3], gt[:, -1, :3])
                    q_error += torch.mean(self.q_criterion(estimated_absolute_pose[:, -1, -1, 3:6], gt[:, -1, 3:]))

                # compute gradient and do optimizer step
                if self.config.train_task != 'EKF':
                    if self.config.sep_training:
                        odom_loss.backward()
                        pose_loss.backward()
                        if (batch_counter-1) % 4 == 0:
                            torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), max_norm=1)
                            self.odom_optimizer.learner.step()
                            self.global_optimizer.learner.step()
                            self.odom_optimizer.learner.zero_grad()
                            self.global_optimizer.learner.zero_grad()
                    else:
                        # with torch.autograd.detect_anomaly():
                        train_loss.backward()
                        self.optimizer.learner.step()
                        self.optimizer.learner.zero_grad()


                epoch_loss += train_loss.item()
                tepoch.set_postfix(loss=train_loss.item())
                iter += 1
        # ADJUST LR
        print(f"learning rate: {self.lr}")        
        print(f"Train iteration: {iter}")
        if not self.config.sep_training:
            self.lr = self.optimizer.adjust_lr(epoch)
            
        else:
            self.lr = self.odom_optimizer.adjust_lr(epoch)
            self.lr = self.global_optimizer.adjust_lr(epoch)
        
        
        torch.cuda.empty_cache()
        return epoch_loss / len(self.train_loader), \
            [t_error.item()/len(self.train_loader), q_error.item()/len(self.train_loader)]
    
        
    @torch.no_grad()
    def valid(self, epoch):
        
        self.model.eval()
        # self.model.dropout.train()
        pred_poses = []  # element: Nx6
        targ_poses = []  # element: Nx6
        total_inference_time = 0
        tensor_filter = ErrorStateKalmanFilterTensor(batch_size=1)
        with tqdm(self.val_loader, unit="frame") as tepoch:
            tepoch.set_description(f"Test")
            for images, gt in tepoch:
                if len(targ_poses) % self.seq_len == 0: # need to change to 499 if dataset is stair
                    tensor_filter = ErrorStateKalmanFilterTensor(batch_size=1)
                # gt_np = torch.clone(gt)
                if self.config.val_task == 'odom':
                    targ_poses.append(compute_vo(gt[:, -2], gt[:, -1])) 
                else:
                    targ_poses.append(gt[:, -1]) 
                
                # with torch.cuda.amp.autocast():  
                if self.config.is_cuda:
                    images, gt = images.to(self.device), gt.to(self.device)
                
                if self.config.val_task == 'EKF' :
                    start_time = time.time()                         
                    estimated_relative_pose, estimated_absolute_pose = self.model(images.float())
                    Inference_Time = (time.time()-start_time) *1000
                    if not tensor_filter.x_init:
                        estimated_absolute_pose = torch.mean(estimated_absolute_pose, dim=1)
                        tensor_filter.x = estimated_absolute_pose[:, :6]
                        tensor_filter.x_init = True
                        # define uncertainty matrix
                        tensor_filter.P = torch.diag_embed(torch.exp(estimated_absolute_pose[:, 6:12]))
                        tensor_filter.x_post = tensor_filter.x.clone()
                    else:
                        # define uncertainty matrix
                        tensor_filter.W = torch.diag_embed(torch.exp(estimated_relative_pose[:, 6:12])/self.scale**2)
                        # predict motion from vo
                        tensor_filter.predict(estimated_relative_pose[:, :6]/self.scale)
                        predictor_count = estimated_absolute_pose.shape[1] # emsemble times
                        for c in range(predictor_count):
                            tensor_filter.R = torch.diag_embed(torch.exp(estimated_absolute_pose[:, c, 6:12]))                        
                            tensor_filter.update(estimated_absolute_pose[:, c, :6])

                    tepoch.set_postfix(Inference_Time=f"{Inference_Time:.1f}")
                    total_inference_time += Inference_Time
                    pred_poses.append(tensor_filter.x_post)
                    
                elif self.config.val_task == 'odom':
                    start_time = time.time()  
                    estimated_relative_pose, estimated_absolute_pose = self.model(images.float())
                    Inference_Time = (time.time()-start_time) *1000
                    tepoch.set_postfix(Inference_Time=f"{Inference_Time:.1f}")
                    total_inference_time += Inference_Time
                    pred_poses.append(estimated_relative_pose[:, :6]/self.scale)
                    
                elif self.config.val_task == 'apr':
                    start_time = time.time()  
                    estimated_relative_pose, estimated_absolute_pose = self.model(images.float())
                    Inference_Time = (time.time()-start_time) *1000
                    tepoch.set_postfix(Inference_Time=f"{Inference_Time:.1f}")
                    total_inference_time += Inference_Time
                    pred_poses.append(estimated_absolute_pose[:, -1, :6])
                    
        targ_poses = torch.cat(targ_poses)
        pred_poses = torch.cat(pred_poses).cpu().data
        
        t_loss = self.t_criterion(pred_poses[:, :3], targ_poses[:, :3]).item()
        q_loss = torch.median(self.q_criterion(pred_poses[:, 3:], targ_poses[:, 3:])).item()
        

        return t_loss, q_loss
    
    def fit(self) -> None:
        print('start training ...')
        
        for epoch in range(self.start_epoch, self.config.epochs):
            
            # Train
            train_loss, [t_error, q_error] = self.train_epoch(epoch)
            if self.config.valid :
                if epoch % self.config.valid_feq == 0:
                    val_t_error, val_q_error = self.valid(epoch)
                else:
                    val_t_error, val_q_error = self.t_error_best, self.q_error_best

                if (val_t_error + val_q_error) < (self.t_error_best + self.q_error_best):
                    self.t_error_best = val_t_error
                    self.q_error_best = val_q_error

                    filename = osp.join(
                        self.config.logdir, self.config.scene, 'best_weight.pth.tar')
                
                    checkpoint_dict = {'epoch': epoch, 'model_state_dict': self.model.state_dict(), 
                                    't_error_best': self.t_error_best, 'q_error_best': self.q_error_best}
                    torch.save(checkpoint_dict, filename)
                    print(f'Save best checkpoint to {filename}')
            else:
                val_t_error, val_q_error = 0.0, 0.0

            train_info = f"Train {self.config.train_task},  Epoch: {epoch}, " + \
                f"Train_Loss: {train_loss}\n train_t_error_midan: {t_error}, train_q_error: {q_error}, " + \
                    f"val_t_error_midan: {val_t_error}, val_q_error_midan: {val_q_error}"
            
            
            self.log_update(epoch=epoch, loss=train_loss, t_error=t_error, q_error=q_error, \
                val_t_error = val_t_error, val_q_error = val_q_error)
            self.log_writer()
            
            # save checkpoint
            if epoch % self.config.valid_feq == 0 :
                self.save_checkpoint(epoch)
        
            print(train_info)
    
    