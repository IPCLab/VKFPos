import sys
sys.path.insert(0, '../')
import torch
import torch.nn as nn
from utils.transformation import compute_vo


def l2_loss(output, target):
    loss = (output - target) ** 2
    return loss


class Criterion(nn.Module):
    def __init__(self, task:str, batchsize:int, requires_grad:bool=False):
        super(Criterion, self).__init__()
        
        x_init = 0
        q_init = -3
        self.task = task
        self.sx_abs = nn.Parameter(torch.Tensor(
            [x_init]), requires_grad=requires_grad)
        self.sq_abs = nn.Parameter(torch.Tensor(
            [q_init]), requires_grad=requires_grad)

        self.sx_rel = nn.Parameter(torch.Tensor(
            [x_init]), requires_grad=requires_grad)
        self.sq_rel = nn.Parameter(torch.Tensor(
            [q_init]), requires_grad=requires_grad)
        
        self.sx_vo = nn.Parameter(torch.Tensor(
            [x_init]), requires_grad=requires_grad)
        self.sq_vo = nn.Parameter(torch.Tensor(
            [q_init]), requires_grad=requires_grad)
        
        self.all_param_list = [self.sx_vo, self.sq_vo,
                               self.sx_abs, self.sq_abs,
                               self.sx_rel, self.sq_rel]
        self.odom_param_list = [self.sx_vo, self.sq_vo]
        self.global_param_list = [self.sx_abs, self.sq_abs,
                                  self.sx_rel, self.sq_rel]
        
        self.abs_weight = 1
        self.rel_weight = 0.5
        self.odom_weight = 1

        self.zeros3 = torch.zeros(3).to(torch.device("cuda:0"))
        self.loss_func = l2_loss

    
    def get_saveable_params(self):
        return {
            'sx_abs': self.sx_abs,
            'sq_abs': self.sq_abs,
            'sx_rel': self.sx_rel,
            'sq_rel': self.sq_rel,
            'sx_vo': self.sx_vo,
            'sq_vo': self.sq_vo,
        }    
            
    def forward(self, xq_odom=None, xq_global=None, xq_gt=None, train_uncer=False, transform_scale=None):
        pose_loss, odom_loss = None, None
            

        if self.task == 'both' or self.task == 'odom':
            odom_gt = compute_vo(xq_gt[:, -2], xq_gt[:, -1])
            if transform_scale is not None:
                odom_gt = odom_gt * transform_scale
            
            # compute odom loss
            odom_x_loss = self.loss_func(odom_gt[:, :3], xq_odom[:, :3])
            odom_q_loss = self.loss_func(odom_gt[:, 3:], xq_odom[:, 3:6])
            odom_loss = 0.5 * torch.exp(-xq_odom[:, 6:9]) *(odom_x_loss) + 0.5 * xq_odom[:, 6:9] \
                + (0.5 * torch.exp(-xq_odom[:, 9:12])*odom_q_loss + 0.5 * xq_odom[:, 9:12])
            odom_loss = odom_loss.mean()
            odom_loss = self.odom_weight * odom_loss
            
        if self.task == 'both' or self.task == 'apr': 
            # for xq_global shape BxTx2x12
            predictor_count = xq_global.shape[1]
            pose_loss = []
            for c in range(predictor_count):
                
                # global
                x_global = xq_global[:, c, -1, :3]
                q_global = xq_global[:, c, -1, 3:6]
                x_global_uncer = xq_global[:, c, -1, 6:9]
                q_global_uncer = xq_global[:, c, -1, 9:12]

                # compute global loss
                abs_x_loss = self.loss_func(x_global, xq_gt[:, -1, :3])
                abs_q_loss = self.loss_func(q_global, xq_gt[:, -1, 3:])

                abs_global_loss = 0.5* torch.exp(-x_global_uncer)*(abs_x_loss) + 0.5* x_global_uncer \
                    + (0.5 * torch.exp(-q_global_uncer)*(abs_q_loss) + 0.5* q_global_uncer)

                abs_global_loss = abs_global_loss.mean()
                
                # global
                x_global = xq_global[:, c, -2, :3]
                q_global = xq_global[:, c, -2, 3:6]
                x_global_uncer = xq_global[:, c, -2, 6:9]
                q_global_uncer = xq_global[:, c, -2, 9:12] 
            
                # compute global loss
                abs_x_loss = self.loss_func(x_global, xq_gt[:, -2, :3])
                abs_q_loss = self.loss_func(q_global, xq_gt[:, -2, 3:])

                abs_global_loss0 = 0.5 * torch.exp(-x_global_uncer)*(abs_x_loss) + 0.5* x_global_uncer \
                    + (0.5 * torch.exp(-q_global_uncer)*(abs_q_loss) + 0.5* q_global_uncer)
                abs_global_loss0 = abs_global_loss0.mean() # + uncer_loss
                
                pose_loss.append(abs_global_loss)
            pose_loss = torch.stack(pose_loss).mean()

            
            
        if self.task == 'EKF':
            abs_global_loss = 0
            
            clip_len = xq_global.shape[0]
            for idx in range(clip_len):
                predict = xq_global[idx]
                gt = xq_gt[idx]
                if train_uncer:
                    global_x_loss = self.loss_func(predict[:, :3], gt[:, :3])
                    global_q_loss = self.loss_func(predict[:, 3:6], gt[:, 3:6])
                    abs_global_loss += 0.5 *global_x_loss/predict[:, 6:9] + 0.5 * torch.log(predict[:, 6:9]) \
                                + 100*(0.5 * global_q_loss/predict[:, 9:12] + 0.5 * torch.log(predict[:, 9:12]))

                else:
                    abs_x_loss = self.loss_func(predict[:, :3], gt[:, :3])
                    abs_q_loss = self.loss_func(predict[:, 3:6], gt[:, 3:6])
                    abs_global_loss += torch.exp(-self.sx_abs)*(abs_x_loss) + self.sx_abs \
                                    + torch.exp(-self.sq_abs)*(abs_q_loss) + self.sq_abs
            abs_global_loss = abs_global_loss.mean()     
            pose_loss = self.abs_weight * abs_global_loss 
            
        return pose_loss, odom_loss
    
  