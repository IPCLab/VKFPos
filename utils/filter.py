import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from utils.transformation import compute_vo, apply_vo
import pytorch3d.transforms as p3t

class ErrorStateKalmanFilterTensor:
    """
        Error State Kalman Filter Implementation in Pytorch Tensor
        that enables batch optimization
    """
    def __init__(self, device='cuda:0', batch_size=1):
        self.dim_x = 6
        self.dim_z = 6
        self.dim_u = 6
        
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        self.x = torch.zeros((batch_size, self.dim_x), dtype=torch.float32).to(self.device)  # state
        self.x_init = False
        self.P = torch.cat([torch.eye(self.dim_x, dtype=torch.float32).to(self.device).unsqueeze(0) 
                            for i in range(batch_size)])  # batch uncertainty covariance (b, 6, 6)
        self.F = torch.cat([torch.eye(self.dim_x, dtype=torch.float32).to(self.device).unsqueeze(0) 
                            for i in range(batch_size)])  # state transition matrix
        self.R = torch.cat([torch.eye(self.dim_z, dtype=torch.float32).to(self.device).unsqueeze(0) 
                            for i in range(batch_size)])  # motion uncertainty
        self.W = torch.cat([torch.eye(self.dim_x, dtype=torch.float32).to(self.device).unsqueeze(0) 
                            for i in range(batch_size)])  # measurement uncertainty
        self.z = None  # measurement
        
        self.eye3 = torch.eye(3, dtype=torch.float32).to(self.device)
        self.eye6 = torch.eye(6, dtype=torch.float32).to(self.device)
        self.eye3_batch = torch.cat([torch.eye(3, dtype=torch.float32).to(self.device).unsqueeze(0) 
                            for i in range(batch_size)])
        self.eye6_batch = torch.cat([torch.eye(6, dtype=torch.float32).to(self.device).unsqueeze(0) 
                            for i in range(batch_size)])
        self.zeros3 = torch.zeros((3, 3), dtype=torch.float32).to(self.device)
        self.zeros6 = torch.zeros((6, 6), dtype=torch.float32).to(self.device)
        
        # the following variables are always provided for reading
        self.x_prior = self.x.clone()
        self.P_prior = self.P.clone()
        self.x_post = self.x.clone()
        self.P_post = self.P.clone()
    
    
    def predict(self, u:torch.tensor):
        """
        predict the next state of x, with u in shape (6,) 
        u := (t, log(q)) in camera system
        """
        
        # predict state by motion model
        self.x = apply_vo(self.x, u)
        self.P = self.P + self.W
        self.x_prior = torch.clone(self.x)
        self.P_prior = torch.clone(self.P)
        
        
    def update(self, z:torch.tensor):
        """
        update the prediction through measurement z, with z in shape (6,)
        z := (t, log(q)) in global coordinate
        """

        self.r = compute_vo(self.x, z)
        self.K = torch.bmm(self.P, torch.inverse(self.P+self.R))
        delta_r = torch.bmm(self.K, self.r.unsqueeze(-1)).squeeze(-1)
        self.x = apply_vo(self.x, delta_r)
        
        # update covariance of system
        S = self.P + self.R
        self.P = self.P - torch.bmm(torch.bmm(self.K, S), torch.transpose(self.K, 1, 2))
        self.x_post = self.x.clone()
        self.P_post = self.P.clone()
        
    
    def hat_op(self, x:torch.Tensor) -> torch.Tensor:
        return torch.tensor([[0., -x[2], x[1]],
                     [x[2], 0., -x[0]],
                     [-x[1], x[0], 0.]], dtype=torch.float32).to(self.device)
        
    def Jl(self, theta:torch.Tensor) -> torch.Tensor:
        # from eq 145
        if torch.norm(theta) < 1e-5:
            return self.eye3
        theta_hat = self.hat_op(theta)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        return self.eye3 + ((1-cos_theta) / theta**2) @ theta_hat + \
            ((theta - sin_theta) / theta**3) @ theta_hat @ theta_hat
            
    def Jr(self, theta:torch.Tensor) -> torch.Tensor:
        # from eq 143
        Jr_list = []
        for i in range(self.batch_size):
            if torch.norm(theta[i]) < 1e-5:
                Jr_list.append(self.eye3.unsqueeze(0))
                continue
            theta_hat = self.hat_op(theta[i])
            cos_theta = torch.cos(theta[i])
            sin_theta = torch.sin(theta[i])
            first_term = (1-cos_theta)/(cos_theta**2) @ theta_hat
            second_term = ((theta[i]-sin_theta)/(cos_theta**3)) @ theta_hat**2
            Jr_list.append((self.eye3 - first_term + second_term).unsqueeze(0))
        Jr = torch.cat(Jr_list)
        return Jr
    
    def Q(self, rho:torch.Tensor, theta:torch.Tensor) -> torch.Tensor:
        # from eq. 180
        rho_hat = self.hat_op(rho)
        theta_hat = self.hat_op(theta)
        if torch.norm(theta) < 1e-5:
            return 1/2 * rho_hat
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        Q = 1/2 * rho_hat + \
            ((theta - sin_theta) / theta**3) * (theta_hat@rho_hat+rho_hat@theta_hat+theta_hat@rho_hat@theta_hat) - \
                (1-theta**2/2-cos_theta)/theta**4 * (theta_hat@theta_hat@rho_hat+rho_hat@theta_hat@theta_hat-3*theta_hat@rho_hat@theta_hat) -\
                    1/2 * ((1-theta**2/2-cos_theta)/theta**4 - 3*(theta-sin_theta-theta**3/6)/theta**5) * \
                        (theta_hat@rho_hat@theta_hat@theta_hat + theta_hat@theta_hat@rho_hat@theta_hat)
        return Q


if __name__ == '__main__':
    EKF = ErrorStateKalmanFilterTensor(device='cuda:0', batch_size=5)
    torch.manual_seed(22)

    gt = torch.zeros((5, 6), dtype=torch.float32).to(torch.device('cuda:0'))
    noise_sum = torch.zeros((5, 6), dtype=torch.float32).to(torch.device('cuda:0'))
    gt_list = [[gt[0, 0].cpu().item(), gt[0, 1].cpu().item()]]
    gt_list_with_noise = [[gt[0, 0].cpu().item(), gt[0, 1].cpu().item()]]
    EKF_x_prior = [[EKF.x_prior[0, 0].cpu().item(), EKF.x_prior[0, 1].cpu().item()]]
    EKF_x_post = [[EKF.x_post[0, 0].cpu().item(), EKF.x_post[0, 1].cpu().item()]]
    
    for i in range(10):
        vo = torch.tensor([[1, 0.0, 0.0, 0.0, 0.0, 0.0] for j in range(5)], dtype=torch.float32).to(torch.device('cuda:0'))
        gt += vo
        
        noise = torch.randn((5, 6), dtype=torch.float32).to(torch.device('cuda:0')) * 0.1
        noise_sum += noise
        gt_with_noise = gt+noise *0.1
        gt_list.append([gt[0, 0].cpu().item(), gt[0, 1].cpu().item()])
        gt_list_with_noise.append([gt_with_noise[0, 0].cpu().item(), gt_with_noise[0, 1].cpu().item()])
        EKF.predict(vo+noise)
        EKF.update(z = gt) 
        EKF_x_prior.append([EKF.x_prior[0, 0].cpu().item(), EKF.x_prior[0, 1].cpu().item()])
        EKF_x_post.append([EKF.x_post[0, 0].cpu().item(), EKF.x_post[0, 1].cpu().item()])
        

    import matplotlib.pyplot as plt
    plt.plot(list(zip(*gt_list))[0], list(zip(*gt_list))[1], label = 'gt')
    plt.plot(list(zip(*gt_list_with_noise))[0], list(zip(*gt_list_with_noise))[1], label = 'gt_noise')
    plt.plot(list(zip(*EKF_x_prior))[0], list(zip(*EKF_x_prior))[1], label = 'EKF_prior')
    plt.plot(list(zip(*EKF_x_post))[0], list(zip(*EKF_x_post))[1], label = 'EKF_post')
    plt.legend()
    plt.show()
    