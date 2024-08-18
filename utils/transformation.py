import torch
import pytorch3d.transforms as p3t


def apply_vo(tq1, vo):
    # tq1 @ vo
    # tq1 @ vo = tq2
    q1_exp = p3t.so3_exp_map(tq1[:, 3:])
    qvo_exp = p3t.so3_exp_map(vo[:, 3:])
    
    q2_exp = torch.bmm(q1_exp, qvo_exp)
    t2 = torch.bmm(q1_exp, vo[:, :3].unsqueeze(-1)).squeeze(-1)+tq1[:, :3]
    return torch.cat([t2, p3t.so3_log_map(q2_exp)], dim=1)
    

def compute_vo(tq1, tq2):
    # tq1^{-1} @ tq2 = vo, tq2 is new, tq1 is old
    # tq1 @ vo = tq2
    q1_exp = p3t.so3_exp_map(tq1[:, 3:])
    q2_exp = p3t.so3_exp_map(tq2[:, 3:])
    try:
        q1_exp_inv = q1_exp.inverse()
    except:
        print(q1_exp)
        
    
    qvo_exp = torch.bmm(q1_exp_inv, q2_exp)
    tvo = torch.bmm(q1_exp_inv, (tq2[:, :3] - tq1[:, :3]).unsqueeze(-1)).squeeze(-1)
    return torch.cat([tvo, p3t.so3_log_map(qvo_exp)], dim=1)

def angular_error(q1, q2):
    return p3t.so3_relative_angle(p3t.so3_exp_map(q1), 
                                    p3t.so3_exp_map(q2)) / torch.pi * 180
    
def angular_error_np(q1, q2):
    q1 = torch.from_numpy(q1).view((-1, 3))
    q2 = torch.from_numpy(q2).view((-1, 3))
    error = p3t.so3_relative_angle(p3t.so3_exp_map(q1), 
                                    p3t.so3_exp_map(q2)) / torch.pi * 180
    return error.squeeze().numpy()


import numpy as np

def qlog( q):
    if all(q[1:] == 0):
        q = np.zeros(3)
    else:
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
    return q





if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    
    tq1 = torch.tensor([[0, 0, 0, 0.0, 0.0, 0.0]])
    tq2 = torch.tensor([[0, 0, 0, 0.0, 0.0, 0.0]])
    print(compute_vo(tq1, tq2))
    print(p3t.so3_exp_map(tq1[:, 3:]) - p3t.so3_exp_map(tq2[:, 3:]))
    print(angular_error(tq1[:, 3:], tq2[:, 3:]))
    x_idx = []
    ang_error = []
    ang_error2 = []
    ang_error3 = []
    for i in range(600):
        x_idx.append(0.01*i*180/np.pi)
        
        tq2 = torch.tensor([[0, 0, 0, 0.0, 0.0, 0.01*i]])
        ang_error.append(angular_error(tq1[:, 3:], tq2[:, 3:]))
    
    plt.plot(x_idx, ang_error, label='so(3)')
      
    q1 = np.array([0.0, 0.0, 0.0])
    x_idx = []
    ang_error3 = []
    ang_error4 = []
    for i in range(600):
        x_idx.append(0.01*i)
        q3 = np.array([0.0, 0.0, 0.01*i])
        
    plt.plot(x_idx, ang_error3, label='log(q)')
    plt.xlabel('The difference of one variable')
    plt.ylabel('Angle Erorr (deg)')
    plt.axis("equal")
    plt.legend()
    plt.show()

    