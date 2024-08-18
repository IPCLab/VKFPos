import argparse
import torch
import sys
from PyQt5.QtWidgets import QApplication
from utils.visualizer import Visualizer
from dataloader.seven_scenes import SevenScenes
from dataloader.robotcar import RobotCar
from torchvision import transforms
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=7)
    parser.add_argument('--model', type=str,
                        choices=['VKFPosOdom', 'VKFPosBoth'], 
                        default='VKFPosOdom')
    # for test
    parser.add_argument('--is_cuda', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_task', type=str,
                        choices=['both', 'odom', 'apr'], default='both')
    parser.add_argument('--checkpoint_file', type=str, default=None)
    
    # for dataloader
    parser.add_argument('--data_set', type=str, default="7scenes",
                    help='Choose the data set in [7scenes, robotcar]')
    parser.add_argument('--data_dir', type=str, required=True,
                    help='The root dir of dataset')
    parser.add_argument("--dataset", type=str, default='7Scenes')
    parser.add_argument("--scene", type=str, help="Only for the 7Scenes dataset")
    parser.add_argument("--seq_assign", type=str, default=None,
                        help="Only for the 7Scenes dataset")
    parser.add_argument("--mode", type=str, default='global',
                        choices=['global', 'vo', 'EKF'])
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()
    
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.deterministic=True
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())
    if args.data_set == '7scenes':
        
        stats_file = os.path.join(args.data_dir, args.scene, 'stats.txt')
        stats = np.loadtxt(stats_file)
        image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=stats[0],
                std=np.sqrt(stats[1]))
        ])
        test_dataset = SevenScenes(args.scene, args.data_dir, clip=2, 
                                seq_assign=args.seq_assign,
                          transform=image_transform, 
                          target_transform=target_transform,
                          train=False)

    elif args.data_set == 'robotcar':
        stats_file = os.path.join(args.data_dir, args.scene, 'stats.txt')
        stats = np.loadtxt(stats_file)
        image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=stats[0],
                std=np.sqrt(stats[1]))
        ])
        target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

        test_dataset = RobotCar(args.scene, args.data_dir, clip=2, seq_assign=args.seq_assign,
                            transform=image_transform, target_transform=target_transform, train=False)

    app = QApplication(sys.argv)
    viewer = Visualizer(test_dataset, args)
    viewer.show()
    torch.backends.cudnn.deterministic=False
    sys.exit(app.exec_())
    
    