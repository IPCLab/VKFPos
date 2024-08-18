import argparse
import torch
from model.trainer import Trainer
from dataloader.seven_scenes import SevenScenes
from dataloader.robotcar import RobotCar
from torchvision import transforms
import torch.multiprocessing
import numpy as np

def safe_collate(batch):
        """
        Collate function for DataLoader that filters out None's
        :param batch: minibatch
        :return: minibatch filtered for None's
        """
        real_batch = [item for item in batch if item[0] is not None]
        return real_batch

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=7)
    parser.add_argument('--model', type=str,
                        choices=['VKFPosOdom', 'VKFPosBoth'], 
                        default='VKFPosOdom')
    parser.add_argument('--sep_training', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.5)
    
    # for Training
    parser.add_argument('--is_cuda', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--checkpoint_file', type=str, default=None)
    parser.add_argument('--pretrain_file', type=str, default=None)
    parser.add_argument('--train_task', type=str,
                        choices=['both', 'odom', 'apr'], default='both')
    parser.add_argument('--val_task', type=str,
                        choices=['odom', 'apr', 'EKF'], default='both')
    parser.add_argument('--valid_feq', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--hisdir', type=str, default='his')
    parser.add_argument('--hispath', type=str, default='trial_0.csv')
    parser.add_argument('--valid', type=int, default=1)
    # for dataloader
    parser.add_argument('--data_set', type=str, default="7scenes",
                    help='Choose the data set in [7scenes, RobotCar]')
    parser.add_argument('--t_scale', type=int, default=1, help='for rescale data in training stage')
    parser.add_argument('--r_scale', type=int, default=1, help='for rescale data in training stage')
    parser.add_argument('--data_dir', type=str, required=True,
                    help='The root dir of dataset')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument("--scene", type=str, help="Only for the 7Scenes dataset")
    parser.add_argument("--seq_assign", type=list, help="Only for the 7Scenes dataset")
    # for optimizer
    parser.add_argument('--optimizer', type=str, 
                        choices=['sgd', 'adam', 'rmsprop'], default='adam')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--eps', type=float, default=1e-10)
    parser.add_argument("--lr_decay", type=float, default=1,
                    help="The decaying rate of learning rate")
    parser.add_argument('--lr_stepvalues', type=str, default='[]',
                    help='The decaying epoch of learning rate in the form of a list')

    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.benchmark=True
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    
    print(f"Seperate training set as {args.sep_training}")
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
        train_dataset = SevenScenes(args.scene, args.data_dir, clip=2, 
                            transform=image_transform, target_transform=target_transform)
        if args.valid:
            val_dataset = SevenScenes(args.scene, args.data_dir, clip=2,
                            transform=image_transform, target_transform=target_transform, train=False)
            print('Do validation')
        else:
            val_dataset = 'No data'
            print('No validation')

    elif args.data_set == 'robotcar':
        stats_file = os.path.join(args.data_dir, args.scene, 'stats.txt')
        stats = np.loadtxt(stats_file)
        image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=stats[0],
                std=np.sqrt(stats[1]))
        ])
        target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())
        
        train_dataset = RobotCar(args.scene, args.data_dir, clip=2, train=True,
                            transform=image_transform, target_transform=target_transform)
        if args.valid:
            image_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=stats[0],
                    std=np.sqrt(stats[1]))
            ])
            val_dataset = RobotCar(args.scene, args.data_dir, clip=2,
                            transform=image_transform, target_transform=target_transform, train=False)
            print('Do validation')
        else:
            val_dataset = 'No data'
            print('No validation')
        
    trainer = Trainer(train_dataset, val_dataset, config=args)
    trainer.fit()