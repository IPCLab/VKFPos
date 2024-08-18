# import io
import numpy as np
# import cv2
import os.path as osp
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Ellipse

import time
import sys, os
import csv
from model.evaluator import Evaluator
from utils.transformation import angular_error_np




class EllipsePlotter:
    def __init__(self, ax, initial_params):
        self.ax = ax
        self.ellipse_params = initial_params
        self.ellipse = Ellipse(*self.ellipse_params)
        self.ax.add_patch(self.ellipse)

    def update_ellipse(self, new_params):
        self.ellipse_params = new_params
        self.ellipse.set_center(new_params[0])
        self.ellipse.width = new_params[1]
        self.ellipse.height = new_params[2]

    
        
class Visualizer(QMainWindow):
    def __init__(self, test_dataset, config) -> None:
        super().__init__()
        self.evaluator = Evaluator(test_dataset, config)
        self.mode = config.mode
        self.data_len = test_dataset.__len__()
        self.base_dir = config.data_dir
        self.export_dir = 'export'  # Directory to save exported data
        os.makedirs(self.export_dir, exist_ok=True)
        
        self.t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
        self.q_criterion = angular_error_np
        self.t_error = []
        self.q_error = []
        self.x_indice = []
        
        self.setWindowTitle("Real-Time Image Viewer")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        if config.data_set == 'stairs':
            self.seq_len = 499
        elif config.data_set == 'heads':
            self.seq_len = 999
        else :
            self.seq_len = test_dataset.__len__()
        
            
        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)
        
        # Set a fixed size for the QLabel to match the original image size
        self.image_label.setFixedSize(640, 480)
        
        self.figure = plt.figure(figsize=(6, 6))
        self.pose_scatter = None
        self.ax = self.figure.add_subplot(3,1,1)
        self.ax2 = self.figure.add_subplot(3,1,2)
        self.ax3 = self.figure.add_subplot(3,1,3)
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.timer.start(0)

        self.imgs = []
        self.poses = np.array([])
        self.current_image_idx = 0  # Index of the current image being displayed
        
        self.start_t = time.time()
        self.model_infer_time = 0
        self.sigma = None
        
        if self.mode == 'EKF':
            self.img, pose, covariance, gt = self.get_image_and_pose(self.current_image_idx)
            self.sigma = np.expand_dims(covariance, axis=0) 
        else: 
            self.img, pose, gt = self.get_image_and_pose(self.current_image_idx)
        self.poses = np.expand_dims(pose, axis=0)
        self.gts = np.expand_dims(gt, axis=0)
        self.paused = False  # Flag to control whether updates are paused
        self.update_image()
        self.show_closed_hint = False
        
        
    def update_image(self):
        if self.paused or self.current_image_idx == self.data_len:
            if self.current_image_idx == self.data_len and not self.show_closed_hint:
                print("Please close the windows ")
                self.show_closed_hint = True
            return  # Don't update if paused
        img = np.transpose(self.img, (1, 2, 0)) # numpy array
        img = self.img_denormal(img)*255
        img = np.clip(img, 0, 255).astype(np.uint8)  # Clamp values to [0, 255]

        # Convert the image to QImage
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_image = QImage(img.data.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Display the QImage in the QLabel
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)
        
        t = 0
        pose_x = self.poses[t:, 0]
        pose_y = self.poses[t:, 1]
        gt_x = self.gts[t:, 0]
        gt_y = self.gts[t:, 1]
        if self.sigma is not None:
            sigma_x = self.sigma[t:, 0]
            sigma_y = self.sigma[t:, 1]
        
        # compute error
        self.x_indice.append(len(self.t_error))
        self.t_error.append(self.t_criterion(self.poses[-1, :3], self.gts[-1, :3]))
        self.q_error.append(self.q_criterion(self.poses[-1, 3:], self.gts[-1, 3:]))
        
        
        if self.pose_scatter is None:
            # trajectory
            
            self.pose_scatter = self.ax.scatter([pose_x], [pose_y], 
                                                c='g', label='Pose', linewidths=0.7, zorder=3)
            self.gt_scatter = self.ax.scatter([gt_x], [gt_y], 
                                              c='r', label='Ground Truth', linewidths=0.7, zorder=2)
            self.plotter, = self.ax.plot([pose_x, gt_x], [pose_y, gt_y], 
                                         c='b', linewidth=0.2, zorder=1)
            if self.sigma is not None:
                self.ellipse_plotter = EllipsePlotter(self.ax, ((pose_x, pose_y), sigma_x, sigma_y))
            self.ax.legend()
            
            self.display_fps_label = QLabel(self.central_widget)
            self.display_fps_label.setStyleSheet("color: red; font-size: 16pt;") # Set font size here
            self.layout.addWidget(self.display_fps_label)
            
            self.model_infer_time_label = QLabel(self.central_widget)
            self.model_infer_time_label.setStyleSheet("color: red; font-size: 16pt;") # Set font size here
            self.layout.addWidget(self.model_infer_time_label)
            
            # error plot
            self.t_plotter, = self.ax2.plot(self.x_indice, self.t_error, 
                                            c='r', label='t_error')
            self.q_plotter, = self.ax3.plot(self.x_indice, self.q_error, 
                                                  c='b', label='R_error')
                
            self.ax2.legend(loc='upper left')
            self.ax3.legend(loc='upper left')
            self.ax2.set_ylabel('meter')
            self.ax3.set_ylabel('degree')
        else:
            if self.gts.shape[0] % 1 == 0 :
                self.plotter.set_data(np.column_stack((pose_x, gt_x)), np.column_stack((pose_y, gt_y)))
                self.pose_scatter.set_offsets(np.column_stack((pose_x, pose_y)))
                self.gt_scatter.set_offsets(np.column_stack((gt_x, gt_y)))
                if self.sigma is not None:
                    self.ellipse_plotter.update_ellipse(((pose_x[-1], pose_y[-1]), sigma_x[-1], sigma_y[-1]))


                # error plot update
                self.t_plotter.set_data(self.x_indice, self.t_error)
                self.q_plotter.set_data(self.x_indice, self.q_error)
        
        x_min, x_max = np.min([np.min(pose_x), np.min(gt_x)]), np.max([np.max(pose_x), np.max(gt_x)])
        y_min, y_max = np.min([np.min(pose_y), np.min(gt_y)]), np.max([np.max(pose_y), np.max(gt_y)])


        margin = 0.1  # You can adjust this margin based on your data

        self.ax.set_xlim(x_min - margin, x_max + margin)
        self.ax.set_ylim(y_min - margin, y_max + margin)
        self.ax2.set_xlim(0, max(self.x_indice)+1)
        self.ax3.set_xlim(0, max(self.x_indice)+1)
        if self.mode == 'vo':
            self.ax2.set_ylim(0, 0.05)
            self.ax3.set_ylim(0, 30)
        else:
            self.ax2.set_ylim(0, 1)
            self.ax3.set_ylim(0, 180)
        
        
        
        if self.gts.shape[0] % 10 == 0 :
            self.end_t = time.time()
            display_fps_text = f'Display fps: {10/(self.end_t-self.start_t):.2f}'
            self.display_fps_label.setText(display_fps_text)
            model_infer_time_text = f"Model Inference Time: {self.model_infer_time*1000/10:.2f} ms"
            self.model_infer_time_label.setText(model_infer_time_text)
            self.model_infer_time = 0
            self.start_t = time.time()
        
        self.ax.relim()
        self.ax2.relim()
        self.ax3.relim()
        self.ax.autoscale_view()
        self.ax2.autoscale_view()
        self.ax3.autoscale_view()
        self.canvas.draw()
      
        # start time of display        
        # Get the next image and pose
        self.current_image_idx += 1
        if self.current_image_idx < self.data_len:
            if self.mode == "EKF":
                if self.current_image_idx % self.seq_len == 0: # change to 499 if dataset is stair
                    self.img, pose, covariance, gt = self.get_image_and_pose(self.current_image_idx, reset=True)
                else:
                    self.img, pose, covariance, gt = self.get_image_and_pose(self.current_image_idx)
                self.sigma = np.vstack([self.sigma, np.expand_dims(covariance, axis=0)])
            else:
                self.img, pose, gt = self.get_image_and_pose(self.current_image_idx)
        
            self.poses = np.vstack([self.poses, np.expand_dims(pose, axis=0)])
            self.gts = np.vstack([self.gts, np.expand_dims(gt, axis=0)])

    
   
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_P:
            # Toggle the paused flag
            self.paused = not self.paused
            if not self.paused:
                # If unpaused, restart the timer
                self.timer.start(0)
                 
    def img_denormal(self, img):
        mean = np.array([0.34721234, 0.36705238, 0.36066107])
        std = np.array([0.30737526, 0.31515116, 0.32020183])
        
        for channel in range(3):
            img[:, :, channel] = img[:, :, channel] * std[channel] + mean[channel]
        return img
        
    def get_image_and_pose(self, index, reset=False):
        if self.mode == 'EKF':
            img, pose, covariance, gt, inference_time = self.evaluator.predict_next(reset)
            self.model_infer_time += inference_time
            return img, pose.copy(), covariance.copy(), gt.copy()
        else:
            img, pose, gt, inference_time = self.evaluator.predict_next()
            self.model_infer_time += inference_time
            return img, pose.copy(), gt.copy()

    def str2tq(self, s: list[str]) -> np.ndarray:
        [x, y, z, qx, qy, qz, qw] = [float(i) for i in s]  # [x, y, z, qx, qy, qz, qw]
        t = np.array([x, y, z])  # [x, y, z, qw, qx, qy, qz]
        q = np.array([qw, qx, qy, qz])
        return np.concatenate((t, q))

    def save_csv(self, filename, data):
        print(f'save pose info to {filename}...')
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz',
                                 'gt_x', 'gt_y', 'gt_z', 'gt_qw', 'gt_qx', 'gt_qy', 'gt_qz'])
            csvwriter.writerows(data)

        # compute error statistics
        t_error_mean = np.mean(self.t_error)
        t_error_median = np.median(self.t_error)
        t_error_std = np.std(self.t_error)
        t_error_e95 = np.percentile(self.t_error, 95)
        
        q_error_mean = np.mean(self.q_error)
        q_error_median = np.median(self.q_error)
        q_error_std = np.std(self.q_error)
        q_error_e95 = np.percentile(self.q_error, 95)
        
        print("\n==========================================")
        print("Translation Error")
        print(f"E95: {round(t_error_e95, 3)}, mean: {round(t_error_mean, 3)}, median: {round(t_error_median, 3)}, std: {round(t_error_std, 3)}")
        print("\nRotation Error")
        print(f"E95: {round(q_error_e95, 3)}, mean: {round(q_error_mean, 3)}, median: {round(q_error_median, 3)}, std: {round(q_error_std, 3)}")
        print("==========================================")
        
        # Save t_error and q_error to a single CSV file
        error_data = np.column_stack((self.x_indice, self.t_error, self.q_error))
        error_filename = osp.join(self.export_dir, 'errors.csv')
        print(f'save pose info to {error_filename}...')
        with open(error_filename, 'w', newline='') as error_csvfile:
            error_csvwriter = csv.writer(error_csvfile)
            error_csvwriter.writerow(['Index', 't_error', 'q_error'])
            error_csvwriter.writerows(error_data)
        if self.mode == 'EKF':
            sigma_data = self.sigma
            sigma_filename = osp.join(self.export_dir, 'sigma.csv')    
            print(f"save sigma to {sigma_filename}...")
            with open(sigma_filename, 'w', newline='') as sigma_file:
                sigma_csvwriter = csv.writer(sigma_file)
                sigma_csvwriter.writerow(['tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                sigma_csvwriter.writerows(sigma_data)
        
    def save_figure(self, filename):
        print(f'save figure to {filename}...')
        self.figure.savefig(filename, dpi=300)

    def closeEvent(self, event):
        # Save the CSV and figure when the application is closed
        poses_filename = osp.join(self.export_dir, 'poses.csv')
        data = np.hstack([self.poses, self.gts])
        self.save_csv(poses_filename, data)
        
        figure_filename = osp.join(self.export_dir, 'figure.png')
        self.save_figure(figure_filename)
        
        
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = Visualizer('rgbd_dataset_freiburg2_pioneer_slam2')
    viewer.show()
    sys.exit(app.exec_())
