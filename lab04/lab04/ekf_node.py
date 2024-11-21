import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from landmark_msgs.msg import LandmarkArray
from scipy.linalg import inv
import yaml
import numpy as np
import sympy
sympy.init_printing(use_latex='mathjax')
from sympy import symbols, Matrix, latex
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import cos, sin, sqrt
import matplotlib as mpl
from lab04.ekf import RobotEKF
from lab04.probabilistic_models import velocity_mm_simpy, sample_velocity_motion_model, landmark_sm_simpy
from lab04.plot_utils import plot_covariance
from geometry_msgs.msg import Twist

class EKFNode(Node):
    def __init__(self):
        super().__init__('ekf_node')

        _, eval_Gt, eval_Vt = velocity_mm_simpy()

        self.v = 1e-9
        self.w = 1e-9

        # Initialize EKF with 3 state dimensions (x, y, theta) and 2 control dimensions (v, omega)
        self.ekf = RobotEKF(dim_x=3, dim_u=2, eval_gux=sample_velocity_motion_model, eval_Gt=eval_Gt, eval_Vt=eval_Vt)

        # Load landmarks from YAML file
        yaml_file_path = '/home/federico/lecture_ws/src/turtlebot3_perception/turtlebot3_perception/config/landmarks.yaml'
        self.landmarks = self.load_landmarks(yaml_file_path)

        # Initialize odom subscriber and timer for prediction
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.ground_truth_sub = self.create_subscription(Odometry, '/ground_truth', self.ground_truth_callback, 10)
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.last_odom = np.array([self.v,self.w])
        #self.last_odom = None
        self.ekf_timer = self.create_timer(1.0 / 20 , self.prediction_step)

        self.a = np.array([0.05, 0.1, 0.05, 0.1, 0.025, 0.025])
        self.dt = 1.0 / 20

        # ROS 2 subscribers and publishers
        
        self.landmark_subscriber = self.create_subscription(LandmarkArray, '/landmarks', self.landmark_callback, 10)
        self.ekf_publisher = self.create_publisher(Odometry, '/ekf', 10)

        
        ###########################
        # Tracking data for plotting
        self.track = []
        self.track_odom = []
        self.track_ground_truth = []
        self.track_ekf = [self.ekf.mu.copy()]
        self.covariances = []


    
    def cmd_vel_callback(self, msg):
        """Callback to store the latest velocity command from /cmd_vel."""
        self.v = max(msg.linear.x,1e-9)
        self.w = max(msg.angular.z,1e-9)
        self.last_odom = np.array([self.v,self.w])
    

    def load_landmarks(self, yaml_file_path):
        with open(yaml_file_path, 'r') as file:
            data = yaml.safe_load(file)

        # Extract the coordinates and IDs from the YAML data
        ids = data['landmarks']['id']
        x_coords = data['landmarks']['x']
        y_coords = data['landmarks']['y']
        

        # Create a dictionary with landmark IDs as keys and [x, y, z] coordinates as values
        landmarks = {landmark_id: np.array([x, y]) for landmark_id, x, y in zip(ids, x_coords, y_coords)}
        
        return landmarks
    
    
    def odom_callback(self, msg):
        #linear_velocity = msg.twist.twist.linear.x
        #angular_velocity = msg.twist.twist.angular.z
        #self.last_odom = np.array([linear_velocity, angular_velocity])
        

        ############################ per plot
        self.track_odom.append(self.ekf.mu[:2].copy())  # Store odometry position for plotting
    
    def ground_truth_callback(self, msg):
        true_position = msg.pose.pose.position
        
        ############################ per plot
        self.track_ground_truth.append(self.ekf.mu[:2].copy())  # Store odometry position for plotting
        

    def landmark_callback(self, msg):
        # Iterate over each landmark in the LandmarkArray message
        for landmark in msg.landmarks:
            # Get the landmark coordinates based on its ID
            # Retrieve the landmark position for the current landmark ID
            lmark_pos = self.landmarks[landmark.id]

            # Range and bearing from the message
            z = np.array([landmark.range, landmark.bearing])

            # Perform EKF update with this landmark measurement
            print('faccio update')
            eval_hx, eval_Ht = landmark_sm_simpy()
            Ht_args=(self.ekf.mu[0], self.ekf.mu[1], self.ekf.mu[2], lmark_pos[0], lmark_pos[1])
            hx_args=(self.ekf.mu[0], self.ekf.mu[1], self.ekf.mu[2], lmark_pos[0], lmark_pos[1])
            self.ekf.update(
                z=z,
                eval_hx=eval_hx,
                eval_Ht=eval_Ht,
                Qt=np.diag([0.1, 0.1]),
                Ht_args=(self.ekf.mu[0], self.ekf.mu[1], self.ekf.mu[2], lmark_pos[0], lmark_pos[1]),
                hx_args=(self.ekf.mu[0], self.ekf.mu[1], self.ekf.mu[2], lmark_pos[0], lmark_pos[1])
            )

        # After all updates for landmarks are processed, publish the Odometry message
        self.publish_ekf_state()
        

        
        ######################################## per plot
        self.track_ekf.append(self.ekf.mu[:2].copy())  # Store EKF position for plotting
        self.covariances.append(self.ekf.Sigma.copy())  # Store covariance for plotting
        

    def prediction_step(self):
        print(self.last_odom)
        if self.last_odom is not None:
            u = self.last_odom
            sigma_u = self.a  # Assuming `self.a` as the noise parameters
            dt = self.dt  # 0.05, based on your timer setup
            print('faccio il predict')
            self.ekf.predict(u, sigma_u, g_extra_args=(dt,))
            
            ############################# per plot
            self.track.append(self.ekf.mu[:2].copy())  # Store real robot position for plotting
            


    def publish_ekf_state(self):
        """Crea e pubblica un messaggio di tipo Odometry con lo stato stimato dall'EKF."""
        if self.last_odom is None:
            self.get_logger().warn("No odometry data received yet, skipping EKF state publish.")
            return
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'map'
        odom_msg.child_frame_id = 'base_link'
        
        # Popola la posizione stimata e la covarianza
        odom_msg.pose.pose.position.x = self.ekf.mu[0]
        odom_msg.pose.pose.position.y = self.ekf.mu[1]
        odom_msg.pose.pose.position.z = 0.0

        # Popola velocità stimata
        odom_msg.twist.twist.linear.x = self.last_odom[0]
        odom_msg.twist.twist.angular.z = self.last_odom[1]

        # Pubblica il messaggio
        self.ekf_publisher.publish(odom_msg)


    def plot_localization_data(self):
        sim_step_s=0.1
        ellipse_step_s=5.0
        sim_length_s=1
        steps = int(sim_length_s / sim_step_s)
        ekf_step = max(1, int(self.dt / sim_step_s))
        ellipse_step = int(ellipse_step_s / sim_step_s)
        """Plot localization data with improved visualization."""
        track = np.array(self.track)
        track_odom = np.array(self.track_odom)
        track_ground_truth = np.array(self.track_ground_truth)
        
        # Filtra stati EKF con forma corretta
        track_ekf = np.array([state for state in self.track_ekf if state.shape == (2,)])
        covariances = self.covariances

        
        fig, ax = plt.subplots(1, 3, figsize=(9, 4))
        
        landmarks_array = np.array(list(self.landmarks.values()))  # Convert dictionary values to a NumPy array
        lmarks_legend = ax[0].scatter(
            landmarks_array[:, 0], landmarks_array[:, 1], marker="s", s=60, label="Landmarks"
        )

        fig.suptitle("EKF Robot localization, Velocity Motion Model")
        fig.tight_layout()
        track = np.array(track)

        # Plot covariance ellipses
        pri_ellipse, post_ellipse = None, None
        for i, covariance in enumerate(covariances):
            if i % 5 == 0:  # Adjust frequency of plotting ellipses
                pri_ellipse = plot_covariance(
                    (track_ekf[i, 0], track_ekf[i, 1]),
                    covariance[:2, :2],
                    std=6,
                    facecolor="k",
                    alpha=0.4,
                    ax=ax[0],
                )
                post_ellipse = plot_covariance(
                    (track_ekf[i, 0], track_ekf[i, 1]),
                    covariance[:2, :2],
                    std=6,
                    facecolor="g",
                    alpha=0.8,
                    ax=ax[0],
                )

        '''
        # plot the prior covariance ellipses every ellipse_step_s seconds
        if i % ellipse_step == 0:  
            pri_ellipse = plot_covariance(
                (ekf.mu[0], ekf.mu[1]),
                ekf.Sigma[0:2, 0:2],
                std=6,
                facecolor="k",
                alpha=0.4,
                label="Predicted Cov",
                ax=ax[0],
                )
        
        if i % ellipse_step == 0:  
            post_ellipse = plot_covariance(
                (ekf.mu[0], ekf.mu[1]),
                ekf.Sigma[0:2, 0:2],
                std=6,
                facecolor="g",
                alpha=0.8,
                label="Corrected Cov",
                ax=ax[0],
            )
        '''

        # trajectory plots
        (track_legend,) = ax[0].plot(track[:, 0], track[:, 1], label="Real robot path", color='b')
        (track_odom_legend,) = ax[0].plot(track_odom[:, 0], track_odom[:, 1], "--", label="Odometry path", color='orange')
        ax[0].axis("equal")
        ax[0].set_title("EKF Robot localization")
        ax[0].legend(handles=[lmarks_legend, track_legend, track_odom_legend, pri_ellipse, post_ellipse])

        
        # error plots
        ekf_err, =  ax[1].plot(
            np.arange(0, sim_length_s, self.dt), 
            np.linalg.norm(track[::ekf_step, :2] - track_ekf[:, :2], axis=1), 
            '-o',
            label="EKF error",
        )
        odom_err, = ax[1].plot(
            np.arange(0, sim_length_s, sim_step_s), 
            np.linalg.norm(track[:, :2] - track_odom[:, :2], axis=1), 
            label="Odometry error",
        )
        
        ax[1].legend(handles=[ekf_err, odom_err])
        ax[1].set_title("Robot path error")
        
        fig.suptitle("EKF Robot localization, Velocity Motion Model")
        fig.tight_layout()

        plt.show()        
        
        '''
        # Plot landmarks, paths
        if self.landmarks:  # Verifica se ci sono landmark disponibili
            landmarks = np.array(list(self.landmarks.values()))
            lmarks_legend = ax[0].scatter(
                landmarks[:, 0], landmarks[:, 1], marker="s", s=60, label="Landmarks"
            )
        else:
            lmarks_legend = None

        track_legend, = ax[0].plot(track[:, 0], track[:, 1], label="Real robot path", color='b')
        track_odom_legend, = ax[0].plot(track_odom[:, 0], track_odom[:, 1], "--", label="Odometry path", color='orange')
        track_ekf_legend, = ax[0].plot(track_ekf[:, 0], track_ekf[:, 1], label="EKF path", color='g')

        # Plot covariance ellipses
        pri_ellipse, post_ellipse = None, None
        for i, covariance in enumerate(covariances):
            if i % 5 == 0:  # Adjust frequency of plotting ellipses
                pri_ellipse = plot_covariance(
                    (track_ekf[i, 0], track_ekf[i, 1]),
                    covariance[:2, :2],
                    std=6,
                    facecolor="k",
                    alpha=0.4,
                    ax=ax[0],
                )
                post_ellipse = plot_covariance(
                    (track_ekf[i, 0], track_ekf[i, 1]),
                    covariance[:2, :2],
                    std=6,
                    facecolor="g",
                    alpha=0.8,
                    ax=ax[0],
                )

        ax[0].axis("equal")
        ax[0].set_title("EKF Robot Localization")
        legend_items = [item for item in [lmarks_legend, track_legend, track_odom_legend, track_ekf_legend, pri_ellipse, post_ellipse] if item is not None]
        ax[0].legend(handles=legend_items)

        # Calculate errors
        min_len = min(len(track[::5]), len(track_ekf))
        ekf_error = np.linalg.norm(track[::5, :2][:min_len] - track_ekf[:min_len, :2], axis=1)
        odom_error = np.linalg.norm(track[:, :2][:min_len] - track_odom[:min_len, :2], axis=1)

        # Error plots
        ekf_err_legend, = ax[1].plot(
            np.arange(0, len(ekf_error)) * self.dt * 5,
            ekf_error,
            "-o",
            label="EKF error",
            color="g"
        )
        odom_err_legend, = ax[1].plot(
            np.arange(0, len(odom_error)) * self.dt,
            odom_error,
            label="Odometry error",
            color="orange"
        )
        ax[1].set_title("Localization Errors")
        ax[1].legend(handles=[ekf_err_legend, odom_err_legend])

        fig.suptitle("EKF Robot Localization Visualization")
        fig.tight_layout()

        plt.show()
        '''
        
def main():
    rclpy.init()
    node = EKFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.plot_localization_data()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()