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
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist

class EKFNode(Node):
    def __init__(self):
        super().__init__('ekf_node')

        _, eval_Gt, eval_Vt = velocity_mm_simpy()


        # Initialize EKF with 5 state dimensions (x, y, theta,v,omega) and 2 control dimensions (v, omega)
        self.ekf = RobotEKF(dim_x=5, dim_u=2, eval_gux=sample_velocity_motion_model, eval_Gt=self.custom_eval_Gt, eval_Vt=self.custom_eval_Vt)
        self.ekf.mu = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # Stato iniziale [x, y, theta, v, omega]
        self.ekf.Sigma = np.diag([0.1, 0.1, np.deg2rad(5), 0.05, 0.05])  # Covarianza iniziale

        # Load landmarks from YAML file
        yaml_file_path = '/home/federico/lecture_ws/src/turtlebot3_perception/turtlebot3_perception/config/landmarks.yaml'
        self.landmarks = self.load_landmarks(yaml_file_path)

        # Initialize odom subscriber and timer for prediction
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_correction, 10)
        self.last_odom = None
        self.ekf_timer = self.create_timer(1.0 / 20 , self.prediction_step)
        self.imu_sub = self.create_subscription(Imu, '/imu',self.imu_correction, 10)

        self.a = np.array([0.05, 0.1, 0.05, 0.1, 0.025, 0.025])
        self.dt = 1.0 / 20

        # ROS 2 subscribers and publishers
        self.landmark_subscriber = self.create_subscription(LandmarkArray, '/landmarks', self.landmark_callback, 10)
        self.ekf_publisher = self.create_publisher(Odometry, '/ekf', 10)
        
        ###########################
        # Tracking data for plotting
        self.track = []
        self.track_cmd_vel = []
        self.track_ekf = [self.ekf.mu.copy()]
        self.covariances = []
        

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

    
    def custom_eval_Gt(x, u, dt):
        """
        Calcola il Jacobiano Gt per lo stato esteso.
        """
        # Estrai variabili: x = [x, y, theta, v, omega]
        theta = x[2]
        v = x[3]
        omega = x[4]

        # Jacobiano Gt calcolato manualmente (modificato per stato esteso)
        Gt = np.eye(5)
        Gt[0, 2] = -v * sin(theta) * dt
        Gt[0, 3] = cos(theta) * dt
        Gt[1, 2] = v * cos(theta) * dt
        Gt[1, 3] = sin(theta) * dt
        Gt[2, 4] = dt  # Aggiornamento angolare
        return Gt

    def custom_eval_Vt(x, u, dt):
        """
        Calcola il Jacobiano Vt per lo stato esteso.
        """
        theta = x[2]
        Vt = np.zeros((5, 2))
        Vt[0, 0] = cos(theta) * dt
        Vt[1, 0] = sin(theta) * dt
        Vt[2, 1] = dt
        Vt[3, 0] = 1
        Vt[4, 1] = 1
        return Vt

    
    
    def cmd_vel_callback(self, msg):
        linear_velocity = msg.linear.x
        angular_velocity = msg.angular.z
        self.last_odom = np.array([linear_velocity, angular_velocity])

        self.cmd_vel_correction(msg)
        ############################ per plot
        self.track_cmd_vel.append(self.ekf.mu[:2].copy())  # Store cmd position for plotting
        

    def landmark_callback(self, msg):
        # Iterate over each landmark in the LandmarkArray message
        for landmark in msg.landmarks:
            # Get the landmark coordinates based on its ID
            if landmark.id not in self.landmarks:
                print(f"Landmark ID {landmark.id} not found in YAML file.")
                continue

            # Retrieve the landmark position for the current landmark ID
            lmark_pos = self.landmarks[landmark.id]

            # Range and bearing from the message
            z = np.array([landmark.range, landmark.bearing])

            # Perform EKF update with this landmark measurement
            print('faccio update')
            eval_hx, eval_Ht = landmark_sm_simpy()
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
        
    
    def cmd_vel_correction(self,msg):
        """
        Correzione basata su dati dell'odom per v e omega.
        """
        v_measured =  msg.linear.x
        omega_measured = msg.angular.z
        z = np.array([v_measured, omega_measured])

        # Definisci Ht e Qt per l'odom
        Ht_cmd = np.zeros((2, 5))
        Ht_cmd[0, 3] = 1  # Jacobiano per v
        Ht_cmd[1, 4] = 1  # Jacobiano per omega

        Qt_odom = np.diag([0.05, 0.05])  # Incertezza odom

        self.ekf.update(
        z=z,
        eval_hx=lambda *args: z,  # Misura diretta
        eval_Ht=lambda *args: Ht_cmd,
        Qt=Qt_odom,
        )

    def imu_correction(self, imu_msg):
        """
        Correzione basata su dati dell'IMU per omega.
        """
        omega_measured = imu_msg.angular_velocity.z
        z = np.array([omega_measured])

        # Definisci Ht e Qt per l'IMU
        Ht_imu = np.zeros((1, 5))
        Ht_imu[0, 4] = 1  # Jacobiano per omega

        Qt_imu = np.diag([0.02])  # Incertezza IMU

        self.ekf.update(
        z=z,
        eval_hx=lambda *args: z,  # Misura diretta
        eval_Ht=lambda *args: Ht_imu,
        Qt=Qt_imu,
        )


    def prediction_step(self):
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
        """Plot localization data after node termination."""
        track = np.array(self.track)
        track_odom = np.array(self.track_odom)
        
        # Filtra stati EKF con forma corretta
        track_ekf = np.array([state for state in self.track_ekf if state.shape == (2,)])
        covariances = self.covariances

        if len(track_ekf) == 0:
            print("Errore: Nessuno stato EKF valido trovato per il plotting.")
            return

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Plot landmarks, paths
        ax[0].plot(track[:, 0], track[:, 1], label="Real path", color='b')
        ax[0].plot(track_odom[:, 0], track_odom[:, 1], label="Odometry path", color='orange', linestyle='--')
        ax[0].plot(track_ekf[:, 0], track_ekf[:, 1], label="EKF path", color='g')

        # Plot covariance ellipses
        for i, covariance in enumerate(covariances):
            if i % 5 == 0:  # Adjust frequency of plotting ellipses
                plot_covariance(
                    (track_ekf[i, 0], track_ekf[i, 1]),
                    covariance[:2, :2],
                    std=6,
                    ax=ax[0]
                )

        ax[0].set_title("EKF Robot Localization")
        ax[0].axis("equal")
        ax[0].legend()

        # Ensure arrays have same length for error calculation
        min_len = min(len(track[::5]), len(track_ekf))
        ekf_error = np.linalg.norm(track[::5, :2][:min_len] - track_ekf[:min_len, :2], axis=1)
        odom_error = np.linalg.norm(track[:, :2][:min_len] - track_odom[:min_len, :2], axis=1)

        ax[1].plot(ekf_error, label="EKF error")
        ax[1].plot(odom_error, label="Odometry error")
        ax[1].set_title("Localization Errors")
        ax[1].legend()

        plt.suptitle("EKF Robot Localization Visualization")
        plt.show()
        
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