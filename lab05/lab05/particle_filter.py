import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from landmark_msgs.msg import LandmarkArray
from scipy.linalg import inv
import yaml
import numpy as np
import sympy
from lab05.pf import RobotPF
from lab05.probabilistic_models import sample_velocity_motion_model,landmark_range_bearing_model
from geometry_msgs.msg import Twist
from tf_transformations import quaternion_from_euler
from lab05.utils import residual, state_mean, simple_resample, stratified_resample, systematic_resample, residual_resample   

class PFNode(Node):
    def __init__(self):
        super().__init__('particle_filter')

        self.v = 1e-9
        self.w = 1e-9

        
        self.pf = RobotPF(dim_x=3, 
                          dim_u=2, 
                          eval_gux=sample_velocity_motion_model,
                          resampling_fn=systematic_resample,
                          boundaries=[(-3.0, 3.0), (-3.0, 3.0), (-np.pi, np.pi)],  
                          N=1000)
        self.pf.initialize_particles()

        yaml_file_path = '/home/federico/lecture_ws/src/turtlebot3_perception/turtlebot3_perception/config/landmarks.yaml'
        self.landmarks = self.load_landmarks(yaml_file_path)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.ground_truth_sub = self.create_subscription(Odometry, '/ground_truth', self.ground_truth_callback, 10)
        self.pf_publisher = self.create_publisher(Odometry, '/pf', 10)
        self.latest_vel = np.array([self.v,self.w])
        self.pf_timer = self.create_timer(1.0 / 20 , self.prediction_step)
        std_lin_vel = 0.1  # [m/s]
        std_ang_vel = np.deg2rad(1.0)  # [rad/s]
        self.a = np.array([std_lin_vel, std_ang_vel])
        self.dt = 1.0 / 20  
        self.landmark_subscriber = self.create_subscription(LandmarkArray, '/landmarks', self.landmark_callback, 10)
    
    def cmd_vel_callback(self, msg):
        
        if abs(msg.linear.x) < 1e-9:
            self.v = 1e-9
        else:
            self.v = msg.linear.x
        
        if abs(msg.linear.x) < 1e-9:
            self.w = 1e-9
        else:
            self.w = msg.angular.z

        self.latest_vel = np.array([self.v,self.w])
    
    def ground_truth_callback(self, msg):
        true_position = msg.pose.pose.position
    

    def load_landmarks(self, yaml_file_path):
        with open(yaml_file_path, 'r') as file:
            data = yaml.safe_load(file)

        ids = data['landmarks']['id']
        x_coords = data['landmarks']['x']
        y_coords = data['landmarks']['y']

        landmarks = {landmark_id: np.array([x, y]) for landmark_id, x, y in zip(ids, x_coords, y_coords)}
        
        return landmarks
    
    
    def odom_callback(self, msg):
        return

    def landmark_callback(self, msg):

        eval_hx_landm = landmark_range_bearing_model
        std_range = 0.1  # [m]
        std_bearing = np.deg2rad(1.0)  # [rad]
        sigma_z = np.array([std_range, std_bearing])

        for landmark in msg.landmarks:
            lmark_pos = self.landmarks[landmark.id]

            z = np.array([landmark.range, landmark.bearing])
            self.pf.update(z, sigma_z, eval_hx=eval_hx_landm, hx_args=(lmark_pos, sigma_z))
            print('faccio update')

        # Normalize weights
        self.pf.normalize_weights()

        # Resample particles if necessary
        neff = self.pf.neff()

        if neff < self.pf.N / 2:
            self.pf.resampling(
                self.pf.resampling_fn,  # simple, residual, stratified, systematic
                resampling_args=(self.pf.weights,),  # tuple: only pf.weights if using pre-defined functions
            )

        # Publish the estimated state
        self.publish_estimated_state()


    def prediction_step(self):
        self.pf.predict(self.latest_vel, sigma_u=self.a, g_extra_args=(self.dt,))
        print('faccio il predict')


    def publish_estimated_state(self):

        self.pf.estimate(mean_fn=state_mean, residual_fn=residual, angle_idx=2)

        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        
        odom_msg.pose.pose.position.x = self.pf.mu[0]
        odom_msg.pose.pose.position.y = self.pf.mu[1]
        odom_msg.pose.pose.position.z = 0.0
        q = quaternion_from_euler(0,0,self.pf.mu[2])
        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = q[2]

        odom_msg.twist.twist.linear.x = self.latest_vel[0]
        odom_msg.twist.twist.angular.z = self.latest_vel[1]

        self.pf_publisher.publish(odom_msg)


def main():
    rclpy.init()
    node = PFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
    

if __name__ == '__main__':
    main()

