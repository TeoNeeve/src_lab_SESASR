import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from landmark_msgs.msg import LandmarkArray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tf_transformations
from  matplotlib.patches import Arc
import yaml
from lab04.ekf_five import RobotEKF
from lab04.probabilistic_models_five import squeeze_sympy_out, sample_velocity_motion_model_five_state, landmark_sm_simpy_five_state, velocity_mm_simpy_five_state
from lab04.utils import residual
import sympy
from sympy import symbols, Matrix
from sensor_msgs.msg import Imu
from tf_transformations import quaternion_from_euler
from geometry_msgs.msg import Twist

class Lab04_node_task2(Node):
    def __init__(self):
        super().__init__("lab04_node_task2")
        self.odometry_ubscription = self.create_subscription(Odometry,"/odom",self.odom_callback,10)
        self.timer = self.create_timer(0.05,self.prediction) #20Hz
        self.landsub = self.create_subscription(LandmarkArray, "/landmarks",self.landmark_callback,10)
        self.imusub = self.create_subscription(Imu, "/imu",self.imu_callback,10)
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.wheelsub = self.create_subscription(Odometry, "/odom",self.whell_encoder_callback,10)
        self.pub_data = self.create_publisher(Odometry,'/ekf',10)
        self.get_logger().info('Controller node has started.')
        self.v = 1e-9
        self.w_z = 1e-9
        eval_gux = sample_velocity_motion_model_five_state
        _, eval_Gt, eval_Vt = velocity_mm_simpy_five_state()
        self.ekf = RobotEKF(
        dim_x = 5,
        dim_u = 2,
        eval_gux = eval_gux,
        eval_Gt = eval_Gt,
        eval_Vt = eval_Vt
        )
        x,y,theta,v,w = symbols("x y theta v w")
        hx_imu = Matrix([w])
        Ht_imu = hx_imu.jacobian(Matrix([x, y, theta, v, w]))
        self.eval_hx_imu = squeeze_sympy_out(sympy.lambdify((x, y, theta, v, w), hx_imu, "numpy"))
        self.eval_Ht_imu = squeeze_sympy_out(sympy.lambdify((x, y, theta, v, w), Ht_imu, "numpy"))
        hx_odom = Matrix([[v],[w]])
        Ht_odom = hx_odom.jacobian(Matrix([x, y, theta, v, w]))
        self.eval_hx_odom = squeeze_sympy_out(sympy.lambdify((x, y, theta, v, w), hx_odom, "numpy"))     
        self.eval_Ht_odom = squeeze_sympy_out(sympy.lambdify((x, y, theta, v, w), Ht_odom, "numpy"))
    
    def load_landmarks(self,yaml_file_path):
        with open(yaml_file_path, 'r') as file:
            data = yaml.safe_load(file)
            
        # Convert the loaded data into a dictionary with landmark IDs as keys
        landmarks = {}
        for idx, landmark_id in enumerate(data['landmarks']['id']):
            landmarks[landmark_id] = {
                'x': data['landmarks']['x'][idx],
                'y': data['landmarks']['y'][idx],
            }
        return landmarks
    
    def cmd_vel_callback(self, msg):
        """Callback to store the latest velocity command from /cmd_vel."""
        self.v = max(msg.linear.x,1e-9)
        self.w = max(msg.angular.z,1e-9)
        self.last_odom = np.array([self.v,self.w_z])

    def odom_callback(self,msg):
        vel_lin = msg.twist.twist.linear.x
        #self.v = msg.twist.twist.linear.x
        #self.w_z = msg.twist.twist.angular.z 

    def prediction(self):
        dt = 0.05
        u = np.array([self.v,self.w_z])
        sigma_u = np.array([0.05, 0.1, 0.05, 0.1, 0.025, 0.025]) # noise variance
        Mt = np.diag([sigma_u[0]*u[0]**2+sigma_u[1]*u[1]**2, sigma_u[2]*u[0]**2+sigma_u[3]*u[1]**2])
        print('faccio predict')
        self.ekf.predict_five(u,sigma_u,g_extra_args=(dt,))
        self.get_logger().info(f'Mu predict: {self.ekf.mu}')
        
    def landmark_callback(self,msg):
        ekf_msg = Odometry()
        ekf_msg.header.stamp = self.get_clock().now().to_msg()

        yaml_file_path = '/home/federico/lecture_ws/src/turtlebot3_perception/turtlebot3_perception/config/landmarks.yaml'
        landmarks = self.load_landmarks(yaml_file_path)
        std_range = 0.1  #come li scelgo?
        std_bearing = np.deg2rad(1.0)


        for key,lmark in landmarks.items():
            for measure in msg.landmarks:
                self.get_logger().info(f'landmark position = {measure}')
                if key == measure.id:
                    lmark = np.array(list(lmark.values()))
                    #self.get_logger().info(f'lmark value and type = {lmark}, {type(lmark)}')
                    z = np.array([measure.range,measure.bearing])
                    #self.get_logger().info(f'measurement = {z},id = {measure.id}')
                    eval_hx_landm, eval_Ht_landm = landmark_sm_simpy_five_state()
                    print('faccio update')
                    self.ekf.update_five(z, eval_hx=eval_hx_landm, eval_Ht=eval_Ht_landm, Qt= np.diag([std_range**2, std_bearing**2]), 
                               Ht_args=(*self.ekf.mu[:3], *lmark), # the Ht function requires a flattened array of parameters
                               hx_args=(*self.ekf.mu[:3], *lmark),
                               residual=residual, 
                               angle_idx=-1)
                    ekf_msg.pose.pose.position.x = self.ekf.mu[0]
                    ekf_msg.pose.pose.position.y = self.ekf.mu[1]
                    q = quaternion_from_euler(0,0,self.ekf.mu[2])
                    ekf_msg.pose.pose.orientation.x = q[0]
                    ekf_msg.pose.pose.orientation.y = q[1]
                    ekf_msg.pose.pose.orientation.z = q[2]
                    ekf_msg.pose.pose.orientation.w = q[3]
                    self.pub_data.publish(ekf_msg)  #va bene o no?

    def imu_callback(self, msg):
        ekf_msg = Odometry()
        ekf_msg.header.stamp = self.get_clock().now().to_msg()        

        w_measure = msg.angular_velocity.z
        std_imu = 0.1
        self.get_logger().info(f'prima valore mu: {self.ekf.mu}')
        
        # Update EKF
        self.ekf.update_five(
            z=np.array([w_measure]),  # Scalar wrapped as 1D array
            eval_hx=lambda x, y, theta, v, w: np.array([w]),  # Measurement model
            eval_Ht=lambda x, y, theta, v, w: np.array([[0, 0, 0, 0, 1]]),  # Jacobian
            Qt=np.diag([std_imu**2]), 
            Ht_args=(*self.ekf.mu,),  # Flattened state
            hx_args=(*self.ekf.mu,),  # Flattened state
            residual=residual
        )
        
        self.get_logger().info(f'dopo valore mu: {self.ekf.mu}')
        
        # Publish updated state
        ekf_msg.pose.pose.position.x = self.ekf.mu[0]
        ekf_msg.pose.pose.position.y = self.ekf.mu[1]
        q = quaternion_from_euler(0, 0, self.ekf.mu[2])
        ekf_msg.pose.pose.orientation.x = q[0]
        ekf_msg.pose.pose.orientation.y = q[1]
        ekf_msg.pose.pose.orientation.z = q[2]
        ekf_msg.pose.pose.orientation.w = q[3]
        ekf_msg.twist.twist.linear.x = self.ekf.mu[3]
        ekf_msg.twist.twist.angular.z = self.ekf.mu[4]
        
        self.pub_data.publish(ekf_msg)

       
    def whell_encoder_callback(self,msg):

        ekf_msg = Odometry()
        ekf_msg.header.stamp = self.get_clock().now().to_msg() 
        std_wheel = 0.1
        v_wheel = msg.twist.twist.linear.x
        w_wheel = msg.twist.twist.angular.z
        z_wheel = np.array([v_wheel,w_wheel])
        self.ekf.update_five(z_wheel, eval_hx=self.eval_hx_odom, eval_Ht=self.eval_Ht_odom, Qt= np.diag([std_wheel**2, std_wheel**2]), 
           Ht_args=(*self.ekf.mu,), # the Ht function requires a flattened array of parameters
           hx_args=(*self.ekf.mu,),
           residual=residual)
        #self.ekf.mu = self.ekf.mu[0]
        self.get_logger().info(f'dopo valore mu wheel: {self.ekf.mu}')
        ekf_msg.pose.pose.position.x = self.ekf.mu[0]
        ekf_msg.pose.pose.position.y = self.ekf.mu[1]
        q = quaternion_from_euler(0,0,self.ekf.mu[2])
        ekf_msg.pose.pose.orientation.x = q[0]
        ekf_msg.pose.pose.orientation.y = q[1]
        ekf_msg.pose.pose.orientation.z = q[2]
        ekf_msg.pose.pose.orientation.w = q[3]
        self.pub_data.publish(ekf_msg)  #va bene o no?

def main():
    rclpy.init()
    lab04_node_task2 = Lab04_node_task2()
    try:
        rclpy.spin(lab04_node_task2)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()









