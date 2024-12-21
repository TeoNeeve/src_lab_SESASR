import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path, Odometry
from tf_transformations import euler_from_quaternion
from lab06.PurePursuit.pure_pursuit import PurePursuitController
from lab06.PurePursuit.utils import proportional_control
from lab06.PurePursuit.utils import DifferentialDriveRobot, interpolate_waypoints, RobotStates


class PurePursuite(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')

        # Parametri del nodo
        self.Lt = 0.5  # Distanza di lookahead
        self.l = 2.0  # Parametro adattivo per la distanza di lookahead
        self.max_accel = 2.84  # Accelerazione massima per TurtleBot3
        self.max_speed = 0.22  # Velocità lineare massima per TurtleBot3
        self.robot_pose = np.array([0.0, 0.0, 0.0])  # [x, y, yaw]
        self.path = []  # Percorso globale
        self.pind = 0  # Indice del percorso
        self.current_speed = 0.0  # Velocità lineare corrente

        self.robot = DifferentialDriveRobot(init_pose=self.robot_pose)

        # Creazione del controller Pure Pursuit
        self.controller = None

        # Oggetto per memorizzare gli stati del robot
        self.states = RobotStates()

        # Subscriber
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.path_sub = self.create_subscription(
            Path, '/global_path', self.path_callback, 10
        )

        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Timer per il loop di controllo (15 Hz)
        self.timer = self.create_timer(1 / 15.0, self.control_loop)

        # Variabili per il tempo
        self.time = 0.0  # Tempo iniziale

    def odom_callback(self, msg):
        """Aggiorna la posa del robot dai dati di odometria."""
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])
        self.robot_pose = np.array([position.x, position.y, yaw])
        self.current_speed = msg.twist.twist.linear.x  # Velocità lineare corrente

    def path_callback(self, msg):
        """Aggiorna il percorso globale e lo interpola."""
        self.path = [
            np.array([pose.pose.position.x, pose.pose.position.y])
            for pose in msg.poses
        ]
        self.pind = 0  # Resetta l'indice del percorso

        # Interpola il percorso per ottenere waypoints con maggiore risoluzione
        if self.path:
            self.path = interpolate_waypoints(np.array(self.path), resolution=0.01)

            # Inizializza il controller dopo aver ricevuto il percorso
            self.controller = PurePursuitController(
                robot=self.robot,  # Passa l'oggetto robot
                path=self.path,
                pind=self.pind,
                Lt=self.Lt,
                vt=self.max_speed
            )

    def control_loop(self):
        """Loop di controllo per calcolare e pubblicare i comandi di velocità."""
        if not self.controller:
            return

        # Incrementa il tempo
        self.time += 1 / 15.0  # Aggiorna il tempo a 15 Hz

        # Update adaptive lookahead distance
        self.Lt = max(0.2, min(self.current_speed * self.l, 2.0))  # Clamp lookahead distance between 0.2 and 2.0
        self.controller.Lt = self.Lt  # Update lookahead distance in controller
        self.get_logger().info(f"Lookahead Distance (Lt): {self.Lt}")

        # Find lookahead point
        _, goal_point = self.controller.lookahead_point()
        if goal_point is None:
            self.get_logger().info("Percorso completato o punto di lookahead non trovato.")
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
            return
        else:
            self.get_logger().info(f"Lookahead Point: {goal_point}")

        # Calcolare la velocità lineare e angolare
        linear_velocity = proportional_control(self.controller.target_velocity(), self.current_speed, kp=0.5)
        angular_velocity = self.controller.angular_velocity()
        angular_velocity = max(-1.5, min(angular_velocity, 1.5))


        # Calcolare l'input del robot
        u = [linear_velocity, angular_velocity]

        # Aggiorna lo stato del robot
        dt = 1 / 15.0  # Passo temporale basato sulla frequenza del loop di controllo (15 Hz)
        self.robot.update_state(u, dt)

        # Memorizza lo stato corrente
        self.states.append(self.time, self.controller.pind, self.robot, linear_velocity)

        self.get_logger().info(f"Robot Pose: {self.robot.pose}")

        # Crea il messaggio Twist da pubblicare
        cmd = Twist()
        cmd.linear.x = linear_velocity
        cmd.angular.z = angular_velocity
        self.cmd_vel_pub.publish(cmd)

        self.get_logger().info(f"Current Speed: {self.current_speed}")
        self.get_logger().info(f"Linear Velocity: {linear_velocity}, Angular Velocity: {angular_velocity}")
        self.get_logger().info(f"State recorded at time: {self.time}")


def main():
    rclpy.init()
    node = PurePursuite()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()


# import rclpy
# from rclpy.node import Node
# import numpy as np
# from geometry_msgs.msg import Twist
# from nav_msgs.msg import Path, Odometry
# from tf_transformations import euler_from_quaternion
# from lab06.PurePursuit.pure_pursuit import PurePursuitController
# from lab06.PurePursuit.utils import proportional_control
# from lab06.PurePursuit.utils import DifferentialDriveRobot

# class PurePursuite(Node):
#     def __init__(self):
#         super().__init__('pure_pursuit_node')

#         # Parametri del nodo
#         self.Lt = 0.5  # Distanza di lookahead
#         self.l = 2.0  # Tunable parameter for adaptive lookahead
#         self.max_accel = 2.84  # Maximum acceleration for TurtleBot3
#         self.max_speed = 0.22  # Velocità lineare massima per il TurtleBot3
#         self.robot_pose = np.array([0.0, 0.0, 0.0])  # [x, y, yaw]
#         self.path = []  # Percorso globale
#         self.pind = 0  # Indice del percorso
#         self.current_speed = 0.0  # Current linear speed

#         self.robot = DifferentialDriveRobot(init_pose=self.robot_pose)

#         # Creazione del controller Pure Pursuit
#         self.controller = None

#         # Subscriber
#         self.odom_sub = self.create_subscription(
#             Odometry, '/odom', self.odom_callback, 10
#         )
#         self.path_sub = self.create_subscription(
#             Path, '/global_path', self.path_callback, 10
#         )

#         # Publisher
#         self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

#         # Timer per il loop di controllo (15 Hz)
#         self.timer = self.create_timer(1 / 15.0, self.control_loop)
        
#     def odom_callback(self, msg):
#         """Aggiorna la posa del robot dai dati di odometria."""
#         # pose = msg.pose.pose
#         # position = pose.position
#         # orientation = pose.orientation
#         # _, _, yaw = euler_from_quaternion([
#         #     orientation.x, orientation.y, orientation.z, orientation.w
#         # ])
#         # self.robot_pose = np.array([position.x, position.y, yaw])
#         self.current_speed = msg.twist.twist.linear.x  # Velocità lineare corrente

#     def path_callback(self, msg):
#         """Aggiorna il percorso globale."""
#         self.path = [
#             np.array([pose.pose.position.x, pose.pose.position.y])
#             for pose in msg.poses
#         ]
#         self.pind = 0  # Resetta l'indice del percorso

#         # Inizializza il controller dopo aver ricevuto il percorso
#         if self.path:
#             self.controller = PurePursuitController(
#                 robot=self.robot,  # Passa l'oggetto robot
#                 path=self.path,
#                 pind=self.pind,
#                 Lt=self.Lt,
#                 vt=self.max_speed
#             )
    
#     def control_loop(self):
#         """Loop di controllo per calcolare e pubblicare i comandi di velocità."""
#         if not self.controller:
#             return

#         # Update adaptive lookahead distance
#         self.Lt = max(0.2, min(self.current_speed * self.l, 2.0))  # Clamp lookahead distance between 0.2 and 2.0

#         #self.Lt = self.current_speed * self.l
#         self.controller.Lt = self.Lt  # Update lookahead distance in controller
#         self.get_logger().info(f"Lookahead Distance (Lt): {self.Lt}")

#         # Find lookahead point
#         _, goal_point = self.controller.lookahead_point()
#         if goal_point is None:
#             self.get_logger().info("Percorso completato o punto di lookahead non trovato.")
#             cmd = Twist()
#             cmd.linear.x = 0.0
#             cmd.angular.z = 0.0
#             self.cmd_vel_pub.publish(cmd)
#             return

#         else:
#             self.get_logger().info(f"Lookahead Point: {goal_point}")
        
#         # Compute linear and angular velocities
#         linear_velocity = proportional_control(self.controller.target_velocity(), self.current_speed, kp=0.5)
#         angular_velocity = self.controller.angular_velocity() 
#         u = [linear_velocity, angular_velocity]
        
#         dt = 1 / 15.0  # Time step based on the control loop frequency (15 Hz)

#         self.robot.update_state(u, dt)  # Update the robot state using DifferentialDriveRobot
#         self.get_logger().info(f"Robot Pose: {self.robot.pose}")

#         # Create the Twist message to publish
#         cmd = Twist()
#         cmd.linear.x = linear_velocity
#         # if angular_velocity < 1.5:
#         #      cmd.angular.z = angular_velocity
#         # else:
#         #    cmd.angular.z = 1.5
#         self.cmd_vel_pub.publish(cmd)

#         self.get_logger().info(f"Current Speed: {self.current_speed}")
#         self.get_logger().info(f"Linear Velocity: {linear_velocity}, Angular Velocity: {angular_velocity}")

# def main():
#     rclpy.init()
#     node = PurePursuite()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()