import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path, Odometry
from tf_transformations import euler_from_quaternion
from lastlab_pkg.pure_pursuit import PurePursuitController
from lastlab_pkg.utils_P import proportional_control
from lastlab_pkg.utils_P import DifferentialDriveRobot, interpolate_waypoints, RobotStates

class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')

        # Parametri del nodo
        self.Lt = 0.7  # Distanza di lookahead
        self.l = 2.0  # Parametro adattivo per la distanza di lookahead
        self.max_accel = 2.84  # Accelerazione massima per TurtleBot3
        self.target_vel = 0.22  # Velocità lineare massima per TurtleBot3
        init_pose = np.array([0.0, 0.0, 0.0])  # [x, y, yaw]
        self.path = []  # Percorso globale
        self.pind = 0  # Indice del percorso

        self.robot = DifferentialDriveRobot(init_pose)
        self.robot.x = init_pose
        self.robot.u = np.array([[0.0], [0.0]])

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
        self.debug_mode = True  # Attiva o disattiva i log di debug

    def odom_callback(self, msg):
        """Aggiorna la posa del robot dai dati di odometria."""
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])
        self.robot.x = np.array([position.x, position.y, yaw])

    def path_callback(self, msg):
        """Aggiorna il percorso globale e lo interpola."""
        self.path = [
            np.array([pose.pose.position.x, pose.pose.position.y])
            for pose in msg.poses
        ]

        if len(self.path) < 2:
            self.get_logger().error("Percorso troppo corto. Attesa di un nuovo percorso.")
            self.path = []
            self.controller = None
            return

        self.pind = 0  # Resetta l'indice del percorso

        # Interpola il percorso per ottenere waypoints con maggiore risoluzione
        self.path = interpolate_waypoints(np.array(self.path), resolution=0.1)

        # Inizializza il controller dopo aver ricevuto il percorso
        self.controller = PurePursuitController(
            robot=self.robot,
            path=self.path,
            pind=self.pind,
            Lt=self.Lt,
            vt=self.target_vel
        )

    def control_loop(self):
        """Loop di controllo per calcolare e pubblicare i comandi di velocità."""
        try:
            if not self.controller:
                return

            # Incrementa il tempo
            self.time += 1 / 15.0

            # Update adaptive lookahead distance
            self.Lt = max(0.7, self.robot.v * self.l)  # Lookahead minimo 0.7
            self.controller.Lt = self.Lt

            if self.debug_mode:
                self.get_logger().info(f"Lookahead Distance (Lt): {self.Lt}")

            # Trova il punto di lookahead
            _, goal_point = self.controller.lookahead_point()
            if goal_point is None:
                self.get_logger().warning("Percorso completato o punto di lookahead non trovato.")
                self.stop_robot()
                return

            self.get_logger().info(f"Robot Pose: {self.robot.x}")
            self.get_logger().info(f"Lookahead Point: {goal_point}")

            # Calcolare la velocità lineare e angolare
            distance_to_goal = np.linalg.norm(goal_point - self.robot.x[:2])
            self.target_vel = min(0.22, distance_to_goal / 2.0)
            acceleration = proportional_control(self.controller.target_velocity(), self.robot.v, kp=0.5)
            angular_velocity_target = self.controller.angular_velocity()
            angular_velocity = proportional_control(
            angular_velocity_target,  # Valore target
            self.robot.x[2],          # Orientamento attuale del robot (yaw)
            kp=1.0                    # Guadagno proporzionale (puoi regolarlo)
                    )
            angular_velocity = np.clip(angular_velocity, -1.0, 1.0)  # Limita tra -1 e 1

            #angular_velocity = self.controller.angular_velocity()
            #angular_velocity = max(-1.0, min(angular_velocity, 1.0))

            # Calcolare l'input del robot
            control_inputs = [acceleration, angular_velocity]

            # Aggiorna lo stato del robot
            dt = 1 / 15.0
            self.robot.update_state(control_inputs, dt)

            # Memorizza lo stato corrente
            self.states.append(self.time, self.controller.pind, self.robot, acceleration)

            # Crea il messaggio Twist da pubblicare
            cmd = Twist()
            cmd.linear.x = self.robot.v
            cmd.angular.z = angular_velocity
            self.cmd_vel_pub.publish(cmd)

            if self.debug_mode:
                self.get_logger().info(f"Linear Velocity: {self.robot.v}, Angular Velocity: {angular_velocity}")

        except Exception as e:
            self.get_logger().error(f"Errore nel loop di controllo: {e}")

    def stop_robot(self):
        """Ferma il robot pubblicando un comando Twist nullo."""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

    def proportional_control(target, current, kp):
        """Calcola un controllo proporzionale semplice."""
        return kp * (target - current)



def main():
    rclpy.init()
    node = PurePursuit()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down Pure Pursuit Node.")
        rclpy.shutdown()


if __name__ == '__main__':
    main()
