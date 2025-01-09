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
        self.Lt = 0.5  # Distanza di lookahead iniziale ma poi viene subito cambiata in quello adattivo

        #self.kp = 0.5 #per il proportional control l'ho abbassato così ha una reazione piu lenta e riesce a ruotarsi meglio all'inizio
        #in sostanza su kp ho visto che cambia la velocita all' inizio e alla finie cioè auentandolo acelera di più la vel lineare all inizio 
        # e allo stesso modo la decelera alla fine credo. quindi dato che in alcuni casi delle simulazioni appena parte deve ruotarsi di 180 gradi
        #lho messo basso se invece deve partire dritto si puo aumentare forse (da provare meglio perche non ricordo besissimo l'ultima simulazione)
        #self.kp = 0.2 #troppo basso
        #self.kp = 3
        #self.kp = 0.2
        #prova(-6 9) :
        self.kp = 1.5

        #self.l = 2  # Parametro adattivo per la distanza di lookahead
        #self.l = 3
        #self.l = 5
        self.l = 3.8 #prova(-6 9)
        #self.l = 4.4 #aumentandola sembra andare meglio ma non riesce a fare curve strette, piu smoot ma meno preciso 
        #praticamente se lo il percoso è più smoothed non so bene come dire ma non riesce bene a fare le curve strette 
        # perche se ricordo bene o le taglia o le prende troppo larghe. se invece lo abbasso fa bene le curev strette 
        # pero non è molto preciso peche magari ruota un po di piu o u po di meno e qindi poi non riesce a riprendere il percoso retttilineo e va a sbattere

        self.max_accel = 2.84  # Accelerazione massima per TurtleBot3
        self.target_vel = 0.22  # Velocità lineare massima per TurtleBot3
        #self.target_vel = 0.15  # Velocità lineare massima per TurtleBot3 prva(-6 9)
        init_pose = np.array([0.0, 0.0, 0.0])  # [x, y, yaw]
        self.path = []  # Percorso globale
        self.pind = 0  # Indice del percorso
        #self.current_speed = 0.0  # Velocità lineare corrente

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

    def odom_callback(self, msg):
        """Aggiorna la posa del robot dai dati di odometria."""
        # position = msg.pose.pose.position
        # orientation = msg.pose.pose.orientation
        # _, _, yaw = euler_from_quaternion([
        #     orientation.x, orientation.y, orientation.z, orientation.w
        # ])
        # self.robot_pose = np.array([position.x, position.y, yaw])
        #self.current_speed = msg.twist.twist.linear.x  # Velocità lineare corrente

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
                vt=self.target_vel
            )

    def control_loop(self):
        """Loop di controllo per calcolare e pubblicare i comandi di velocità."""
        if not self.controller:
            return

        # Incrementa il tempo
        self.time += 1 / 15.0  # Aggiorna il tempo a 15 Hz

        # Update adaptive lookahead distance
        #self.Lt = max(0.5, self.robot.v * self.l) 
        self.Lt = self.robot.v * self.l
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
        #else:
            #self.get_logger().info(f"Lookahead Point: {goal_point}")

        # Calcolare la velocità lineare e angolare
        aceleration = proportional_control(self.controller.target_velocity(), self.robot.v, self.kp)
        angular_velocity = self.controller.angular_velocity()
        angular_velocity = max(-1.5, min(angular_velocity, 1.5))


        # Calcolare l'input del robot
        u = [aceleration, angular_velocity]

        # Aggiorna lo stato del robot
        dt = 1 / 15.0  # Passo temporale basato sulla frequenza del loop di controllo (15 Hz)
        self.robot.update_state(u, dt)

        # Memorizza lo stato corrente
        u = [aceleration, angular_velocity]
        self.states.append(self.time, self.controller.pind, self.robot, aceleration)

        #self.get_logger().info(f"Robot Pose: {self.robot.pose}")

        # Crea il messaggio Twist da pubblicare
        cmd = Twist()
        cmd.linear.x = self.robot.v
        cmd.angular.z = angular_velocity
        self.cmd_vel_pub.publish(cmd)

        #self.get_logger().info(f"Linear Velocity: {self.robot.v}, Angular Velocity: {angular_velocity}")
        #self.get_logger().info(f"State recorded at time: {self.time}")


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