import rclpy
from rclpy.node import Node
from rclpy.time import Time, Duration
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import tf_transformations
from array import array # per passare da array a lista
import math

class BumpAndGoController(Node):
    def __init__(self):
        super().__init__('bump_e_go_controller')
        self.get_logger().info('Nodo bump_and_go avviato')

        # Parametri
        self.max_lin_vel = 0.22  # m/s
        self.max_ang_vel = 1.0  # rad/s
        self.control_loop_frequency = 10 # hz

        # Timer per il ciclo di controllo
        self.timer = self.create_timer(1.0 / self.control_loop_frequency, self.control_loop)

        # Publisher e subscriber
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)  # Pubblica i comandi di velocità
        self.scan_subscriber = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)  # Lidar
        self.odom_subscriber = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)  # Odometro
        self.ground_subscriber = self.create_subscription(Odometry, '/ground_truth', self.ground_truth_callback, 10) #ground_truth

        self.x= 0.0 
        self.y= 0.0 

        # Stato del robot
        self.current_velocity = Twist()
        #self.distance_limit = 0.5
        self.distance_limit = 0.7
        self.scan_ranges = []
        self.yaw = 0.0
        # self.angle_direction = 1
        self.go_forward = 1


    def ground_truth_callback(self, msg):

        true_position = msg.pose.pose.position
        
        if 5.0 < true_position.x < 8.0 and  1.9 < true_position.y < 4.0:
            self.current_velocity.linear.x = 0.0  # Ferma il robot
            self.get_logger().info("Package Delivered")
            self.cmd_vel_publisher.publish(self.current_velocity)
            self.timer.cancel()
        else:
            pass

    def scan_callback(self, msg):
        self.scan_ranges = msg.ranges.tolist() # Aggiorna i dati del laser e li mette in una lista
        self.scan_ranges = [3.5 if x == float('inf') else x for x in self.scan_ranges] # I valori "inf" li rendo 3.5 quindi il massimo che vede
        # self.get_logger().info(f'Scan ranges: {self.scan_ranges}') # per vedere la lista che manda

    def odom_callback(self, msg):
        quaternion = msg.pose.pose.orientation  # Estrai l'orientamento
        quat = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        _, _, self.yaw = tf_transformations.euler_from_quaternion(quat)  # Converte in euler

        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        

    #####################
    #GIRA CON GLI ANGOLI#
    #####################
    
    def control_loop(self):
        if not self.scan_ranges:
            return

        min_distance = min(min(self.scan_ranges[-30:]), min(self.scan_ranges[:30]))

        if self.go_forward == 1:
            self.get_logger().info('Vado avanti')
            self.current_velocity.linear.x = self.max_lin_vel

            if min_distance < self.distance_limit:
                self.get_logger().info('Ostacolo rilevato')
                self.current_velocity.linear.x = 0.0  # Ferma il robot
                self.initial_yaw = self.yaw  # Salva l'angolo di inizio rotazione
                self.choose_angle = 1
                self.go_forward = 0

        else:
            if self.choose_angle == 1:
                self.get_logger().info('Scegliendo angolo')

                left_distance = sum(self.scan_ranges[80:100])
                right_distance = sum(self.scan_ranges[260:280])

                if left_distance >= right_distance: #ho aggiunto = perche se si trova in un caso in cui la distanza è uguale da entrambi i lati gira di qui tanto per...
                    self.target_yaw = (self.initial_yaw + 1.57) % (2 * math.pi)  
                    self.current_velocity.angular.z = self.max_ang_vel  # Gira a sinistra
                    self.get_logger().info('Vado a sinistra')
                    
                else:
                    self.target_yaw = (self.initial_yaw - 1.57) % (2 * math.pi)  
                    self.current_velocity.angular.z = -self.max_ang_vel  # Gira a destra
                    self.get_logger().info('Vado a destra')
                    
                self.choose_angle = 0

            # Check if the yaw has reached or surpassed the target yaw
            yaw_difference = (self.target_yaw - self.yaw + math.pi) % (2 * math.pi) - math.pi  # Shortest angular difference

            if abs(yaw_difference) > 0.1:  # Continue turning until close to target
                self.get_logger().info('Sto girando')
                self.get_logger().info(f'angolo: {math.degrees(self.yaw)} gradi')
            else:
                self.go_forward = 1
                self.current_velocity.angular.z = 0.0  # Stop turning
                self.get_logger().info('Smetto di girare')

        self.cmd_vel_publisher.publish(self.current_velocity)
    '''
    ###################
    #GIRA CON IL TEMPO#
    ###################

    def __init__(self):
        self.choose_angle = 0
        self.last_turn_time = self.get_clock().now()
        self.turn_duration = Duration(seconds=1) # (pi*1/6)/1.5 # in realtà lo cambio un po' a caso
        # per farlo girare 30° alla volta sapendo che gira a 1,5 rad/s 

    def control_loop(self):
        if not self.scan_ranges:
            return

        min_distance = min(min(self.scan_ranges[-30:]),min(self.scan_ranges[:30])) # vede il minimo tra ultimi e primi 30 gradi
        # ho spostato le distanze da sinistra e destra per calcolarle solo prima della scela dell'angolo

        if self.go_forward == 1:
            self.get_logger().info('Vado avanti')
            self.current_velocity.linear.x = self.max_lin_vel
            # print(f'min dist: {min_distance}')
            if min_distance < self.distance_limit:
                self.get_logger().info('Ostacolo rilevato')
                self.current_velocity.linear.x = 0.0 # ferma il robot
                self.start_time = self.get_clock().now() # si salva il tempo attule di inizio giro
                self.choose_angle = 1
                self.go_forward = 0
        else:
            turn_elapsed_time = self.get_clock().now() - self.start_time # vede quanto tempo è passato da inizio giro
            if self.choose_angle == 1: # questo gli fa scegliere dove girare solo la prima volta
                self.get_logger().info('Scegliendo angolo')
                # ho fatto un comando che vede la distanza media a sinistra e destra, inutile dividere hanno stesso divisore
                left_distance = sum(self.scan_ranges[80:100]) # / len(self.scan_ranges[80:100])  # Parte sinistra
                right_distance = sum(self.scan_ranges[260:280]) # / len(self.scan_ranges[260:280]) # parte destra
                if left_distance > right_distance: # Seglie in base alla distanza minima
                    self.current_velocity.angular.z = self.max_ang_vel  # Gira a sinistra
                    self.get_logger().info('Vado a sinistra')
                    self.choose_angle = 0
                else:
                    self.current_velocity.angular.z = -self.max_ang_vel  # Gira a destra
                    self.get_logger().info('Vado a destra')
                    self.choose_angle = 0
            if turn_elapsed_time <= self.turn_duration: # vede se il tempo passato meno di quello stabilito per girare
                self.get_logger().info('Sto girando')
                self.get_logger().info(f'angolo:{self.yaw}')
            else: # nel caso è passato abbastanza tempo smette di girare
                self.go_forward = 1 # riprende ad andare avanti al prossimo loop
                self.current_velocity.angular.z = 0.0 # ferma il giro
                self.get_logger().info('Smetto di girare')
                # self.timer.cancel() ATTENZIONE: non stoppare il timer perchè termina anche quello del loop_control

        # Pubblica la velocità calcolata
        self.cmd_vel_publisher.publish(self.current_velocity)
    '''
def main():
    rclpy.init()  # Inizializza ROS2
    node = BumpAndGoController()  # Crea il nodo
    try:
        rclpy.spin(node)  # Mantiene il nodo attivo
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()  # Arresta ROS2

if __name__ == '__main__':
    main()

# export TURTLEBOT3_MODEL=burger

# colcon build
# source ./install/setup.bash
# ros2 launch turtlebot3_gazebo lab02.launch.py

# ros2 run lab02 bump_and_go