# salvare e colcon build per costruire prima di eseguire
import rclpy # importa libreria nodi per python
from rclpy.node import Node

from turtlesim.msg import Pose # voglio vedere la pose della turlesim

class MyNode(Node): # deninosco una classe da node
    def __init__(self):
        super().__init__('my_node')
        self.get_logger().info('Hi from my_node.') # il log di ros dal node
        # ora creiamo una subscription e deve avere type topic e callback
        self.create_subscription(Pose, '/turtle1/pose', self.pose_callback, 10)

    def pose_callback(self, msg): # craiamo le callback per sapere la posizione
        self.get_logger().info(f'x: {msg.x}, y: {msg.y}, theta: {msg.theta}', throttle_duration_sec=1.0)

def main():
    rclpy.init() # initialize ros
    node = MyNode() # creo un nodo con la classe
    try:
        rclpy.spin(node) # spin aspetta input 
    except KeyboardInterrupt:
        pass # fino a ctrl+C per interrompere
    finally:
        rclpy.try_shutdown() # interrompe

if __name__ == '__main__':
    main()
