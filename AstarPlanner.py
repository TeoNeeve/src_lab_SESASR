import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import os
from lastlab_pkg.utils import get_map, Movements8Connectivity, plot_costmap,plot_gridmap, world_to_map, map_to_world
from lastlab_pkg.a_star import AStar
from lastlab_pkg.planner_2d import Planner2D
from lastlab_pkg.bfs import BFS
from lastlab_pkg.dijkstra import Dijkstra
from lastlab_pkg.greedy import Greedy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, ArtistAnimation
#!/usr/bin/env python
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header



class AStarPlanner(Node):
    def __init__(self):
        super().__init__('a_star_planner')

        # Timer to periodically invoke path planning (every 2 seconds)
        self.timer = self.create_timer(2.0, self.plan_path)  # 0.5 Hz -> every 2 seconds

        relative_path = '/home/matteo/lecture_ws/src/turtlebot3_simulations/turtlebot3_gazebo/maps/project_map/map.yaml'
        print("Testing dijkstra on map1.png")
        print("current directory: ", os.getcwd())
        curr_dir = os.getcwd()
                                    
        self.map_path = os.path.join(curr_dir, relative_path)

        self.xy_reso = 3
        self.full_map, self.gridmap, self.metadata = get_map(self.map_path, self.xy_reso)

        # Generate the costmap
        self.costmap = self.generate_costmap()
        self.map_origin = [-11.000000, -11.000000, 0.0]
        self.map_shape = self.costmap.shape
        self.scaling_factor = 0.050000
        self.start = world_to_map((0, 0), self.xy_reso * self.scaling_factor, self.map_origin,self.map_shape)
        self.goal = world_to_map((7,-2), self.xy_reso * self.scaling_factor, self.map_origin,self.map_shape)
        
        print(self.costmap)

        # Initialize A* planner and set the movement strategy to 8-directional
        self.movements_class = Movements8Connectivity()  # Initialize Movements8Connectivity
        self.astar_planner = AStar(self.gridmap,self.movements_class)
        self.astar_planner.movements_class = self.movements_class  # Set the movement class for A*

        # Publisher for the global path
        self.path_pub = self.create_publisher(Path, '/global_path', 10)
        self.costmap_pub = self.create_publisher(OccupancyGrid, '/global_costmap',10)
        
        self.publish_costmap()
        # Initial path planning
        self.plan_path()

    def publish_costmap(self):
        grid = OccupancyGrid()
        grid.header = Header()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = "odom"  # Frame of reference

        grid.info.resolution = float(self.xy_reso * self.scaling_factor )  # Map resolution
        grid.info.width = self.costmap.shape[1]  # Number of columns
        grid.info.height = self.costmap.shape[0]  # Number of rows
        grid.info.origin.position.x = self.metadata.get("origin_x", -11.000000)
        grid.info.origin.position.y = self.metadata.get("origin_y", -11.000000)
        grid.info.origin.position.z = 0.0

        # Convert the costmap to a 1D list of integers in the range [-128, 127]
        costmap_clipped = np.clip(self.costmap, 0, 100)  # Ensure values are between 0 and 100
        costmap_scaled = (costmap_clipped / 100.0 * 127).astype(np.int8)  # Scale to range [0, 127]
        grid.data = costmap_scaled.flatten().tolist()  # Flatten and convert to list

        self.costmap_pub.publish(grid)
        self.get_logger().info("Costmap published.")

    def plan_path(self):

            # Plan the path using A* algorithm
            path = self.astar_planner.plan(self.start, self.goal)
            #self.get_logger().info(f"start: {self.start}, goal: {self.goal}")
            ros_path = Path()
            ros_path.header.stamp = self.get_clock().now().to_msg()
            ros_path.header.frame_id = '/odom'

            for p in path:
                pose = PoseStamped()
                pose.header = ros_path.header
                map_pos = (p[0], p[1])
                world_pos = map_to_world(map_pos, resolution=self.xy_reso * self.scaling_factor,origin=self.map_origin, map_shape=self.map_shape)
                pose.pose.position.x = world_pos[0]
                pose.pose.position.y = world_pos[1]
                ros_path.poses.append(pose)

            self.path_pub.publish(ros_path)
            self.publish_costmap()
            self.get_logger().info(f"Planned path: {path}")
            # self.get_logger().info(f"ros_path: {ros_path}")

    
    def generate_costmap(self):
        """Method to generate the cost map."""
        # Initialize the costmap based on the gridmap (no initial modification here)
        costmap = self.gridmap.copy() + 1

        # Inflate the costmap near obstacles
        for i in range(costmap.shape[0]):
            for j in range(costmap.shape[1]):
                if self.gridmap[i, j] > 0:  # If there is an obstacle
                    # Add inflation with bounds checking to avoid out of index errors
                    if i - 2 >= 0 and i + 3 < costmap.shape[0] and j - 2 >= 0 and j + 3 < costmap.shape[1]:
                        costmap[i - 2:i + 3, j - 2:j + 3] += 10  # Larger inflation
                    if i - 1 >= 0 and i + 2 < costmap.shape[0] and j - 1 >= 0 and j + 2 < costmap.shape[1]:
                        costmap[i - 1:i + 2, j - 1:j + 2] += 20  # Smaller inflation

        # Set the cost of obstacles to 100 (maximum cost)
        costmap[self.gridmap > 0] = 100

        # Ensure the costmap values are within [0, 100]
        costmap = np.clip(costmap, 0, 100)

        return costmap
    
def main():
    rclpy.init()
    node = AStarPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()