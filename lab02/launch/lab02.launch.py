# my_package/launch/bump_and_go_launch.py

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='lab02',
            executable='bump_and_go',
            name='bump_and_go_node',
            output='screen',
        ),
    ])