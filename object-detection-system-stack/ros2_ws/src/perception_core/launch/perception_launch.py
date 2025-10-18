
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='perception_core',
            executable='camera_pub',
            name='camera_pub'
        ),
        Node(
            package='perception_core',
            executable='detector_node',
            name='detector_node'
        ),
        Node(
            package='perception_core',
            executable='tracker_node',
            name='tracker_node'
        ),
        Node(
            package='perception_core',
            executable='visualizer_node',
            name='visualizer_node'
        ),
    ])
