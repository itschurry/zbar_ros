from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, EnvironmentVariable
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    image_topic = LaunchConfiguration("image_topic")
    qr_code_topic = LaunchConfiguration("qr_code_topic")
    throttle_repeated_barcodes = LaunchConfiguration("throttle_repeated_barcodes")
    ns = LaunchConfiguration("namespace")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "namespace",
                default_value=EnvironmentVariable("ROS_NAMESPACE", default_value=""),
                description="Node namespace (defaults to ROS_NAMESPACE if set)",
            ),
            DeclareLaunchArgument(
                "image_topic",
                default_value="camera/image/compressed",
                description="Topic to subscribe for compressed images",
            ),
            DeclareLaunchArgument(
                "qr_code_topic",
                default_value="/barcode",
                description="Topic to publish decoded barcode strings",
            ),
            DeclareLaunchArgument(
                "throttle_repeated_barcodes",
                default_value="0.0",
                description="Seconds to suppress repeated barcode detections (0 to disable)",
            ),
            Node(
                package="zbar_ros",
                executable="barcode_reader",
                name="barcode_reader",
                namespace=ns,
                parameters=[
                    {
                        "image_topic": image_topic,
                        "qr_code_topic": qr_code_topic,
                        "throttle_repeated_barcodes": ParameterValue(
                            throttle_repeated_barcodes, value_type=float
                        ),
                    }
                ],
                arguments=['--ros-args', '--log-level', 'debug'],
                output="screen",
            ),
        ]
    )
