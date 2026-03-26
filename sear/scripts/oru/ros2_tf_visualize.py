import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener


class CameraPosePublisher(Node):
    """
    A class to visualize ros2 poses. It publishes camera poses (TF converted to Pose),
    and Path to visualize the trajectory.
    """

    def __init__(
        self,
        parent_frame_topic_name: str = "map",
        child_frame_topic_name: str = "ids_camera",
        camera_link_pose_topic_name: str = "/camera_link_pose",
        camera_path_topic_name: str = "/camera_path",
    ) -> None:
        """
        Instantiates the class. The `parent_frame_topic_name` defines the topic name of
        the "main" transform, i.e. which is the root in the hierarchy. The
        `child_frame_topic_name` specifies the name of the topic we want to observe. The
        `camera_link_pose_topic_name` and `camera_path_topic_name` specify the names of
        the produces camera pose and camera path topics respectively.
        """
        super().__init__("camera_pose_publisher")

        self._parent_frame = parent_frame_topic_name
        self._child_frame = child_frame_topic_name

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._pose_pub = self.create_publisher(
            PoseStamped, camera_link_pose_topic_name, 10
        )
        self._path_pub = self.create_publisher(Path, camera_path_topic_name, 10)

        self._path = Path()
        self._path.header.frame_id = self._parent_frame

        self.timer = self.create_timer(0.05, self.timer_callback)

    def timer_callback(self) -> None:
        """
        Processes one camera TF into, and publishes associated Pose and Path.
        """

        try:
            transform = self._tf_buffer.lookup_transform(
                self._parent_frame, self._child_frame, Time(sec=0, nanosec=0)
            )

        except Exception as e:
            self.get_logger().warn(f"TF unavailable: {e}")
            return

        pose = PoseStamped()
        pose.header.stamp = transform.header.stamp
        pose.header.frame_id = self._parent_frame

        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.position.z = transform.transform.translation.z
        pose.pose.orientation = transform.transform.rotation

        self._pose_pub.publish(pose)

        self._path.header.stamp = pose.header.stamp
        self._path.poses.append(pose)
        self._path_pub.publish(self._path)


def main(
    parent_frame_topic_name: str = "map",
    child_frame_topic_name: str = "ids_camera",
    camera_link_pose_topic_name: str = "/camera_link_pose",
    camera_path_topic_name: str = "/camera_path",
) -> None:
    """
    Creates camera poses and camera path of a ros bag. The `parent_frame_topic_name`
    defines the topic name of the "main" transform, i.e. which is the root in the
    hierarchy. The `child_frame_topic_name` specifies the name of the topic we want to
    observe. The `camera_link_pose_topic_name` and `camera_path_topic_name` specify the
    names of the produces camera pose and camera path topics respectively.
    """

    rclpy.init(args=None)
    node = CameraPosePublisher(
        parent_frame_topic_name=parent_frame_topic_name,
        child_frame_topic_name=child_frame_topic_name,
        camera_link_pose_topic_name=camera_link_pose_topic_name,
        camera_path_topic_name=camera_path_topic_name,
    )
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
