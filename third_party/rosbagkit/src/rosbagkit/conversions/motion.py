def read_odometry_msg(msg) -> dict:
    """nav_msgs/Odometry"""
    return {
        "x": msg.pose.pose.position.x,
        "y": msg.pose.pose.position.y,
        "z": msg.pose.pose.position.z,
        "qx": msg.pose.pose.orientation.x,
        "qy": msg.pose.pose.orientation.y,
        "qz": msg.pose.pose.orientation.z,
        "qw": msg.pose.pose.orientation.w,
        "vx": msg.twist.twist.linear.x,
        "vy": msg.twist.twist.linear.y,
        "vz": msg.twist.twist.linear.z,
        "wx": msg.twist.twist.angular.x,
        "wy": msg.twist.twist.angular.y,
        "wz": msg.twist.twist.angular.z,
    }


def read_twist_msg(msg) -> dict:
    """geometry_msgs/Twist"""
    return {
        "vx": msg.linear.x,
        "vy": msg.linear.y,
        "vz": msg.linear.z,
        "wx": msg.angular.x,
        "wy": msg.angular.y,
        "wz": msg.angular.z,
    }


def read_twist_stamped_msg(msg) -> dict:
    """geometry_msgs/TwistStamped"""
    return {
        "vx": msg.twist.linear.x,
        "vy": msg.twist.linear.y,
        "vz": msg.twist.linear.z,
        "wx": msg.twist.angular.x,
        "wy": msg.twist.angular.y,
        "wz": msg.twist.angular.z,
    }


def read_imu_msg(msg) -> dict:
    """sensor_msgs/Imu"""
    return {
        "qx": msg.orientation.x,
        "qy": msg.orientation.y,
        "qz": msg.orientation.z,
        "qw": msg.orientation.w,
        "ax": msg.linear_acceleration.x,
        "ay": msg.linear_acceleration.y,
        "az": msg.linear_acceleration.z,
        "wx": msg.angular_velocity.x,
        "wy": msg.angular_velocity.y,
        "wz": msg.angular_velocity.z,
    }


def read_pose_with_covariance_stamped_msg(msg) -> dict:
    """geometry_msgs/PoseWithCovarianceStamped"""
    return {
        "x": msg.pose.pose.position.x,
        "y": msg.pose.pose.position.y,
        "z": msg.pose.pose.position.z,
        "qx": msg.pose.pose.orientation.x,
        "qy": msg.pose.pose.orientation.y,
        "qz": msg.pose.pose.orientation.z,
        "qw": msg.pose.pose.orientation.w,
    }
