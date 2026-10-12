def read_gps_msg(msg) -> dict:
    """sensor_msgs/NavSatFix"""
    return {
        "status": msg.status.status,
        "service": msg.status.service,
        "latitude": msg.latitude,
        "longitude": msg.longitude,
        "altitude": msg.altitude,
        "position_covariance": msg.position_covariance,
        "position_covariance_type": msg.position_covariance_type,
    }
