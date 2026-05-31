import rclpy

def check_for_rgb_depth(msg):
    '''
    Checks for existence of rgb and depth field in GenericProcImgAnnotator message
    '''
    message_ok = True
    if msg.rgb.width < 1:
        message_ok = False
        rclpy.logging.get_logger('v4r_util.message_checks').error('Passed RGB image is not valid')
    if msg.depth.width < 1:
        message_ok = False
        rclpy.logging.get_logger('v4r_util.message_checks').error('Passed Depth image is not valid')
    return message_ok
