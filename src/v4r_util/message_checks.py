import rospy

def check_for_rgb_depth(msg):
    '''
    Checks for existence of rgb and depth field in GenericProcImgAnnotator message
    '''
    message_ok = True
    if msg.rgb.width < 1:
        message_ok = False
        rospy.logerr('Passed RGB image is not valid')
    if msg.depth.width < 1:
        message_ok = False
        rospy.logerr('Passed Depth image is not valid')
    return message_ok
