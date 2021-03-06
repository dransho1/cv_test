import rospy
import numpy as np
import math
from sensor_msgs.msg import LaserScan

class Scanner:

    def __init__(self):
        #initialize rospy
        rospy.init_node('scan_listener')
        # subscribe
        rospy.Subscriber('scan', LaserScan, self.test_callback)

    def test_callback(self, scan):
        rospy.loginfo('scan has arrived')
        angle_min = scan.angle_min
        angle_max = scan.angle_max
        angle_inc = scan.angle_increment
        scan_time = scan.scan_time
        range_min = scan.range_min
        range_max = scan.range_max
        ranges = scan.ranges
        #rospy.loginfo('%f min, %f max, %f inc, %f time',
                      #angle_min, angle_max, angle_inc, scan_time)
        #rospy.loginfo('%f meter range min, %f range meter max',
                      #range_min, range_max)
        #for i in range(len(ranges)):
            #rospy.loginfo('%f ranges', ranges[i])
        min_range = np.nanmin(ranges) # ignores nan values
        max_range = np.nanmax(ranges)
        # how close and how far things are from the camera
        rospy.loginfo('max range is: %f', max_range)
        rospy.loginfo('min range is: %f', min_range)
        index_min = np.nanargmin(ranges)
        rospy.loginfo('index of min is %d',index_min)
        angle = math.degrees(angle_min + (angle_inc * index_min))
        rospy.loginfo('angle is %f',angle)
        
    def run(self):
        rospy.spin()

# main function
if __name__ == '__main__':
    try:
        scanner = Scanner()
        scanner.run()
    except rospy.ROSInterruptException:
        pass

