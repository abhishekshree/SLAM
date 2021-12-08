#!/usr/bin/env python3
from KalmanFilter import KalmanFilter
import rospy
from sensor_msgs.msg import Imu
from cirs_girona_cala_viuda.msg import LinkquestDvl
from tf.transformations import euler_from_quaternion
import numpy as np

class EKFData:
    def __init__(self):
        self.orientation1 = np.zeros(3)
        self.orientation2 = np.zeros(3)
        self.velocity = np.zeros(3)

    def imu_callback(self, data):
        orientation_in_quaternion = data.orientation
        quaternion_list = [orientation_in_quaternion.x, orientation_in_quaternion.y, orientation_in_quaternion.z, orientation_in_quaternion.w]

        (roll, pitch, yaw) = euler_from_quaternion(quaternion_list)
        self.orientation1 = np.array([roll, pitch, yaw])

    def imu_callback2(self, data):
        orientation_in_quaternion = data.orientation
        quaternion_list = [orientation_in_quaternion.x, orientation_in_quaternion.y, orientation_in_quaternion.z, orientation_in_quaternion.w]

        (roll, pitch, yaw) = euler_from_quaternion(quaternion_list)
        self.orientation2 = np.array([roll, pitch, yaw])

    def dvl_callback(self, data):  
        self.velocity = np.array(data.velocityInst)
        # rospy.loginfo("DVL data received: %s", self.velocity)

    def get_data(self):
        return np.vstack((self.orientation1, self.orientation2, np.zeros(3))).T
    

if __name__ == '__main__':
    rospy.init_node('imu_subscriber', anonymous=True)
    data = EKFData()

    obj = KalmanFilter(A = np.eye(3), B = np.zeros((3,3)), C = np.eye(3), Q = np.random.rand(3,3), R = np.random.rand(3,3), mu = np.zeros(3), cov = np.random.rand(3,3))

    u = np.random.rand(3,3)

    rospy.Subscriber("/imu_xsens_mti_ros", Imu, data.imu_callback)
    rospy.Subscriber("/imu_adis_ros", Imu, data.imu_callback2)
    rospy.Subscriber("/dvl_linkquest", LinkquestDvl, data.dvl_callback)

    rate = rospy.Rate(150)
    while rospy.is_shutdown() == False:
        obj.iterate(u, data.get_data())
        rospy.loginfo("\n EKF data: %s: \n %s \n ------------- \n %s", obj.count, obj.mu, obj.cov)
        rate.sleep()
    
    rospy.spin()
