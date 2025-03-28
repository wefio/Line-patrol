#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class ImagePublisher:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('image_publisher', anonymous=True)

        # 初始化轨迹点列表
        self.trajectory_points = []  # 用于存储多个点，初始为空

        # 创建CvBridge实例
        self.bridge = CvBridge()

        # 订阅摄像头的图像主题
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)

        # 发布处理后的图像
        self.image_pub = rospy.Publisher("/camera/processed_image", Image, queue_size=10)

        # 发布控制命令
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # 初始化卡尔曼滤波器
        self.kalman = cv2.KalmanFilter(4, 2)  # 4维状态空间，2维测量
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1e-7, 0, 0, 0], [0, 1e-6, 0, 0], [0, 0, 1e-6, 0], [0, 0, 0, 1e-6]], np.float32)

        # 定义方框区域（这里定义为图像的中心区域，可以根据需要调整）
        self.roi_x1 = 0 + 200  # 左上角X坐标
        self.roi_y1 = 750  # 左上角Y坐标
        self.roi_x2 = 1950 - 200  # 右下角X坐标
        self.roi_y2 = 1000  # 右下角Y坐标

        # 记录是否看到黄色线以及旋转方向
        self.previous_direction = 0  # 0 表示没有旋转，1 表示顺时针旋转，-1 表示逆时针旋转
        self.seeing_yellow_line = False

    def image_callback(self, msg):
        try:
            # 将ROS图像消息转换为OpenCV BRG格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        # 在原图上绘制方框
        cv2.rectangle(cv_image, (self.roi_x1, self.roi_y1), (self.roi_x2, self.roi_y2), (0, 255, 0), 2)

        # 裁剪图像，只处理方框内的区域
        cropped_image = cv_image[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2]

        # 处理裁剪后的图像，寻找黄色轨迹
        hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

        # 寻找黄色轨迹的轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 如果检测到黄色线，更新状态为看到了黄色线
            self.seeing_yellow_line = True

            # 计算轨迹的中心点
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                current_point = np.array([cx, cy], np.float32)

                # 使用卡尔曼滤波进行预测和更新
                self.kalman.correct(current_point)  # 更新卡尔曼滤波器
                predicted = self.kalman.predict()  # 预测下一位置
                predicted_point = (predicted[0], predicted[1])

                # 绘制黄色轨迹和预测轨迹
                cv2.drawContours(cropped_image, [largest_contour], -1, (0, 255, 255), 2)
                cv2.circle(cropped_image, (cx, cy), 5, (0, 255, 0), -1)  # 轨迹中心点
                cv2.circle(cropped_image, (int(predicted_point[0]), int(predicted_point[1])), 5, (0, 0, 255), -1)  # 预测点

                # 更新轨迹点列表，最多保留3个点
                self.trajectory_points.append((cx, cy))
                if len(self.trajectory_points) > 3:  # 保留最近的5个点
                    self.trajectory_points.pop(0)

                # 绘制连接轨迹点的线
                for i in range(1, len(self.trajectory_points)):
                    cv2.line(cropped_image, self.trajectory_points[i-1], self.trajectory_points[i], (0, 255, 0), 2)

                # 判断是否启用卡尔曼控制
                self.control_robot(cx, cy, cv_image.shape[1])

        else:
            # 如果没有检测到黄色线，保持之前的旋转方向
            self.seeing_yellow_line = False
            self.control_robot_when_no_line()

        # 将裁剪区域绘制回原图
        cv_image[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2] = cropped_image

        # 在右上角显示小车运动状态
        self.display_robot_state(cv_image)

        # 显示原图和处理后的图像
        cv2.imshow("Original Image with ROI", cv_image)
        cv2.waitKey(1)

    def control_robot(self, cx, cy, image_width):
        # 计算偏差
        error = cx - image_width / 2

        # 调整比例增益和加入阻尼因子
        Kp = 0.3  # 调小比例增益
        damping_factor = 0.06  # 增加阻尼因子
        distance_to_center = abs(error) / (image_width / 2)  # 偏差的相对距离
        damping = damping_factor * (1 - distance_to_center)

        # 调整角速度和线速度
        angular_velocity = -Kp * error * damping  # 比例控制与阻尼
        linear_velocity = 0.3 * (1 - distance_to_center)  # 减少过近时的前进速度

        # 防止过大的角速度
        if abs(angular_velocity) > 0.15:
            angular_velocity = 0.15* np.sign(angular_velocity)

        # 如果偏差过大，则停止前进
        if abs(error) > 50:
            linear_velocity = 0

        # 发布控制命令
        cmd_msg = Twist()
        cmd_msg.linear.x = linear_velocity
        cmd_msg.angular.z = angular_velocity
        self.cmd_pub.publish(cmd_msg)

    def control_robot_when_no_line(self):
        # 没有检测到黄色线时，根据上一次的旋转方向继续旋转
        if not self.seeing_yellow_line:
            if self.previous_direction == 0:
                self.previous_direction = -1  # 初始顺时针旋转

            # 按照先前的旋转方向旋转
            angular_velocity = 0.1 * self.previous_direction  # 小角速度

            # 发布控制命令（只旋转）
            cmd_msg = Twist()
            cmd_msg.linear.x = 0  # 停止前进
            cmd_msg.angular.z = angular_velocity
            self.cmd_pub.publish(cmd_msg)

    def display_robot_state(self, cv_image):
        # 显示运动状态
        if self.previous_direction == -1:
            # 顺时针，显示右箭头
            cv2.arrowedLine(cv_image, (cv_image.shape[1] - 50, 30), (cv_image.shape[1] - 10, 30), (0, 255, 0), 5)
        elif self.previous_direction == 1:
            # 逆时针，显示左箭头
            cv2.arrowedLine(cv_image, (cv_image.shape[1] - 50, 30), (cv_image.shape[1] - 90, 30), (0, 0, 255), 5)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        image_publisher = ImagePublisher()
        image_publisher.run()
    except rospy.ROSInterruptException:
        pass
