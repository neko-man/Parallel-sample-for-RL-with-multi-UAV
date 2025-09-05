import random
import rospy
import copy
import numpy as np
import math
from cv_bridge import CvBridge

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from std_msgs.msg import Bool


def quaternion2eular(x,y,z,w):
    roll  = math.atan2(2*(w*y-z*x), w**2 - x**2 - y**2 + z**2)
    pitch = math.atan2(2*(w*x-y*z), 1 - 2*(x**2 + y**2))
    yaw   = math.atan2(2*(w*z+x*y), w**2 - x**2 - y**2 + z**2)

    return roll, pitch, yaw

class QuadForwardEnv():
    def __init__(self, index = 0, depth_shape=(640, 480), target_distance=100.0):
        
        self.index = index
        self.target_distance = target_distance  # 目标飞行距离(米)
        node_name = f'QuadEnv_{index}'
        rospy.init_node(node_name, anonymous=None)

        # 传感器数据初始化
        self.depth_image = None
        self.odom = None
        self.collision_flag = False
        self.bridge = CvBridge()
        self.depth_shape = depth_shape

        # 状态变量
        self.position = [0.0, 0.0, 0.0]  # 当前位置 (x, y, z)
        self.prev_position = [0.0, 0.0, 0.0]  # 上一时刻位置
        self.orientation = [0.0, 0.0, 0.0]  # 欧拉角 (roll, pitch, yaw)
        self.linear_vel = [0.0, 0.0, 0.0]  # 线速度 (vx, vy, vz)
        # self.angular_vel = [0.0, 0.0, 0.0]  # 角速度 (wx, wy, wz)
        
        # 任务相关变量
        self.flight_distance = 0.0  # 已飞行距离
        self.max_steps = 2000  # 最大步数

        # -----------ROS 话题设置-------------
        # 订阅深度图
        depth_topic = f'/quad_{index}/depth_image'
        self.depth_sub = rospy.Subscriber(depth_topic, Image, self.depth_callback)
        
        # 订阅里程计
        odom_topic = f'/quad_{index}/sim/odom'
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback)
        
        # 订阅碰撞信息
        collision_topic = f'/quad_{index}/collision'
        self.collision_sub = rospy.Subscriber(collision_topic, Bool, self.collision_callback)
        
        # 发布控制指令
        cmd_topic = f'/quad_{index}/cmd'
        self.cmd_pub = rospy.Publisher(cmd_topic, Twist, queue_size=10)
        
        # 重置服务
        reset_topic = f'quad_{index}/reset'
        self.reset_sim = rospy.Publisher(reset_topic, Float32, queue_size=10)

        # 等待初始化完成
        while self.depth_image is None or self.odom is None:
            rospy.sleep(0.1)
        
        rospy.loginfo(f"QuadForwardEnv environment {index} initialized")

    def depth_callback(self, msg):
        """处理深度图像数据"""
        try:
            # 转换深度图像并缩放
            cv_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            self.depth_image = np.reshape(cv_image,self.depth_shape)
            
            # 处理异常值
            self.depth_image[np.isnan(self.depth_image)] = 10.0
            self.depth_image[np.isinf(self.depth_image)] = 10.0
            self.depth_image = np.clip(self.depth_image, 0.1, 10.0)
            
            # 归一化到[0,1]
            self.depth_image /= 10.0
            
        except Exception as e:
            rospy.logerr(f"Depth image processing error: {str(e)}")

    def odom_callback(self, msg):
        """处理里程计数据"""
        self.odom = True
        # 保存上一时刻位置
        self.prev_position = copy.deepcopy(self.position)
        
        # 更新当前位置
        self.position = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ]
        
        # 转换四元数到欧拉角
        self.orientation = quaternion2eular(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                                            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        
        # 更新速度
        self.linear_vel = (
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        )
        
        # self.angular_vel = [
        #     msg.twist.twist.angular.x,
        #     msg.twist.twist.angular.y,
        #     msg.twist.twist.angular.z
        # ]

    def collision_callback(self, msg):
        """处理碰撞信息"""
        self.collision_flag = msg.data

    def get_observations(self):
        """获取完整观测"""
        # # 深度图 (H, W)
        # depth_obs = self.depth_image if self.depth_image is not None else np.zeros(self.depth_shape)
        
        # # 速度 (vx, vy, vz) - 机体坐标系
        # velocity_obs = copy.deepcopy(self.linear_vel)
        
        # # 欧拉角 (roll, pitch, yaw)
        # orientation_obs = copy.deepcopy(self.orientation)
        
        return {
            'depth': self.depth_image,
            'velocity': self.linear_vel,
            'orientation': self.orientation
        }

    def reset(self):
        """重置环境"""

        reset_pos = Float32()
        reset_pos.data = random.uniform(-5,5)

        self.reset_sim.publish(reset_pos)
        self.collision_flag = False
        self.step_count = 0
        self.flight_distance = 0.0
        
        # 等待传感器数据更新
        rospy.sleep(0.5)
        while self.depth_image is None or self.odom is None:
            rospy.sleep(0.1)
        
        # 记录起始位置

        self.prev_position = copy.deepcopy(self.position)
        
        return self.get_observations()

    def execute_action(self, action):
        """执行动作并返回观测、奖励、完成标志"""
        # 动作应为 [vx, vy, vz, yaw_rate]
        assert len(action) == 4, "Action should be [vx, vy, vz, yaw_rate]"
        
        # 创建并发布控制指令 scaled command
        # print('action: ', action)
        cmd = Twist()
        cmd.linear.x = float(action[0]) * 10
        cmd.linear.y = float(action[1]) * 2
        cmd.linear.z = float(action[2]) * 0.1
        cmd.angular.z = float(action[3])  # 偏航角速度
        self.cmd_pub.publish(cmd)
        
        # 等待环境响应
        rospy.sleep(0.05)
        
        # 获取新观测
        obs = self.get_observations()
        
        # 更新飞行距离 - 只计算XY平面上的前进距离
        current_pos = np.array(self.position[0])
        prev_pos = np.array(self.prev_position[0])
        distance_step = np.linalg.norm(current_pos - prev_pos)
        self.flight_distance += distance_step
        
        # 计算奖励和终止条件
        reward, done, info = self.calculate_reward()
        
        # 更新步数计数器
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
            info['result'] = 'Time out'

        if done == True: self.reset()
            
        return obs, reward, done, info, current_pos

    def calculate_reward(self):
        """计算奖励函数"""
        reward = 0.0
        done = False
        info = {'result': 'In progress'}
        
        # 1. 前进奖励 (主要奖励)
        # 使用机体坐标系下的前向速度作为基础奖励
        forward_dis = self.position[0] - self.prev_position[0]
        reward += forward_dis * 0.1
        
        # 2. 碰撞惩罚
        if self.collision_flag:
            reward -= 50.0
            done = True
            info['result'] = 'Crashed'
            return reward, done, info
        
        # 3. 高度保持奖励
        target_height = 1.0  # 目标高度
        height_error = abs(self.position[2] - target_height)
        reward -= height_error * 0.05
        if self.position[2] > 10:
            reward -= 50.0
            done = True
            info['result'] = 'Overhigh'
            return reward, done, info
        
        # 4. 偏航稳定奖励 - 鼓励保持直线飞行
        yaw_error = abs(self.orientation[2])  # 偏航角
        reward -= yaw_error * 0.01
        
        # 5. 完成目标奖励
        if self.flight_distance >= self.target_distance:
            reward += 100.0
            done = True
            info['result'] = 'Success'
        
        return reward, done, info

    def shutdown(self):
        """关闭环境"""
        rospy.loginfo(f"Shutting down QuadForwardEnv environment {self.index}")
        # 发送停止指令
        cmd = Twist()
        self.cmd_pub.publish(cmd)