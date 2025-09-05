#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Bool

class CollisionDetector:
    def __init__(self):
        rospy.init_node('collision_detector', anonymous=True)
        
        # 参数配置
        self.safety_distance = rospy.get_param('~safety_distance', 0.3)  # 安全距离（米）
        self.safety_width = rospy.get_param('~safety_width', 0.3)       # 安全区域宽度（米）
        self.safety_height = rospy.get_param('~safety_height', 0.3)     # 安全区域高度（米）
        self.min_points_threshold = rospy.get_param('~min_points_threshold', 3)  # 触发碰撞的最小点数
        
        # 订阅点云话题（根据你的无人机实际话题修改）
        self.pc_sub = rospy.Subscriber("/lidar_points", PointCloud2, self.pc_callback)
        
        # 发布碰撞检测结果
        self.collision_pub = rospy.Publisher("/collision_warning", Bool, queue_size=10)
        
        rospy.loginfo("Collision detector initialized with safety distance: %.1f m", self.safety_distance)

    def pc_callback(self, msg):
        # 将PointCloud2消息转换为点坐标列表
        points = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        
        if not points:
            return
        
        # 转换为NumPy数组便于处理
        points = np.array(points)
        
        # 定义安全区域（机体坐标系）
        # 假设：x轴向前，y轴向左，z轴向上
        in_x = points[:, 0] > 0  # 只考虑前方的点
        in_range_x = points[:, 0] < self.safety_distance
        in_range_y = np.abs(points[:, 1]) < self.safety_width / 2.0
        in_range_z = np.abs(points[:, 2]) < self.safety_height / 2.0
        
        # 找出所有在安全区域内的点
        in_safety_zone = in_x & in_range_x & in_range_y & in_range_z
        
        # 计算危险区域内的点数
        danger_points = points[in_safety_zone]
        num_danger_points = len(danger_points)
        
        # 判断是否碰撞
        collision_warning = num_danger_points >= self.min_points_threshold
        
        # 发布警告
        self.collision_pub.publish(Bool(collision_warning))
        
        # 可选：打印调试信息
        if collision_warning:
            rospy.logwarn("Collision warning! Dangerous points: %d", num_danger_points)
            # 打印最近点的距离
            if num_danger_points > 0:
                min_distance = np.min(np.linalg.norm(danger_points, axis=1))
                rospy.loginfo("Min distance: %.2f m", min_distance)

if __name__ == '__main__':
    try:
        detector = CollisionDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
