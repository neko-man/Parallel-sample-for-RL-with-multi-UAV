import math
import torch
import airsim
import rospy
import numpy as np
import gymnasium as gym
import random
from gymnasium import spaces


from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge, CvBridgeError
from .net_struct import DepthCommandNet
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

def quaternion2eular(x,y,z,w):
    roll  = math.atan2(2*(w*y-z*x), w**2 - x**2 - y**2 + z**2)
    pitch = math.atan2(2*(w*x-y*z), 1 - 2*(x**2 + y**2))
    yaw   = math.atan2(2*(w*z+x*y), w**2 - x**2 - y**2 + z**2)


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=32):  # 根据FullModel的输出维度设置
        super().__init__(observation_space, features_dim)
        self.fullmodel = DepthCommandNet()
        
        # 加载预训练参数
        pretrained_weights = torch.load("./tb_logs/depth_command_net.pth",weights_only=True)
        self.fullmodel.load_state_dict(pretrained_weights)
        for param in self.fullmodel.parameters():
            param.requires_grad = False
        print("Loaded the FeaturesExtractor and Frozen it.")
        
        # 可选：冻结预训练层
        # for param in self.fullmodel.parameters():
        #     param.requires_grad = False

    def forward(self, observations):
        # 分离字典中的输入
        depth = observations["depth"]
        # odom = observations["odom"]
        
        vel = observations["local_vel"]

        angle = observations["local_eular"]

        state = torch.cat([vel,angle],dim=1)
        # state = torch.squeeze(state,0)

        # 调整维度（假设输入为[batch, H, W, C]）
        # depth = depth.permute(0, 3, 2, 1)  # -> [batch, C, H, W]
        depth = torch.unsqueeze(depth,dim=1)

        # print("depth's shape", depth.shape)
        # print("state's shape", state.shape)
        return self.fullmodel(depth,state)


class QuadMPCEnv(gym.Env):
    def __init__(self, image_shape):
        super().__init__(image_shape)
        # self.step_length = step_length
        self.image_shape = image_shape

        self.img2d = None
        self.local_pos = None
        self.local_vel = None
        self.local_eular = None
        self.collision = None

        self.ep_timesteps = 0
        self.observation_space = spaces.Dict({
            "depth": spaces.Box(low=0, high=1, shape=image_shape, dtype=np.float32),
            "local_vel": spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
            "local_eular": spaces.Box(low=-4, high = 4, shape = (3,), dtype=np.float32),
        })

        self.state = {
            "pre_position": np.zeros(3),
            "position": np.zeros(3),
            "collision": False,
            "local_vel": np.zeros(3),
            "local_eular": np.zeros(3),
        }

        self.action_space = spaces.Box(low=np.array([-1,-1,-1,-1],dtype=np.float32),
                                       high=np.array([1,1,1,1],dtype=np.float32),  #vx vy vz yaw
                                       dtype=np.float32)
        
        self.depth_subs     = rospy.Subscriber('/depth_image', Image, self.depth_callback)
        self.odom_subs      = rospy.Subscriber('/sim/odom', Odometry, self.odom_callback)
        self.collision_subs = rospy.Subscriber('collision', Bool, self.collision_callback)

        self.action_pub     = rospy.Publisher('so3_cmd', Twist, queue_size=10)

        self._setup_flight()

    def depth_callback(self,msg):
        cv_img = CvBridge.imgmsg_to_cv2(msg)

        self.img2d = np.reshape(cv_img, self.image_shape[1], self.image_shape[0])
        self.img2d[self.img2d > 10] =  10

    def odom_callback(self,msg):
        self.local_pos = msg.pose.pose.position

        self.local_vel = msg.twist.twist.linear

        self.local_eular = quaternion2eular(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                                            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

    def collision_callback(self,msg):
        self.collision = msg.data
    
    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        print("***********setup the flight***************")



    # def transform_obs(self, responses):
    #     img1d = np.array(responses[0].image_data_float, dtype=np.float)
    #     # img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
    #     img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

    #     # from PIL import Image

    #     # image = Image.fromarray(img2d)
    #     img2d[img2d > 10] = 10
    #     # if img2d.size == 0:
    #     #     return np.zeros((640,480,1))
    #     # else:
    #     #     return img2d.reshape([640, 480, 1])
    #     return img2d/10

    def _get_obs(self):
        # self.state["prev_odom"] = self.state["odometry"]
        self.state["prev_position"] = self.state["position"]

        # while(True):
        #     responses = self.drone.simGetImages([self.image_request])
        #     if len(responses[0].image_data_float) != 0:
        #         break
        # if self.img2d != None:
        #     image = self.img2d
        # else:
        #     image = None
        
        # image = self.transform_obs(responses)
        # self.drone_state = self.drone.getMultirotorState()
        self.state["position"] = self.local_pos
        # new_odom= np.array([x,y,z,w,vel_x_self,vel_y_self,vel_z_self])
        # self.state["odometry"] = np.vstack((self.state["odometry"][1:],new_odom))

        self.state["collision"] = self.collision
        # self.state["depth"] = image
        self.state["local_vel"] = self.local_vel
        self.state["local_eular"] = self.local_vel

        # print("Position now: ",self.state["position"].x_val,self.state["position"].y_val,self.state["position"].z_val)

        return {"depth":self.img2d, "local_vel":self.local_vel, "local_eular":self.local_eular}
        # return {"depth":image}

    def _do_action(self, action):
        print("****************")
        print("Prediction action:",action[0],action[1],action[2],action[3])
        # yaw_mode = airsim.YawMode(False,float(action[3]))
        # self.drone.moveByVelocityAsync(float(action[0]) * 10, float(action[1]) * 10, float(action[2])*0.5,yaw_mode=yaw_mode,duration=1)
        cmd = Twist()
        cmd.linear.x =  action[0] * 10
        cmd.linear.y =  action[1] * 10
        cmd.linear.z =  action[2] * 0.5
        cmd.angular.z = action[3]

        self.action_pub.publish(cmd)

    def _compute_reward(self):
        reward = 0
        done = False
        if self.state["collision"]:
            reward = -20
            done = True
        else:
            reward_alt = - 0.05 * np.linalg.norm(self.state["local_eular"][2])

            reward_pos  =   0.25 * (self.state["position"].x_val - self.state["prev_position"].x_val) - \
                            0.05 * abs(self.state["position"].z_val - self.state["prev_position"].z_val)
                        #   0.1 * abs(self.state["position"].y_val - self.state["prev_position"].y_val) - \
                          

            # reward_odom = - np.linalg.norm(self.state["odometry"][1] - \
            #                                self.state["odometry"][2])
            if (self.state["position"].z_val <= -8): 
                reward_pos -= 20
                done = True
            if (self.state["position"].x_val >= 80): 
                # reward_pos += 20
                done = True
            reward = reward_alt + reward_pos   #the still penalization

        return reward, done 

    def _get_info(self):
        return {
            "collision": self.state["collision"]
        }

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()
        truncated = False
        self.ep_timesteps += 1
        if done == True:
            self.ep_timesteps = 0
        else:
            if self.ep_timesteps >= 1000: truncated = True
        print("Reward: ", reward)
        print("****************\n")
        return obs, reward, done, truncated, self.state
    
    def reset(self,seed=None,options=None):
        self.ep_timesteps = 0
        self._setup_flight()
        return self._get_obs(), self._get_info()