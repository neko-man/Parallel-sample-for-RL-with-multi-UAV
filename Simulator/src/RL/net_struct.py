import math
import numpy as np
from PIL import Image
import copy
import torch
import torch.nn as nn
from torchvision import transforms

def log_normal_density(x, mean, log_std, std):
    """returns guassian density given x on log scale"""

    variance = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * variance) - 0.5 *\
        np.log(2 * np.pi) - log_std    # num_env * frames * act_size
    log_density = log_density.sum(dim=-1, keepdim=True) # num_env * frames * 1
    return log_density


# 双流神经网络模型
class DepthCommandNet(nn.Module):
    def __init__(self,action_space):
        super(DepthCommandNet, self).__init__()
        
        self.logstd = nn.Parameter(torch.zeros(action_space))

        # 图像处理分支 (输入: 1x480x640)
        self.image_encoder_actor = nn.Sequential(
            # 阶段1
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 阶段2
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 阶段3
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 阶段4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 全局特征提取
            nn.AdaptiveAvgPool2d((1, 1))  # 输出: 128x1x1
        )
        
        # 状态数据处理分支 (输入维度改为6)
        self.state_fc_actor = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # 融合后的全连接层
        self.fusion_fc_actor = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )

        self.actor = nn.Linear(64,4)

        self.image_encoder_critic = copy.deepcopy(self.image_encoder_actor)
        self.state_fc_critic      = copy.deepcopy(self.state_fc_actor)
        self.fusion_fc_critic     = copy.deepcopy(self.fusion_fc_actor)
        self.critic               = nn.Linear(64, 1)
    
    def forward(self, depth_img, state_data):
        # 图像分支处理
        # print("depth_img shape:", depth_img.shape)
        img_features = self.image_encoder_actor(depth_img)  # [batch, 128, 1, 1]
        # print("img_features' shape", img_features.shape)

        img_features = img_features.view(img_features.shape[0], -1)  # [batch, 128]

        # print("img_features' shape", img_features.shape)
        
        # 状态分支处理
        state_features = self.state_fc_actor(state_data)  # [batch, 64]
        
        # print("img_features' shape", img_features.shape)
        # print("state_features' shape", state_features.shape)

        # 特征融合
        fused = torch.cat((img_features, state_features), dim=1)  # [batch, 192]
        
        # 回归预测
        sub_action = self.fusion_fc_actor(fused)
        mean = torch.sigmoid(self.actor(sub_action))
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean,std)

        logprob = log_normal_density(action,mean,std=std,log_std=logstd)

#------------------------------------------#

        value_img = self.image_encoder_critic(depth_img)
        value_img = value_img.view(value_img.shape[0], -1)
        value_state = self.state_fc_critic(state_data)
        value_fused = torch.cat((value_img, value_state), dim=1)
        value = self.fusion_fc_critic(value_fused)
        value = self.critic(value)

        return value, action, logprob, mean
    
    def evaluate_actions(self, depth_image, state_data, action):
        v, _, _, mean = self.forward(depth_image, state_data)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy
