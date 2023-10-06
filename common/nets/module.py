import torch
import torch.nn as nn
from torch.nn import functional as F
from config import cfg
from nets.layer import make_conv_layers, make_linear_layers, GraphConvBlock, GraphResBlock
from utils.smpl import SMPL


class Pose2Feat(nn.Module):
    def __init__(self, joint_num):
        super(Pose2Feat, self).__init__()
        self.joint_num = joint_num
        self.conv = make_conv_layers([64+joint_num,64])

    def forward(self, img_feat, joint_heatmap):
        feat = torch.cat((img_feat, joint_heatmap),1)
        feat = self.conv(feat)
        return feat


class PositionNet(nn.Module):
    def __init__(self):
        super(PositionNet, self).__init__()
        self.human_model = SMPL()
        self.joint_num = self.human_model.joint_num

        self.hm_shape = [cfg.output_hm_shape[0] // 8, cfg.output_hm_shape[1] // 8, cfg.output_hm_shape[2] // 8]
        self.conv = make_conv_layers([2048, self.joint_num * self.hm_shape[0]], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def soft_argmax_3d(self, heatmap3d):
        heatmap3d = heatmap3d.reshape((-1, self.joint_num, self.hm_shape[0] * self.hm_shape[1] * self.hm_shape[2]))
        heatmap3d = F.softmax(heatmap3d, 2)
        heatmap3d = heatmap3d.reshape((-1, self.joint_num, self.hm_shape[0], self.hm_shape[1], self.hm_shape[2]))

        accu_x = heatmap3d.sum(dim=(2, 3))
        accu_y = heatmap3d.sum(dim=(2, 4))
        accu_z = heatmap3d.sum(dim=(3, 4))

        accu_x = accu_x * torch.arange(self.hm_shape[2]).float().cuda()[None, None, :]
        accu_y = accu_y * torch.arange(self.hm_shape[1]).float().cuda()[None, None, :]
        accu_z = accu_z * torch.arange(self.hm_shape[0]).float().cuda()[None, None, :]

        accu_x = accu_x.sum(dim=2, keepdim=True)
        accu_y = accu_y.sum(dim=2, keepdim=True)
        accu_z = accu_z.sum(dim=2, keepdim=True)

        coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)
        return coord_out

    def forward(self, img_feat):
        # joint heatmap
        joint_heatmap = self.conv(img_feat).view(-1, self.joint_num, self.hm_shape[0], self.hm_shape[1], self.hm_shape[2])

        # joint coord
        joint_coord = self.soft_argmax_3d(joint_heatmap)

        # joint score sampling
        scores = []
        joint_heatmap = joint_heatmap.view(-1, self.joint_num, self.hm_shape[0] * self.hm_shape[1] * self.hm_shape[2])
        joint_heatmap = F.softmax(joint_heatmap, 2)
        joint_heatmap = joint_heatmap.view(-1, self.joint_num, self.hm_shape[0], self.hm_shape[1], self.hm_shape[2])
        for j in range(self.joint_num):
            x = joint_coord[:, j, 0] / (self.hm_shape[2] - 1) * 2 - 1
            y = joint_coord[:, j, 1] / (self.hm_shape[1] - 1) * 2 - 1
            z = joint_coord[:, j, 2] / (self.hm_shape[0] - 1) * 2 - 1
            grid = torch.stack((x, y, z), 1)[:, None, None, None, :]
            score_j = F.grid_sample(joint_heatmap[:, j, None, :, :, :], grid, align_corners=True)[:, 0, 0, 0, 0]  # (batch_size)
            scores.append(score_j)
        scores = torch.stack(scores)  # (joint_num, batch_size)
        joint_score = scores.permute(1, 0)[:, :, None]  # (batch_size, joint_num, 1)
        return joint_coord, joint_score

class RotationNet(nn.Module):
    def __init__(self):
        super(RotationNet, self).__init__()

        self.human_model = SMPL()
        self.joint_num = self.human_model.graph_joint_num
        self.graph_adj = torch.from_numpy(self.human_model.graph_adj).float()

        # graph convs
        self.graph_block = nn.Sequential(*[\
            GraphConvBlock(self.graph_adj, 2048+4, 128),
            GraphResBlock(self.graph_adj, 128),
            GraphResBlock(self.graph_adj, 128),
            GraphResBlock(self.graph_adj, 128),
            GraphResBlock(self.graph_adj, 128)])

        self.hm_shape = [cfg.output_hm_shape[0] // 8, cfg.output_hm_shape[1] // 8, cfg.output_hm_shape[2] // 8]

        self.shape_out = make_linear_layers([self.joint_num*128, self.human_model.shape_param_dim], relu_final=False)
        self.cam_out = make_linear_layers([self.joint_num*128,3], relu_final=False)

        self.global_rot = make_linear_layers([self.joint_num*128, 6], relu_final=False)
        self.pose_branch = nn.ModuleList()
        for _ in range(23):
            self.pose_branch.append(make_linear_layers([self.joint_num*128+2048+6, 128, 128, 6], relu_final=False))

    def sample_image_feature(self, img_feat, joint_coord_img):
        img_feat_joints = []
        for j in range(joint_coord_img.shape[1]):
            x = joint_coord_img [: ,j,0] / (self.hm_shape[2]-1) * 2 - 1
            y = joint_coord_img [: ,j,1] / (self.hm_shape[1]-1) * 2 - 1
            grid = torch.stack( (x, y),1) [:,None,None,:]
            img_feat = img_feat.float()
            img_feat_j = F.grid_sample(img_feat, grid, align_corners=True) [: , : , 0, 0] # (batch_size, channel_dim)
            img_feat_joints.append(img_feat_j)
        img_feat_joints = torch.stack(img_feat_joints) # (joint_num, batch_size, channel_dim)
        img_feat_joints = img_feat_joints.permute(1, 0 ,2) # (batch_size, joint_num, channel_dim)
        return img_feat_joints

    def forward(self, img_feat, joint_coord_img, joint_score, tree_parent):
        # pose parameter
        img_feat_joints = self.sample_image_feature(img_feat, joint_coord_img)
        
        joint_coord_img = self.human_model.reduce_joint_set(joint_coord_img)
        joint_score = self.human_model.reduce_joint_set(joint_score)
        img_feat_joints_reduce = self.human_model.reduce_joint_set(img_feat_joints)
        assert img_feat_joints_reduce.shape[1] == self.joint_num

        feat = torch.cat((img_feat_joints_reduce, joint_coord_img, joint_score),2)
        feat = self.graph_block(feat)
        feat = feat.reshape(-1, self.joint_num*128)
        # shape parameter
        shape_param = self.shape_out(feat.view(-1,self.joint_num*128))
        # camera parameter
        cam_param = self.cam_out(feat.view(-1,self.joint_num*128))

        # sequentially decoding
        pred_global_rot = self.global_rot(feat)
        pred_pose_list = []
        pred_pose_list.append(pred_global_rot)

        for joint_id in range(23):
            cur_joint = joint_id + 1
            p_joint = tree_parent[cur_joint]
            p_rot = pred_pose_list[p_joint]
            local_feat = img_feat_joints[:, joint_id+1, :]
            input_feat = torch.cat((feat, local_feat, p_rot), dim=1)
            c_rot = self.pose_branch[joint_id](input_feat) + p_rot
            pred_pose_list.append(c_rot)

        pred_param = torch.stack(pred_pose_list, dim=1)
        assert pred_param.size(1) == 24 and pred_param.size(2) == 6
        pose_param = pred_param.reshape(-1, 24*6)

        return pose_param, shape_param, cam_param

