import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.resnet import ResNetBackbone
import torchgeometry as tgm
from nets.module import Pose2Feat, PositionNet, RotationNet
from nets.loss import CoordLoss, ParamLoss
from utils.smpl import SMPL
from config import cfg
import math


class Model(nn.Module):
    def __init__(self, backbone, pose2feat, position_net, rotation_net):
        super(Model, self).__init__()
        self.backbone = backbone
        self.pose2feat = pose2feat
        self.position_net = position_net
        self.rotation_net = rotation_net

        self.human_model = SMPL()
        self.human_model_layer = self.human_model.layer['neutral'].cuda()
        self.root_joint_idx = self.human_model.root_joint_idx
        self.mesh_face = self.human_model.face
        self.joint_regressor = self.human_model.joint_regressor

        self.coord_loss = CoordLoss()
        self.param_loss = ParamLoss()

    def get_camera_trans(self, cam_param, meta_info, is_render):
        # camera translation
        t_xy = cam_param[:,:2]
        gamma = torch.sigmoid(cam_param[:,2]) # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(cfg.focal[0]*cfg.focal[1]*cfg.camera_3d_size*cfg.camera_3d_size/(cfg.input_img_shape[0]*cfg.input_img_shape[1]))]).cuda().view(-1)
        if is_render:
            bbox = meta_info['bbox']
            k_value = k_value * math.sqrt(cfg.input_img_shape[0]*cfg.input_img_shape[1]) / (bbox[:, 2]*bbox[:, 3]).sqrt()
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:,None]),1)
        return cam_trans

    def make_2d_gaussian_heatmap(self, joint_coord_img):
        x = torch.arange(cfg.output_hm_shape[2])
        y = torch.arange(cfg.output_hm_shape[1])
        yy, xx = torch.meshgrid(y, x)
        xx = xx[None, None, :, :].cuda().float();
        yy = yy[None, None, :, :].cuda().float();

        x = joint_coord_img[:, :, 0, None, None];
        y = joint_coord_img[:, :, 1, None, None];
        heatmap = torch.exp(
            -(((xx - x) / cfg.sigma) ** 2) / 2 - (((yy - y) / cfg.sigma) ** 2) / 2)
        return heatmap

    def get_coord(self, smpl_pose, smpl_shape, smpl_trans):
        batch_size = smpl_pose.shape[0]
        mesh_cam, _ = self.human_model_layer(smpl_pose, smpl_shape, smpl_trans)
        # camera-centered 3D coordinate
        joint_cam = torch.bmm(torch.from_numpy(self.joint_regressor).cuda()[None,:,:].repeat(batch_size,1,1), mesh_cam)
        root_joint_idx = self.human_model.root_joint_idx

        # project 3D coordinates to 2D space
        x = joint_cam[:,:,0] / (joint_cam[:,:,2] + 1e-4) * cfg.focal[0] + cfg.princpt[0]
        y = joint_cam[:,:,1] / (joint_cam[:,:,2] + 1e-4) * cfg.focal[1] + cfg.princpt[1]
        x = x / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
        y = y / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
        joint_proj = torch.stack((x,y),2)

        mesh_cam_render = mesh_cam.clone()
        # root-relative 3D coordinates
        root_cam = joint_cam[:,root_joint_idx,None,:]
        joint_cam = joint_cam - root_cam
        mesh_cam = mesh_cam - root_cam
        return joint_proj, joint_cam, mesh_cam, mesh_cam_render

    def forward(self, inputs, targets, meta_info, mode):
        early_img_feat = self.backbone(inputs['img'])

        # get pose gauided image feature
        joint_coord_img = inputs['joints']
        with torch.no_grad():
            joint_heatmap = self.make_2d_gaussian_heatmap(joint_coord_img.detach())
            # remove blob centered at (0,0) == invalid ones
            joint_heatmap = joint_heatmap * inputs['joints_mask'][:,:,:,None]
        pose_img_feat = self.pose2feat(early_img_feat, joint_heatmap)
        pose_guided_img_feat = self.backbone(pose_img_feat, skip_early=True)  # 2048 x 8 x 8

        joint_img, joint_score = self.position_net(pose_guided_img_feat)  # refined 2D pose or 3D pose

        # estimate model parameters
        pose_param, shape_param, cam_param = self.rotation_net(pose_guided_img_feat, joint_img.detach(), joint_score.detach(), self.human_model_layer.kintree_parents)
        # change root pose 6d + latent code -> axis angles
        bs = pose_param.shape[0]
        bs, kk = pose_param.size()
        assert kk == 24 * 6
        global_rotmats = rot6d_to_rotmat(pose_param.reshape(bs*24, 6)).reshape(bs, 24, 3, 3)
        kintree_parents = self.human_model_layer.kintree_parents
        root_rotmat = global_rotmats[:, 0, :, :]
        rel_rotmats = [root_rotmat]
        for i in range(23):
            i_val = int(i + 1)
            joint_rotmat = global_rotmats[:, i_val, :, :]
            parent = kintree_parents[i_val]
            parent_rotmat = global_rotmats[:, parent, :, :]
            rel_rotmat_i = torch.matmul(parent_rotmat.permute(0, 2, 1), joint_rotmat)
            rel_rotmats.append(rel_rotmat_i)
        rel_rotmats = torch.stack(rel_rotmats, dim=1).reshape(bs*24, 3, 3)
        pose_param = rotmat_to_axis_angle(rel_rotmats).reshape(bs, 24*3)

        cam_trans = self.get_camera_trans(cam_param, meta_info, is_render=(cfg.render and (mode == 'test')))
        joint_proj, joint_cam, mesh_cam, mesh_cam_render = self.get_coord(pose_param, shape_param, cam_trans)

        if mode == 'train':
            # loss functions
            loss = {}
            # joint_img: 0~8, joint_proj: 0~64, target: 0~64
            loss['body_joint_img'] = (1/8) * self.coord_loss(joint_img*8, targets['orig_joint_img'], meta_info['orig_joint_trunc'], meta_info['is_3D'])
            loss['smpl_joint_img'] = (1/8) * self.coord_loss(joint_img*8, targets['fit_joint_img'], meta_info['fit_joint_trunc'] * meta_info['is_valid_fit'][:, None, None])

            gt_rel_rotmats = tgm.angle_axis_to_rotation_matrix(targets['pose_param'].reshape(bs*24, 3))[:, :3, :3].reshape(bs, 24, 3, 3)
            gt_global_rotmats = [gt_rel_rotmats[:, 0, :, :]]
            for i in range(23):
                i_val = int(i + 1)
                joint_rot = gt_rel_rotmats[:, i_val, :, :]
                parent = kintree_parents[i_val]
                global_rotmat_i = torch.matmul(gt_global_rotmats[parent], joint_rot)
                gt_global_rotmats.append(global_rotmat_i)
            gt_global_rotmats = torch.stack(gt_global_rotmats, dim=1).reshape(bs*24, 3, 3)
            
            gt_rotmats_valid = meta_info['is_valid_fit'][:, None].expand(-1, 24).reshape(bs*24, 1)
            loss['smpl_pose'] = torch.pow(global_rotmats.reshape(bs*24, 9) - gt_global_rotmats.reshape(bs*24, 9), 2).mean(dim=1) * gt_rotmats_valid[:, 0]

            loss['smpl_shape'] = self.param_loss(shape_param, targets['shape_param'], meta_info['is_valid_fit'][:, None])
            loss['body_joint_proj'] = (1/8) * self.coord_loss(joint_proj, targets['orig_joint_img'][:, :, :2], meta_info['orig_joint_trunc'])
            loss['body_joint_cam'] = self.coord_loss(joint_cam, targets['orig_joint_cam'], meta_info['orig_joint_valid'] * meta_info['is_3D'][:, None, None])
            loss['smpl_joint_cam'] = self.coord_loss(joint_cam, targets['fit_joint_cam'], meta_info['is_valid_fit'][:, None, None])

            return loss

        else:
            # test output
            out = {'cam_param': cam_param}
            out['joint_img'] = joint_img * 8
            out['joint_proj'] = joint_proj
            out['joint_score'] = joint_score
            out['smpl_mesh_cam'] = mesh_cam
            out['smpl_pose'] = pose_param
            out['smpl_shape'] = shape_param

            out['mesh_cam_render'] = mesh_cam_render

            if 'smpl_mesh_cam' in targets:
                out['smpl_mesh_cam_target'] = targets['smpl_mesh_cam']
            if 'bb2img_trans' in meta_info:
                out['bb2img_trans'] = meta_info['bb2img_trans']
            if 'img2bb_trans' in meta_info:
                out['img2bb_trans'] = meta_info['img2bb_trans']
            if 'bbox' in meta_info:
                out['bbox'] = meta_info['bbox']
            if 'tight_bbox' in meta_info:
                out['tight_bbox'] = meta_info['tight_bbox']
            if 'aid' in meta_info:
                out['aid'] = meta_info['aid']

            return out

def rotmat_to_axis_angle(x):
    batch_size = x.shape[0]

    rot_mat = x

    rot_mat = torch.cat([rot_mat, torch.zeros((batch_size, 3, 1)).cuda().float()], 2)  # 3x4 rotation matrix
    axis_angle = tgm.rotation_matrix_to_angle_axis(rot_mat).reshape(-1, 3)  # axis-angle
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.reshape(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)

def get_model(vertex_num, joint_num, mode):

    backbone = ResNetBackbone(cfg.resnet_type)
    pose2feat = Pose2Feat(joint_num)
    position_net = PositionNet()
    rotation_net = RotationNet()

    if mode == 'train':
        backbone.init_weights()
        pose2feat.apply(init_weights)
        position_net.apply(init_weights)
        rotation_net.apply(init_weights)
   
    model = Model(backbone, pose2feat, position_net, rotation_net)
    return model

