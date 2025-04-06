# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
from time import time
from skimage.io import imread
import cv2
import pickle
from loguru import logger
from .utils.renderer import SRenderY, set_rasterizer
from .models.encoders import ResnetEncoder
from .models.FLAME import FLAME, FLAMETex
from .models.decoders import Generator
from .utils import util
from .utils.rotation_converter import batch_euler2axis
from .utils.config import cfg

class DECA_EFT(nn.Module):
    def __init__(self, config=None, device='cuda'):
        super(DECA_EFT, self).__init__()
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = torch.device('cuda')
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size

        self._create_model(self.cfg.model)
        self._setup_renderer(self.cfg.model)

    def _setup_renderer(self, model_cfg):
        set_rasterizer(self.cfg.rasterizer_type)
        self.render = SRenderY(
            self.image_size, 
            obj_filename=model_cfg.topology_path, 
            uv_size=model_cfg.uv_size,
            rasterizer_type=self.cfg.rasterizer_type
        ).to(self.device)

    def _create_model(self, model_cfg):
        # set up parameters
        self.n_param = model_cfg.n_shape + model_cfg.n_tex + model_cfg.n_exp + model_cfg.n_pose + model_cfg.n_cam + model_cfg.n_light
        self.n_detail = model_cfg.n_detail
        self.n_cond = model_cfg.n_exp + 3  # exp + jaw pose
        self.num_list = [model_cfg.n_shape, model_cfg.n_tex, model_cfg.n_exp, model_cfg.n_pose, model_cfg.n_cam,
                         model_cfg.n_light]
        self.param_dict = {i: model_cfg.get('n_' + i) for i in model_cfg.param_list}
        
        # encoders
        self.E_flame = ResnetEncoder(outsize=self.n_param).to(self.device)
        
        # decoders
        self.flame = FLAME(model_cfg).to(self.device)
        
        model_path = self.cfg.pretrained_modelpath
        if os.path.exists(model_path):
            logger.info(f'DECA: Found trained model. Load `{model_path}`')
            checkpoint = torch.load(model_path)
            self.checkpoint = checkpoint
            util.copy_state_dict(self.E_flame.state_dict(), checkpoint['E_flame'])
        else:
            logger.error(f'please check model path: {model_path}')
        self.E_flame.train()
        self.exemplerTrainingMode()

    def exemplerTrainingMode(self):
        '''
        from EFT code
        https://github.com/facebookresearch/eft/blob/c8400a6053dbc93771f2c0b3ce6686c357ff26c1/eft/train/eftFitter.py#L191
        '''
        for module in self.E_flame.modules():
            if not type(module):
                continue
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                # print(module)
                module.eval()
                for m in module.parameters():
                    m.requires_grad = False
            if isinstance(module, nn.Dropout):
                # print(module)
                module.eval()
                for m in module.parameters():
                    m.requires_grad = False

    def return_flame(self):
        return self.flame

    def decompose_code(self, code, num_dict):
        ''' Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        '''
        code_dict = {}
        start = 0
        for key in num_dict:
            end = start + int(num_dict[key])
            code_dict[key] = code[:, start:end]
            start = end
            if key in ['light', 'tex']:
                continue
                # code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
        return code_dict

    def encode(self, images):
        parameters = self.E_flame(images)
        codedict = self.decompose_code(parameters, self.param_dict)
        codedict['images'] = images

        if self.cfg.model.jaw_type == 'euler':
            posecode = codedict['pose']
            euler_jaw_pose = posecode[:, 3:].clone()  # x for yaw (open mouth), y for pitch (left ang right), z for roll
            posecode[:, 3:] = batch_euler2axis(euler_jaw_pose)
            codedict['pose'] = posecode
            codedict['euler_jaw_pose'] = euler_jaw_pose
        return codedict


    def decode(self, codedict, return_vis=False, fixed_cam=None):
        ## decode
        verts, landmarks2d, landmarks3d = self.flame(
            shape_params=codedict['shape'], 
            expression_params=codedict['exp'],
            pose_params=codedict['pose']
        )
        landmarks3d_world = landmarks3d.clone()

        ## projection
        landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:, :, :2]
        landmarks2d[:, :, 1:] = -landmarks2d[:, :, 1:]  # ; landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2
        landmarks3d = util.batch_orth_proj(landmarks3d, codedict['cam'])
        landmarks3d[:, :, 1:] = -landmarks3d[:, :, 1:]  # ; landmarks3d = landmarks3d*self.image_size/2 + self.image_size/2

        cam = codedict['cam'] # [batch, 3]
        trans_verts = util.batch_orth_proj(verts, cam)
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]

        if fixed_cam is not None:
            trans_verts_fix = util.batch_orth_proj(verts, fixed_cam)
            trans_verts_fix[:, :, 1:] = -trans_verts_fix[:, :, 1:]
            opdict = {
                'verts': verts,
                'trans_verts': trans_verts,
                'landmarks2d': landmarks2d,
                'landmarks3d': landmarks3d,
                'landmarks3d_world': landmarks3d_world,
                'fix_trans_verts': trans_verts_fix
            }
        else:
            opdict = {
                'verts': verts,
                'trans_verts': trans_verts,
                'landmarks2d': landmarks2d,
                'landmarks3d': landmarks3d,
                'landmarks3d_world': landmarks3d_world,
            }
        return opdict

    def visualize(self, visdict, size=224, dim=2):
        '''
        image range should be [0,1]
        dim: 2 for horizontal. 1 for vertical
        '''
        assert dim == 1 or dim == 2
        grids = {}
        for key in visdict:
            _, _, h, w = visdict[key].shape
            if dim == 2:
                new_h = size;
                new_w = int(w * size / h)
            elif dim == 1:
                new_h = int(h * size / w);
                new_w = size
            grids[key] = torchvision.utils.make_grid(F.interpolate(visdict[key], [new_h, new_w]).detach().cpu())
        grid = torch.cat(list(grids.values()), dim)
        grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
        return grid_image

    def save_obj(self, filename, opdict):
        '''
        vertices: [nv, 3], tensor
        texture: [3, h, w], tensor
        '''
        i = 0
        vertices = opdict['verts'][i].cpu().numpy()
        faces = self.render.faces[0].cpu().numpy()
        texture = util.tensor2image(opdict['uv_texture_gt'][i])
        uvcoords = self.render.raw_uvcoords[0].cpu().numpy()
        uvfaces = self.render.uvfaces[0].cpu().numpy()
        # save coarse mesh, with texture and normal map
        normal_map = util.tensor2image(opdict['uv_detail_normals'][i] * 0.5 + 0.5)
        util.write_obj(filename, vertices, faces,
                       texture=texture,
                       uvcoords=uvcoords,
                       uvfaces=uvfaces,
                       normal_map=normal_map)
        # upsample mesh, save detailed mesh
        texture = texture[:, :, [2, 1, 0]]
        normals = opdict['normals'][i].cpu().numpy()
        displacement_map = opdict['displacement_map'][i].cpu().numpy().squeeze()
        dense_vertices, dense_colors, dense_faces = util.upsample_mesh(vertices, normals, faces, displacement_map,
                                                                       texture, self.dense_template)
        util.write_obj(filename.replace('.obj', '_detail.obj'),
                       dense_vertices,
                       dense_faces,
                       colors=dense_colors,
                       inverse_face_order=True)


    def model_dict(self):
        return {
            'E_flame': self.E_flame.state_dict()
        }