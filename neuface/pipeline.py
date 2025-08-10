import os
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import numpy as np
from loguru import logger
from tqdm import tqdm
# from .utils.tensor_cropper import transform_points
import time
from .utils import util
from .utils.config import cfg
import shutil
from .utils import lossfunc
from .datasets.mead import MEAD
from .datasets.celebv_hq import CELEBV_HQ
from .datasets.nersemble import Nersemble
import wandb

import datetime, sys, copy

from .utils import run_facer_segmentation
# from .utils import network_inference
from PIL import Image
from torchvision.transforms.functional import gaussian_blur
from .utils.rotation_converter import batch_axis2matrix

from .utils.drawing import plot_points

class NeuFacePipeline(object):
    def __init__(self, model, config=None, device='cuda:0'):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = device
        self.batch_size = self.cfg.dataset.batch_size
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size

        # deca model
        self.deca = model.to(self.device)
        self.configure_optimizers()
        self.load_checkpoint()

        # FLAME model
        self.face_model = model.return_flame()

        # uv loss
        self.uv_loss_fn = lossfunc.UVLoss(model_cfg=self.cfg.model,
                                stricter_mask=self.cfg.loss.stricter_uv_mask,
                                delta_uv= self.cfg.loss.delta_uv,
                                dist_uv=self.cfg.loss.dist_uv)

        self.masking = lossfunc.Masking(model_cfg=self.cfg.model)
        self.eye_mask = self.masking.get_mask_eyes_rendering().cuda()
        self.face_mask = self.masking.to_render_mask(self.masking.get_mask_face()).cuda()

        self.start_time = datetime.datetime.now().strftime("%m%d%H%M")

        seq_path_elements = self.cfg.test_seq_path.split('/') 
        if self.cfg.fit.dataset == 'MEAD':
            self.dataset_identity = seq_path_elements[-5]
            self.dataset_expression = seq_path_elements[-3]
            self.dataset_level = seq_path_elements[-2]
            self.dataset_seq_num = seq_path_elements[-1]
            self.data_name = '_'.join(
                [self.dataset_identity, self.dataset_expression, self.dataset_level, self.dataset_seq_num]
            )
        elif self.cfg.fit.dataset == 'Nersemble':
            self.dataset_identity = "N" + seq_path_elements[-1]
            self.data_name = self.dataset_identity
        else:  # CELEBV_HQ
            self.dataset_identity = seq_path_elements[-1]
            self.data_name = self.dataset_identity
        # Logging
        log_format = '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | - <level>{message}</level>'
        if self.cfg.loss.uv_map_super > 0.0:
            self.cfg.output_dir  = self.cfg.output_dir + f'_uv{self.cfg.loss.uv_map_super}'
        if self.cfg.loss.normal_super > 0.0:
            self.cfg.output_dir = self.cfg.output_dir + f'_n{self.cfg.loss.normal_super}'
        if self.cfg.loss.lmk > 0.0:
            self.cfg.output_dir = self.cfg.output_dir + f'_l{self.cfg.loss.lmk}'

        self.logdir = os.path.join(self.cfg.output_dir, self.cfg.fit.dataset, self.data_name, 'logs')
        os.makedirs(self.logdir, exist_ok=True)
        logger.add(os.path.join(self.logdir, 'train.log'), format=log_format)
        shutil.copy(self.cfg.cfg_file, os.path.join(self.cfg.output_dir, self.cfg.fit.dataset, self.data_name, 'config.yaml'))

        if self.cfg.fit.write_summary:
            wandb.login()
            wandb_name = self.cfg.output_dir.split('/')[1] + self.data_name
            wandb.init(project="neuface", name=wandb_name)
            wandb.config.update(self.cfg)


    def configure_optimizers(self):
        self.opt = torch.optim.Adam(
            self.deca.E_flame.parameters(),
            lr=self.cfg.fit.lr,
            amsgrad=False)

    @staticmethod
    def transform_points(points, tform, points_scale=None, out_scale=None):
        points_2d = points[:, :, :2]

        # 'input points must use original range'
        if points_scale:
            assert points_scale[0] == points_scale[1]
            points_2d = (points_2d * 0.5 + 0.5) * points_scale[0]

        batch_size, n_points, _ = points.shape
        trans_points_2d = torch.bmm(
            torch.cat([points_2d, torch.ones([batch_size, n_points, 1], device=points.device, dtype=points.dtype)],
                      dim=-1),
            tform
        )
        if out_scale:  # h,w of output image size
            trans_points_2d[:, :, 0] = trans_points_2d[:, :, 0] / out_scale[1] * 2 - 1
            trans_points_2d[:, :, 1] = trans_points_2d[:, :, 1] / out_scale[0] * 2 - 1
        trans_points = torch.cat([trans_points_2d[:, :, :2], points[:, :, 2:]], dim=-1)
        return trans_points

    def load_checkpoint(self):
        model_dict = self.deca.model_dict()
        if os.path.exists(self.cfg.pretrained_modelpath):
            checkpoint = torch.load(self.cfg.pretrained_modelpath)
            key = 'E_flame'
            util.copy_state_dict(model_dict[key], checkpoint[key])
            self.global_step = 0
        else:
            logger.info('model path not found, start training from scratch')
            self.global_step = 0


    def prepare_eft_data(self, seq_path, dataset):
        """Load data

        Args:
            seq_path (str): 
                What specific data sequence we will use.
                `self.cfg.test_seq_path` (e.g., "/data/STREAM/MEAD/M035/images/angry/level_3/001")
            dataset (str):
                The dataset we will use. 
                `self.cfg.fit.dataset` (e.g., 'MEAD')
        """
        # Dataset
        if dataset == 'MEAD':
            self.train_dataset = MEAD(seq_path, useVislmk=self.cfg.loss.useVislmk, is_train=True)#, scale=1.75)
        elif dataset == 'celebv_hq':
            self.train_dataset = CELEBV_HQ(seq_path)
        elif dataset == 'Nersemble': # view_num으로 동영상찍은 카메라를 변경할수 있음
            self.train_dataset = Nersemble(seq_path, useVislmk=self.cfg.loss.useVislmk, is_train=True, view_num=self.cfg.dataset.view_num, static_crop=self.cfg.dataset.static_crop)
        else:
            raise NotImplementedError

        # Dataloader
        self.fit_dataloader = DataLoader(
            self.train_dataset, batch_size=self.cfg.dataset.batch_size, shuffle=False,
            num_workers=self.cfg.dataset.num_workers,
            drop_last=False
        ) 
        #TODO drop_last??

        # Data to give training process
        seq_data = [data for data in tqdm(self.fit_dataloader)]

        # fast debug
        # for data in tqdm(self.fit_dataloader):
        #     seq_data = [data]
        #     break
        # segmentation
        for data in tqdm(seq_data):
            cropped = data['image'].clone() * 255 # batch, 3, 224, 224 cropped, divided by 255
            run_facer_segmentation.run(seq_path, cropped, data['imagename'])
        if self.cfg.dataset.is_seg:
            raise Exception("여기서 멈춤!")

        # read inferenced data
        img_size = cropped.shape[-1]
        for i, data in enumerate(seq_data):
            uv_map_stacked = []
            uv_mask_stacked = []
            normals_stacked = []
            normal_mask_stacked = []
            for file in data['imagename']:
                seg = np.array(
                    Image.open(f'{seq_path}/seg_og/{file}.png'))
                if len(seg.shape) == 3:
                    seg = seg[..., 0]
                # uv mask, normal mask, fg(=foreground) mask 사용할 부분
                uv_mask = ((seg == 2) | (seg == 6) | (seg == 7) |
                           (seg == 10) | (seg == 12) | (seg == 13) |
                           (seg == 1) |  # neck
                           (seg == 4) | (seg == 5)  # ears
                           )

                normal_mask = ((seg == 2) | (seg == 6) | (seg == 7) |
                               (seg == 10) | (seg == 12) | (seg == 13)
                               ) | (seg == 11)  # mouth interior

                try:
                    normals = ((np.array(Image.open(f'{seq_path}/p3dmm/normals/{file}.png').resize((img_size, img_size))) / 255).astype(np.float32) - 0.5) * 2
                    uv_map = (np.array(Image.open(f'{seq_path}/p3dmm/uv_map/{file}.png').resize((img_size, img_size))) / 255).astype(np.float32)
                except Exception as ex:
                    normals = ((np.array(Image.open(f'{seq_path}/p3dmm/normals/{file}.png').resize((img_size, img_size))) / 255).astype(np.float32) - 0.5) * 2
                    uv_map = (np.array(Image.open(f'{seq_path}/p3dmm/uv_map/{file}.png').resize((img_size, img_size))) / 255).astype(np.float32)
                uv_map_stacked.append(uv_map)
                uv_mask_stacked.append(uv_mask)
                normals_stacked.append(normals)
                normal_mask_stacked.append(normal_mask)

            uv_map_stacked = np.stack(uv_map_stacked, axis=0)
            uv_mask_stacked = np.stack(uv_mask_stacked, axis=0)
            normals_stacked = np.stack(normals_stacked, axis=0)
            normal_mask_stacked = np.stack(normal_mask_stacked, axis=0)

            ret_dict = {
                'normals': normals_stacked,
                'uv_map': uv_map_stacked,
                'uv_mask': uv_mask_stacked,
                'normal_mask': normal_mask_stacked,
            }
            ret_dict = {k: torch.from_numpy(v).float() for k, v in ret_dict.items()}
            ret_dict['uv_mask'] = ret_dict['uv_mask'][:, :, :, None].repeat(1, 1, 1, 3)
            ret_dict['normal_mask'] = ret_dict['normal_mask'][:, :, :, None].repeat(1, 1, 1, 3)
            # (1, h, w, 3) -> (1, 3, h, w) 변경
            channels_first = ['uv_mask', 'normal_mask', 'normals', 'uv_map']
            for k in channels_first:
                ret_dict[k] = ret_dict[k].permute(0, 3, 1, 2)
            seq_data[i] = seq_data[i] | ret_dict
        return seq_data


    def model_forward(self, batch:dict) -> (dict, dict):
        """Single step to reconstruct and get losses
        
        Args:
            batch (dict): 
                data to reconstruct. There are...
                'image', 'imagename', 'tform', 'original_image', 'landmark'

        Returns:
            losses (dict) : 
                Dictionary of losses. There are ...
                    'spatial' : spatial loss (multi-view loss),
                    'temporal' : temporal loss (smoothing the shape along with the time),
                    'data' : 2d landmark loss,
                    'all_loss' : sum of all losses
            opdict (dict) :
                Information and result of reconstruction. There are ...
                    'images' : original image,
                    'landmark' : "GT 2d landmark,
                    'verts' : vertices of mesh which are prediction of DECA,
                    'lmk' : 
                    'landmarks3d'
                    'landmarks3d_world'
                    'landmarks2d' : prediction of DECA,
                    'trans_verts'
        """
        # -----------------------------------------------------------------------------
        # Load data
        # -----------------------------------------------------------------------------
        images = batch['image'].to(self.device) # MEAD: [frame, 7 view, 3 channel, 224 height, 224 width]) 
        images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1]) # [frame * view, 3 channel, 224 height, 224 width] / celebv-hp: [frame, 3, 244, 244]
        lmk = batch['landmark'].to(self.device) 
        lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1]) # celebv-hp: [20, 68, 3]

        uv_map = batch['uv_map'].to(self.device)
        uv_mask = batch['uv_mask'].to(self.device)
        normal_map = batch['normals'].to(self.device)
        normal_mask = batch['normal_mask'].to(self.device)

        # 배경제거?
        if uv_map is not None:
            uv_map[(1 - uv_mask[:, :, :, :]).bool()] = 0  # uv_mask의 0인 부분에서 uv_map의 값을 0으로 만듬
        # -----------------------------------------------------------------------------
        # Get DECA reconstruction result
        #   DECA encoder) dictionary named 'codedict'
        #       'images': [frame * view(=7), channel (=3), size (=224), size (=224)]
        #       'shape': [frame * view(=7), shape_coefficient (=100)]
        #       'exp': [frame * view(=7), expression_coefficient (=50)]
        #       'pose': [frame * view(=7), pose_coefficient (=6)] -> first 3: Root orientation, last 3: jaw
        #       'cam': [frame * view(=7), cam_params (=3)]
        #   DECA decoder) dictionary named 'opdict'
        #       'verts': [frame * view(=7), vertices (=5023), location (=3)]
        #       'trans_verts': [frame * view(=7), vertices (=5023), location (=3)]
        #       'landmarks2d': [frame * view(=7), keypoints (=68), location (=2)]
        #       'shape': [frame * view(=7), 100]
        # -----------------------------------------------------------------------------
        # DECA encoder
        codedict = self.deca.encode(images) # ['shape', 'tex', 'exp', 'pose', 'cam', 'light', 'images']
        
        # (Optional) Setting shape coefficients identical to that of first image
        reference_shape_coeffi = codedict['shape'][0]
        for img_idx in range(1, len(codedict['shape'])):
            codedict['shape'][img_idx] = reference_shape_coeffi

        # DECA decoder
        opdict = self.deca.decode(codedict) # ['verts', 'trans_verts', 'landmarks2d', 'landmarks3d', 'landmarks3d_world']

        # Add original data into DECA result - for convienience (maybe?)
        opdict['images'] = images # original images
        opdict['lmk'] = lmk # GT landmarks

        # -----------------------------------------------------------------------------
        # Compute Losses
        # -----------------------------------------------------------------------------
        losses = {}

        pose_clone = codedict['pose'].clone()
        pose_clone[:, :3] = 0 # root orientation을 0으로 바꿈

        # Fit to FLAME model and Get vertices
        verts, landmarks2d, landmarks3d = self.deca.flame(
            shape_params=codedict['shape'],
            expression_params=codedict['exp'],
            pose_params=pose_clone
        )

        losses['spatial'] = lossfunc.efficient_spatial_consistency_loss(
            verts,
            opdict["verts"],
            self.deca.render.faces,
            sampling=self.cfg.loss.spatial_sample,
            dataset=self.cfg.fit.dataset
        ) * self.cfg.loss.spatial_weight

        if self.cfg.fit.dataset == 'MEAD':
            n_views = 7
        elif self.cfg.fit.dataset == 'celebv_hq':
            n_views = 1
        elif self.cfg.fit.dataset == 'Nersemble':
            n_views = 1
        else:
            raise NotImplementedError
        losses['temporal'] = lossfunc.temporal_smooth_loss(codedict['pose'][:, :3], n_views) * self.cfg.loss.temp_weight

        predicted_landmarks = opdict['landmarks2d']
        if self.cfg.loss.lmk > 0:
            if self.cfg.loss.useWlmk:
                losses['data'] = lossfunc.weighted_landmark_loss(predicted_landmarks, lmk) * self.cfg.loss.lmk
            else:
                losses['data'] = lossfunc.landmark_loss(predicted_landmarks, lmk) * self.cfg.loss.lmk
        # result_vertices = opdict['verts'].view(-1, 7, 5023, 3)  # [frame*view, 5023, 3] -> [frame, view, 5023, 3]
        # result_transformed_vertices = opdict['trans_verts'].view(-1, 7, 5023,
        #                                                          3)  # [frame*view, 5023, 3] -> [frame, view, 5023, 3]
        # original_images = opdict['images'].view(-1, 7, 3, 224,
        #                                         224)  # [frame*view, 3, 224, 224] -> [frame, view, 3, 224, 224]
        # shape = self.deca.render.render_shape(result_vertices[2][1].view(1, 5023, 3), result_transformed_vertices[2][1].view(1, 5023, 3), images=original_images[2][1].view(1,3,224,224))
        # viz = 0
        # for b_i in range(predicted_landmarks.shape[0]):
        #     scaled_pred_lmk = predicted_landmarks.clone()
        #     scaled_pred_lmk[:, :, :2] = (scaled_pred_lmk[:, :, :2] + 1) / 2 * images.shape[-1]
        #     scaled_gt_lmk = lmk.clone()
        #     scaled_gt_lmk[:, :, :2] = (scaled_gt_lmk[:, :, :2] + 1) / 2 * images.shape[-1]
        #     empty = plot_points(images[b_i].detach().cpu().numpy().copy().transpose(1, 2, 0),
        #                         pts=scaled_pred_lmk[b_i, :, :2].detach().cpu().numpy(),
        #                         pts2=scaled_gt_lmk[b_i, :, :2].detach().cpu().numpy())

        """uv loss 
        배치로 처리
        전해줄때 batch['uv_map'] 하면 (70, 224, 224, 3)
        """
        depth = self.deca.render.render_real_depth(opdict['trans_verts'])
        scaled_trans_verts = opdict['trans_verts'].clone()
        scaled_trans_verts[:, :, :2] = (scaled_trans_verts[:, :, :2] + 1) / 2 * depth.shape[-1]
        grabbed_depth = depth[:, 0,
                        torch.clamp(scaled_trans_verts[:, :, 1].long(), 0,
                                    depth.shape[-1] - 1),
                        torch.clamp(scaled_trans_verts[:, :, 0].long(), 0,
                                    depth.shape[-1] - 1),
                        ][:, 0, :]
        is_visible_verts_idx = grabbed_depth > (opdict['trans_verts'][:, :, 2] - 1e-2)  # 카메라가 z가 커지는 방향 반대로 보고 있음


        if self.cfg.loss.uv_map_super > 0.0:  # and (p < iters // 2 or self.config.keep_uv) and not self.config.no2d_verts:
            # uv_loss = get_uv_loss(uv_map, proj_vertices)
            if self.uv_loss_fn.gt_2_verts is None:
                self.uv_loss_fn.compute_corresp(uv_map, selected_frames=None)

            # opdict['trans_verts'] 의 정점들의 위치는 정규화 되어 나타내어 있다. 이미지 사이즈 만큼으로 복원하고 정수로 바꾸는 과정에서 long을 사용하게 되면은 정보손실이 발생한다.
            # 이를 막기 위해 기존 p3dmm에서는 정점들의 위치가 어떻게 주어지는지 알아보자.
            # 정점의 x, y 좌표중에서 이미지 사이즈를 넘어가는 것이 있었으며, 그냥 소수점이 있는채로 compute_loss에 넘겨준다.
            uv_loss = self.uv_loss_fn.compute_loss(scaled_trans_verts, selected_frames=None, uv_map=None,
                                                   l2_loss=self.cfg.loss.uv_l2, is_visible_verts_idx=is_visible_verts_idx)
            losses['loss/uv'] = uv_loss * self.cfg.loss.uv_map_super  # 다른 loss에 비해 숫자가 크다. 값이 처음에 0.0x로 나오고 100곱하면 1.x 나옴.

            # valid_verts_visibility = is_visible_verts_idx[:, self.uv_loss_fn.valid_vertex_index]
            # is_valid_uv_corresp = (self.uv_loss_fn.dists[:,:,0] < self.uv_loss_fn.delta) & valid_verts_visibility
            # valid_trans_verts = scaled_trans_verts[:, self.uv_loss_fn.valid_vertex_index, :]
            # for b_i in range(valid_trans_verts.shape[0]):
            #
            #     valid_pred_2d = valid_trans_verts[b_i, is_valid_uv_corresp[b_i], :]
            #     valid_gt_2d = self.uv_loss_fn.gt_2_verts[b_i, is_valid_uv_corresp[b_i], :]
            #     pixels_pred = torch.stack(
            #         [
            #             torch.clamp(valid_pred_2d[:, 0], 0, depth.shape[-1] - 1),
            #             torch.clamp(valid_pred_2d[:, 1], 0, depth.shape[-1] - 1),
            #         ], dim=-1
            #     ).int()
            #     pixels_gt = torch.stack(
            #         [
            #             torch.clamp(valid_gt_2d[:, 0], 0, depth.shape[-1] - 1),
            #             torch.clamp(valid_gt_2d[:, 1], 0, depth.shape[-1] - 1),
            #         ], dim=-1
            #     ).int()
            #
            #
            #     empty = plot_points(images[b_i].detach().cpu().numpy().copy().transpose(1, 2, 0), pts=pixels_pred.detach().cpu().numpy(), pts2=pixels_gt.detach().cpu().numpy())
            #     vis= 0
        """
        normal loss
        렌더링 대상
        ops['mask_images_eyes']
        ops['normal_images']
        
        flame space normal image를 구해야함. 
        self.deca.render.render_normals 에서 trans_verts를 사용해서 normal image를 구할수 있음.
        이것과 flame space normal image이 같은 건지 궁금함. 
        flame space normal image는 world space normal image랑 생긴건 똑같다. 하지만 normal의 방향이 바뀜
        trans_verts를 사용한 normal map과 그냥 verts를 사용한 normal map과 어떻게 다른지를 확인.
        
        
        파라미터 필요
        variables["R"]
        """


        if self.cfg.loss.normal_super > 0.0:
            mask_images_eyes = self.deca.render.render_eye_mask(self.eye_mask, opdict['trans_verts'])
            # normal_loss_map = normal_loss_map * dilated_eye_mask[:, 0, ...] * (1 - ops['mask_images_eyes_region'][:, 0, ...])
            # use dilated eye mask only
            # maybe also applie eyemask in image not rendering
            # 눈 근처는 loss에서 제외 시키기 위함. gaussian_blur는 그냥 근처 부분도 뭉개기 위함?
            dilated_eye_mask = 1 - (gaussian_blur(mask_images_eyes,
                                                  [self.cfg.loss.normal_mask_ksize, self.cfg.loss.normal_mask_ksize],
                                                  sigma=[self.cfg.loss.normal_mask_ksize,
                                                         self.cfg.loss.normal_mask_ksize]) > 0).float()

            # pred_normals = ops['normal_images']  # 1 3 512 512 normals in world space
            # rot_mat = rotation_6d_to_matrix(variables["R"].repeat_interleave(num_views, dim=0))  # 1 3 3, 방향 벡터이기 때문에 translation은 필요 없다.
            # # world space의 rendering 된 normal을 flame space 즉 view matrix적용한 거같은데 .
            # pred_normals_flame_space = torch.einsum('bxy,bxhw->byhw', rot_mat, pred_normals)

            pred_normals_trans, _ = self.deca.render.render_normals(opdict['verts'], opdict['trans_verts'])

            rot_mat = batch_axis2matrix(codedict['pose'][:, :3])  # .transpose(1, 2)
            pred_normals_flame_space = torch.einsum('bxy,bxhw->byhw', rot_mat, pred_normals_trans)
            if normal_map is not None:
                l_map = (normal_map - pred_normals_flame_space)
                valid = ((l_map.abs().sum(dim=1) / 3) < self.cfg.loss.delta_n).unsqueeze(1) # 차이가 너무 나지 않는것들로만
                normal_loss_map = l_map * valid.float() * normal_mask * dilated_eye_mask
                if self.cfg.loss.normal_l2:
                    losses['loss/normal'] = normal_loss_map.square().mean() * self.cfg.loss.normal_super
                else:
                    losses['loss/normal'] = normal_loss_map.abs().mean() * self.cfg.loss.normal_super
            else:
                losses['loss/normal'] = 0.0


        # -----------------------------------------------------------------------------
        # Sum All Losses
        # -----------------------------------------------------------------------------
        all_loss = 0.
        losses_key = losses.keys()
        for key in losses_key:
            all_loss = all_loss + losses[key]
        losses['all_loss'] = all_loss

        return losses, opdict, codedict


    def fit(self, seq_path):
        """Main training Process
        Args:
            seq_path (str): 
                What specific data sequence we will use. (`self.cfg.test_seq_path`)
        """
        # Load data & batch length
        logger.info("Loading the video data & processing keypoints ...")
        seq_data = self.prepare_eft_data(seq_path, self.cfg.fit.dataset) # 이거 끝났는데 images 처리는 100장밖에 안됨. 이유는 Drop_last
        num_batches_per_step = len(self.fit_dataloader) # self.fit_dataloader is set in self.prepare_eft_data(), 에폭 개수.

        # Optimization Process
        logger.info("Start NeuFace optimization ...")
        start_time = time.time()
        while self.global_step < self.cfg.fit.max_steps:
            for batch_idx, batch in enumerate(seq_data):
                # ---------------------------------------------------------------------
                # In `batch` dict ...
                #   'image' (shape: [frame, 7 view, 3 channel, 224 height, 224 width]) :
                #   'imagename' (shape: []) :
                #   'tform' (shape: []) :
                #   'original_image' (shape: []) :
                #   'landmark' (shape: [frame, 7 view, 68 kpts, 2 location]) :
                # ---------------------------------------------------------------------
                losses, opdict, codedict = self.model_forward(batch)
                # ---------------------------------------------------------------------
                # What are in the return value - `opdict` dictionary
                #   'verts': [frame * view(=7), vertices (=5023), location (=3)]
                #   'trans_verts': [frame * view(=7), vertices (=5023), location (=3)]
                #   'landmarks2d': [frame * view(=7), keypoints (=68), location (=2)]
                #   'shape': [frame * view(=7), 100]
                # ---------------------------------------------------------------------

                # log this single step
                step_info = f"Exp: {self.cfg.exp_name} | Step [{self.global_step + 1}/{self.cfg.fit.max_steps}] | Batch [{batch_idx + 1}/{num_batches_per_step}]"
                logger.info(step_info)

                loss_info = "[Losses] "
                for k, v in losses.items():
                    loss_info = loss_info + f'{k}: {v:.4f}, '
                    if self.cfg.fit.write_summary:
                        wandb.log({'fitting_loss/' + k: v}, self.global_step)
                logger.info(loss_info)
                
                # backprop
                all_loss = losses['all_loss']
                self.opt.zero_grad()
                torch.cuda.empty_cache()
                all_loss.backward()
                self.opt.step()

                self.uv_loss_fn.is_next()

                # Visualization
                if self.global_step % self.cfg.fit.vis_steps == 0:
                    if self.cfg.fit.dataset == 'MEAD':
                        result_vertices = opdict['verts'].view(-1, 7, 5023, 3) # [frame*view, 5023, 3] -> [frame, view, 5023, 3]
                        result_transformed_vertices = opdict['trans_verts'].view(-1, 7, 5023, 3) # [frame*view, 5023, 3] -> [frame, view, 5023, 3]
                        original_images = opdict['images'].view(-1, 7, 3, 224, 224) # [frame*view, 3, 224, 224] -> [frame, view, 3, 224, 224]
                    else:
                        result_vertices = opdict['verts'].view(-1, 1, 5023, 3)
                        result_transformed_vertices = opdict['trans_verts'].view(-1, 1, 5023, 3)
                        original_images = opdict['images'].view(-1, 1, 3, 224, 224)

                    if self.cfg.fit.save_train_img:
                        if self.cfg.fit.dataset == 'MEAD':
                            # save only front images
                            # train_img_path = f'{self.cfg.output_dir}/{self.start_time}-{self.cfg.fit.dataset}-{self.dataset_identity}-{self.dataset_expression}-{self.dataset_level}-{self.dataset_seq_num}/step{self.global_step+1}'
                            train_img_path = f'{self.logdir}/log_imgs/step{self.global_step+1:06d}'
                            os.makedirs(train_img_path, exist_ok=True)
                            for frame in range(len(result_vertices)):
                                shape_image = self.deca.render.render_shape(
                                    vertices=(result_vertices[frame][1].view(1, 5023, 3)),
                                    transformed_vertices=(result_transformed_vertices[frame][1]).view(1, 5023, 3),
                                )
                                save_image(shape_image, f'{train_img_path}/{frame + 1 + batch_idx*20}.jpg') # 20 = batch_size / view = frame
                        else:
                            # save all images in path
                            train_img_path = f'{self.logdir}/log_imgs'
                            os.makedirs(train_img_path, exist_ok=True)
                            total_seq_shapes = []
                            for frame in range(len(result_vertices)):
                                shape_images = self.deca.render.render_shape(
                                    vertices=result_vertices[frame], # [1, 5023, 3]
                                    transformed_vertices=result_transformed_vertices[frame], # [1, 5023 ,3]
                                    images=original_images[frame] # [1, 3, 224, 224]
                                ) # [7, 3, 224, 224]
                                total_seq_shapes += shape_images
                            save_image(total_seq_shapes, f'{train_img_path}/step{self.global_step+1}_batch{batch_idx + 1}.jpg', nrow=5)

            self.global_step += 1

        total_time = time.time() - start_time
        logger.info(f"NeuFace optimization done! Total elapsed time: {total_time:.3f} sec.")
        logger.info("="*50)

        # Saving outputs
        if self.cfg.fit.save_fit_flame or self.cfg.fit.save_fit_img or self.cfg.fit.save_fit_video:
            # Dataset & Saving Paths
            logger.info(f"Saving outputs for {cfg.fit.dataset}-{self.data_name}...")
            imgsave_path = os.path.join(
                self.cfg.output_dir,
                self.cfg.fit.dataset,
                self.data_name,
                'images'
            )
            videosave_path = os.path.join(
                self.cfg.output_dir,
                self.cfg.fit.dataset,
                self.data_name,
                'videos'
            )
            cattedvideo_path = os.path.join(
                self.cfg.output_dir,
                self.cfg.fit.dataset,
                self.data_name,
                'cattedvideos'
            )
            flamesave_path = os.path.join(
                self.cfg.output_dir,
                self.cfg.fit.dataset,
                self.data_name,
                'flame_params'
            )

            if self.cfg.fit.save_fit_img:
                os.makedirs(imgsave_path, exist_ok=True)
                os.makedirs(cattedvideo_path, exist_ok=True)
            if self.cfg.fit.save_fit_flame:
                os.makedirs(flamesave_path, exist_ok=True)
            if self.cfg.fit.save_fit_video:
                os.makedirs(videosave_path, exist_ok=True)

            with torch.no_grad():
                # dataset load
                self.train_dataset.is_train = False
                vis_dataloader = DataLoader(
                    self.train_dataset, 
                    batch_size=1, 
                    shuffle=False,
                    num_workers=self.cfg.dataset.num_workers,
                    drop_last=False
                )
                seq_data = [data for data in vis_dataloader]

                # Find
                tfrom_ref_idx = 0
                if self.cfg.fit.dataset == 'MEAD':
                    for idx, name in enumerate(self.train_dataset.imagepath_list):
                        if 'front' in name:
                            tfrom_ref_idx = idx
                            break
                
                tform_ref_data = copy.deepcopy(seq_data[tfrom_ref_idx])
                images = tform_ref_data['image'].to(self.device)
                images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])

                codedict = self.deca.encode(images)
                fixed_shape = codedict['shape']  # (Optional) Fix beta param 첫번째 정면 얼굴로 shape 구하고 고정

                cam_list, pose_list, expr_list, shape_list, landmarks2d_list = [], [], [], [], []
                for frame_idx, batch in enumerate(tqdm(seq_data)):
                    images = batch['image'].to(self.device)
                    images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                    lmk = batch['landmark'].to(self.device) # [1, 1, 68, 3]
                    lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1]) # [1, 68, 3]

                    # DECA encoder & decoder
                    codedict = self.deca.encode(images)
                    codedict['shape'] = fixed_shape
                    opdict = self.deca.decode(codedict)

                    # Saving Annotations
                    if self.cfg.fit.save_fit_flame:
                        _, h, w, _ =  batch['original_image'][0].shape
                        cam_list.append(codedict['cam'][0].cpu())  # torch.Tensor [3,]
                        pose_list.append(codedict['pose'][0].cpu())  # torch.Tensor [6,]
                        expr_list.append(codedict['exp'][0].cpu())  # torch.Tensor [50,]
                        shape_list.append(codedict['shape'][0].cpu())  # torch.Tensor [100,]
                        landmarks2d_list.append(lmk[0, :, :2].cpu())  # torch.Tensor [68, 2] face detector result

                    if self.cfg.fit.save_fit_img: # 이미지 저장.
                        self.save_image_one_for_batch(
                            data=batch, 
                            opdict=opdict, 
                            imgsave_path=imgsave_path, 
                            background_img=True,
                        )
                        # TODO 학습 끝나고 시각화 동영상 생성 코드 추가
                        self.render_and_save(
                            data=batch,
                            codedict=codedict,
                            opdict=opdict,
                            imgsave_path=cattedvideo_path,
                            gt_path=seq_path
                        )

                cam_all = torch.stack(cam_list, dim=0).numpy()  # [S, 3]
                pose_all = torch.stack(pose_list, dim=0).numpy()  # [S, 6]
                expr_all = torch.stack(expr_list, dim=0).numpy()  # [S, 50]
                shapes_all = torch.stack(shape_list, dim=0).numpy()  # [S, 100]
                landmarks2d_all = torch.stack(landmarks2d_list, dim=0).numpy()  # [S, 68, 2]

                np.savez_compressed(
                    f'{flamesave_path}/params.npz',
                    cam=cam_all,
                    pose=pose_all,
                    expr=expr_all,
                    shapes=shapes_all,
                    landmarks2d=landmarks2d_all
                )

            if self.cfg.fit.save_fit_video:
                if self.cfg.fit.dataset == 'MEAD':
                    img_path_pattern = f"{imgsave_path}/*front.png"
                else:
                    img_path_pattern = f"{imgsave_path}/*.png"
                util.create_video(img_path_pattern, f"{videosave_path}/video.mp4")
                img_path_pattern = f"{cattedvideo_path}/*.png"
                util.create_video(img_path_pattern, f"{videosave_path}/catted_video.mp4", fps=25) #TODO p3dmm은 25임

            logger.info("="*50)
            logger.info(f"Results saved in {self.cfg.output_dir}/{self.cfg.fit.dataset}/{self.data_name}")

    def save_vectices_numpy(self, flame_path):
        flamesave_path = os.path.join(flame_path, 'flame_params')

        data = np.load(f'{flamesave_path}/params.npz')

        cam_all = torch.from_numpy(data['cam']).to(self.device)
        pose_all = torch.from_numpy(data['pose']).to(self.device)
        expr_all = torch.from_numpy(data['expr']).to(self.device)
        shapes_all = torch.from_numpy(data['shapes']).to(self.device)
        landmarks2d_all = torch.from_numpy(data['landmarks2d']).to(self.device)

        codedict = {
            'cam': cam_all,
            'pose': pose_all,
            'exp': expr_all,
            'shape': shapes_all,
            'landmarks2d': landmarks2d_all
        }

        opdict = self.deca.decode(codedict)
        v = opdict['trans_verts'][0].view(1, 5023, 3).cpu().numpy()

        np.save(os.path.join(flamesave_path, "vetices.npy"), v)


    def recover_and_make_video(self, seq_path, flame_path):
        self.train_dataset = MEAD(seq_path, useVislmk=self.cfg.loss.useVislmk, is_train=True)
        # flamesave_path = os.path.join(
        #     self.cfg.output_dir,
        #     self.cfg.fit.dataset,
        #     self.data_name,
        #     'flame_params'
        # )
        #
        # imgsave_path = os.path.join(
        #     self.cfg.output_dir,
        #     self.cfg.fit.dataset,
        #     self.data_name,
        #     'normal_shape'
        # )
        flamesave_path = os.path.join(flame_path, 'flame_params')

        normalsave_path = os.path.join(flame_path, 'normal')
        os.makedirs(normalsave_path, exist_ok=True)

        datalossmap_path = os.path.join(flame_path, 'datalossmap')
        os.makedirs(datalossmap_path, exist_ok=True)
        data = np.load(f'{flamesave_path}/params.npz')
        cam_all = torch.from_numpy(data['cam']).to(self.device)
        pose_all = torch.from_numpy(data['pose']).to(self.device)
        expr_all = torch.from_numpy(data['expr']).to(self.device)
        shapes_all = torch.from_numpy(data['shapes']).to(self.device)
        landmarks2d_all = torch.from_numpy(data['landmarks2d']).to(self.device)

        codedict = {
            'cam': cam_all,
            'pose': pose_all,
            'exp': expr_all,
            'shape': shapes_all,
            'landmarks2d': landmarks2d_all
        }

        opdict = self.deca.decode(codedict)

        with torch.no_grad():
            # dataset load
            self.train_dataset.is_train = False
            vis_dataloader = DataLoader(
                self.train_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.cfg.dataset.num_workers,
                drop_last=False
            )
            seq_data = [data for data in vis_dataloader]

            for frame_idx, batch in enumerate(tqdm(seq_data)):
                # if batch['imagename'][0][-5:] != 'front':
                #     continue
                # render normal
                # pred_normals_trans, alpha_images = self.deca.render.render_normals(opdict['verts'][frame_idx].view(1,5023,3), opdict['trans_verts'][frame_idx].view(1,5023,3))
                #
                # rot_mat = batch_axis2matrix(codedict['pose'][frame_idx, :3].view(1, 3))  # .transpose(1, 2)
                # pred_normals_flame_space = torch.einsum('bxy,bxhw->byhw', rot_mat, pred_normals_trans)
                images = batch['image'].to(self.device)
                images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])

                # alpha_images = alpha_images * 0.7
                # shape = (pred_normals_flame_space + 1) / 2
                # shape_images = shape * alpha_images + images * (1 - alpha_images) #(1,3,224,224)
                # save_imgname = os.path.join(normalsave_path, f"{batch['imagename'][0]}.png")
                # save_image(shape_images, save_imgname)



                lmk = batch['landmark'].to(self.device)
                lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
                predicted_landmarks = opdict['landmarks2d']
                scaled_pred_lmk = predicted_landmarks.clone()
                scaled_pred_lmk[:, :, :2] = (scaled_pred_lmk[:, :, :2] + 1) / 2 * images.shape[-1]
                scaled_gt_lmk = lmk.clone()
                scaled_gt_lmk[:, :, :2] = (scaled_gt_lmk[:, :, :2] + 1) / 2 * images.shape[-1]
                empty = plot_points(images[0].detach().cpu().numpy().copy().transpose(1, 2, 0),
                                    pts=scaled_pred_lmk[frame_idx, :, :2].detach().cpu().numpy(),
                                    pts2=scaled_gt_lmk[0, :, :2].detach().cpu().numpy())
                img = Image.fromarray(empty)
                save_imgname = os.path.join(datalossmap_path, f"{batch['imagename'][0]}.png")
                img.save(save_imgname)


    def save_image_one_for_batch_cropped(self, data, opdict, imgsave_path, fixed_tform=None, background_img=False):
        """
                data: batch
                opdict: output of DECA decoder
                """
        points_scale = [self.deca.image_size, self.deca.image_size]
        data['image'] = data['image'].squeeze(0).permute((0, 3, 1, 2))
        _, _, h, w = data['image'].shape
        lmk = data['landmark'].to(self.device)
        lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])

        if fixed_tform is not None:
            trans_verts = self.transform_points(opdict['fix_trans_verts'], fixed_tform, points_scale, [h, w])
            gt_lmk_2d = self.transform_points(lmk, fixed_tform, points_scale, [h, w])
        else:
            tform = torch.inverse(data['tform']).transpose(1, 2).to(self.device)
            gt_lmk_2d = self.transform_points(lmk, tform, points_scale, [h, w])
            trans_verts = self.transform_points(opdict['trans_verts'], tform, points_scale, [h, w])

        background = data['image'].to(self.device) if background_img else None
        shape_images = self.deca.render.render_shape(opdict['verts'], trans_verts, h=h, w=w, images=background)

        # save single image
        save_imgname = os.path.join(imgsave_path, f"{data['imagename'][0]}.png")
        save_image(shape_images, save_imgname)

    def save_image_one_for_batch(self, data, opdict, imgsave_path, fixed_tform=None, background_img=False):
        """
        data: batch
        opdict: output of DECA decoder
        """
        points_scale = [self.deca.image_size, self.deca.image_size]
        data['original_image'] = data['original_image'].squeeze(0).permute((0, 3, 1, 2))
        _, _, h, w = data['original_image'].shape
        lmk = data['landmark'].to(self.device)
        lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])

        
        if fixed_tform is not None:
            trans_verts = self.transform_points(opdict['fix_trans_verts'], fixed_tform, points_scale, [h, w])
            gt_lmk_2d = self.transform_points(lmk, fixed_tform, points_scale, [h, w])
        else:
            tform = torch.inverse(data['tform']).transpose(1, 2).to(self.device)
            gt_lmk_2d = self.transform_points(lmk, tform, points_scale, [h, w])
            trans_verts = self.transform_points(opdict['trans_verts'], tform, points_scale, [h, w])

        background = data['original_image'].to(self.device) if background_img else None
        shape_images = self.deca.render.render_shape(opdict['verts'], trans_verts, h=h, w=w, images=background)

        # save single image
        save_imgname = os.path.join(imgsave_path, f"{data['imagename'][0]}.png")
        save_image(shape_images, save_imgname)

    def render_and_save(self, data, codedict, opdict, imgsave_path, gt_path):
        image = data['image']
        img_size = image.shape[-1]
        data['image'] = data['image'].squeeze(0)
        background = data['image'].to(self.device)[0]
        pred_normals_trans, alpha_images = self.deca.render.render_normals(opdict['verts'].view(1, 5023, 3),
                                                                           opdict['trans_verts'].view(1, 5023, 3))
        rot_mat = batch_axis2matrix(codedict['pose'][0, :3].view(1, 3))  # .transpose(1, 2)
        pred_normals_flame_space = torch.einsum('bxy,bxhw->byhw', rot_mat, pred_normals_trans)
        mask_images_face = self.deca.render.render_face_mask(self.face_mask, opdict['trans_verts'])

        shape_mask = (mask_images_face > 0.).int()[0]
        shape = (pred_normals_flame_space[0] + 1) / 2 * shape_mask
        blend = background * (1 - shape_mask) + background * shape_mask * 0.3 + shape * 0.7 * shape_mask

        predicted_normal = ((pred_normals_flame_space[0].permute(1, 2, 0)[...,
                             :3] + 1) / 2 * 255).detach().cpu().numpy().astype(np.uint8)

        gt_normal = np.array(Image.open(f"{gt_path}/p3dmm/normals/{data['imagename'][0]}.png").resize((img_size, img_size)))

        to_be_catted = [(background.cpu().permute(1, 2, 0).numpy()*255).astype(np.uint8),
                        (blend.permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8),
                        predicted_normal,
                        gt_normal]

        # TODO  visualize loss
        loss_catted = []

        # data load
        uv_map_stacked = []
        uv_mask_stacked = []
        normals_stacked = []
        normal_mask_stacked = []
        file = data['imagename'][0]
        seg = np.array(
            Image.open(f'{gt_path}/seg_og/{file}.png'))
        if len(seg.shape) == 3:
            seg = seg[..., 0]
        # uv mask, normal mask, fg(=foreground) mask 사용할 부분
        uv_mask = ((seg == 2) | (seg == 6) | (seg == 7) |
                   (seg == 10) | (seg == 12) | (seg == 13) |
                   (seg == 1) |  # neck
                   (seg == 4) | (seg == 5)  # ears
                   )

        normal_mask = ((seg == 2) | (seg == 6) | (seg == 7) |
                       (seg == 10) | (seg == 12) | (seg == 13)
                       ) | (seg == 11)  # mouth interior

        try:
            normals = ((np.array(
                Image.open(f'{gt_path}/p3dmm/normals/{file}.png').resize((img_size, img_size))) / 255).astype(
                np.float32) - 0.5) * 2
            uv_map = (np.array(
                Image.open(f'{gt_path}/p3dmm/uv_map/{file}.png').resize((img_size, img_size))) / 255).astype(
                np.float32)
        except Exception as ex:
            normals = ((np.array(
                Image.open(f'{gt_path}/p3dmm/normals/{file}.png').resize((img_size, img_size))) / 255).astype(
                np.float32) - 0.5) * 2
            uv_map = (np.array(
                Image.open(f'{gt_path}/p3dmm/uv_map/{file}.png').resize((img_size, img_size))) / 255).astype(
                np.float32)
        uv_map_stacked.append(uv_map)
        uv_mask_stacked.append(uv_mask)
        normals_stacked.append(normals)
        normal_mask_stacked.append(normal_mask)

        uv_map_stacked = np.stack(uv_map_stacked, axis=0)
        uv_mask_stacked = np.stack(uv_mask_stacked, axis=0)
        normals_stacked = np.stack(normals_stacked, axis=0)
        normal_mask_stacked = np.stack(normal_mask_stacked, axis=0)

        ret_dict = {
            'normals': normals_stacked,
            'uv_map': uv_map_stacked,
            'uv_mask': uv_mask_stacked,
            'normal_mask': normal_mask_stacked,
        }
        ret_dict = {k: torch.from_numpy(v).float() for k, v in ret_dict.items()}
        ret_dict['uv_mask'] = ret_dict['uv_mask'][:, :, :, None].repeat(1, 1, 1, 3)
        ret_dict['normal_mask'] = ret_dict['normal_mask'][:, :, :, None].repeat(1, 1, 1, 3)
        # (1, h, w, 3) -> (1, 3, h, w) 변경
        channels_first = ['uv_mask', 'normal_mask', 'normals', 'uv_map']
        for k in channels_first:
            ret_dict[k] = ret_dict[k].permute(0, 3, 1, 2)

        uv_map = ret_dict['uv_map'].to(self.device)
        uv_mask = ret_dict['uv_mask'].to(self.device)
        normal_map = ret_dict['normals'].to(self.device)
        normal_mask = ret_dict['normal_mask'].to(self.device)

        # 배경제거?
        if uv_map is not None:
            uv_map[(1 - uv_mask[:, :, :, :]).bool()] = 0  # uv_mask의 0인 부분에서 uv_map의 값을 0으로 만듬

        """uv loss 
                배치로 처리
                전해줄때 batch['uv_map'] 하면 (70, 224, 224, 3)
                """
        depth = self.deca.render.render_real_depth(opdict['trans_verts'])
        scaled_trans_verts = opdict['trans_verts'].clone()
        scaled_trans_verts[:, :, :2] = (scaled_trans_verts[:, :, :2] + 1) / 2 * depth.shape[-1]
        grabbed_depth = depth[:, 0,
                        torch.clamp(scaled_trans_verts[:, :, 1].long(), 0,
                                    depth.shape[-1] - 1),
                        torch.clamp(scaled_trans_verts[:, :, 0].long(), 0,
                                    depth.shape[-1] - 1),
                        ][:, 0, :]
        is_visible_verts_idx = grabbed_depth > (opdict['trans_verts'][:, :, 2] - 1e-2)  # 카메라가 z가 커지는 방향 반대로 보고 있음

        # uv_loss = get_uv_loss(uv_map, proj_vertices)
        self.uv_loss_fn.compute_corresp(uv_map, selected_frames=None)

        # opdict['trans_verts'] 의 정점들의 위치는 정규화 되어 나타내어 있다. 이미지 사이즈 만큼으로 복원하고 정수로 바꾸는 과정에서 long을 사용하게 되면은 정보손실이 발생한다.
        # 이를 막기 위해 기존 p3dmm에서는 정점들의 위치가 어떻게 주어지는지 알아보자.
        # 정점의 x, y 좌표중에서 이미지 사이즈를 넘어가는 것이 있었으며, 그냥 소수점이 있는채로 compute_loss에 넘겨준다.
        uv_loss = self.uv_loss_fn.compute_loss(scaled_trans_verts, selected_frames=None, uv_map=None,
                                               l2_loss=self.cfg.loss.uv_l2,
                                               is_visible_verts_idx=is_visible_verts_idx)

        valid_verts_visibility = is_visible_verts_idx[:, self.uv_loss_fn.valid_vertex_index]
        is_valid_uv_corresp = (self.uv_loss_fn.dists[:,:,0] < self.uv_loss_fn.delta) & valid_verts_visibility
        valid_trans_verts = scaled_trans_verts[:, self.uv_loss_fn.valid_vertex_index, :]
        for b_i in range(valid_trans_verts.shape[0]):

            valid_pred_2d = valid_trans_verts[b_i, is_valid_uv_corresp[b_i], :]
            valid_gt_2d = self.uv_loss_fn.gt_2_verts[b_i, is_valid_uv_corresp[b_i], :]
            pixels_pred = torch.stack(
                [
                    torch.clamp(valid_pred_2d[:, 0], 0, depth.shape[-1] - 1),
                    torch.clamp(valid_pred_2d[:, 1], 0, depth.shape[-1] - 1),
                ], dim=-1
            ).int()
            pixels_gt = torch.stack(
                [
                    torch.clamp(valid_gt_2d[:, 0], 0, depth.shape[-1] - 1),
                    torch.clamp(valid_gt_2d[:, 1], 0, depth.shape[-1] - 1),
                ], dim=-1
            ).int()

        # v_dist_2d = (valid_gt_2d-valid_pred_2d[:, :2])
        # dist_uv = v_dist_2d.abs().sum(dim=-1) < self.uv_loss_fn.dist_uv # TODO 얘로 찍어보는게 맞는건지. 봤는데 dist_uv가 20일때는 전부 true이더라
        # dist_pixels_pred = pixels_pred[dist_uv] # 일단
        uv_loss_map = plot_points(background.detach().cpu().numpy().copy().transpose(1, 2, 0), pts=pixels_pred.detach().cpu().numpy(), pts2=pixels_gt.detach().cpu().numpy())


        mask_images_eyes = self.deca.render.render_eye_mask(self.eye_mask, opdict['trans_verts'])
        # normal_loss_map = normal_loss_map * dilated_eye_mask[:, 0, ...] * (1 - ops['mask_images_eyes_region'][:, 0, ...])
        # use dilated eye mask only
        # maybe also applie eyemask in image not rendering
        # 눈 근처는 loss에서 제외 시키기 위함. gaussian_blur는 그냥 근처 부분도 뭉개기 위함?
        dilated_eye_mask = 1 - (gaussian_blur(mask_images_eyes,
                                              [self.cfg.loss.normal_mask_ksize, self.cfg.loss.normal_mask_ksize],
                                              sigma=[self.cfg.loss.normal_mask_ksize,
                                                     self.cfg.loss.normal_mask_ksize]) > 0).float()
        l_map = (normal_map - pred_normals_flame_space) # [1 3 224 224]
        valid = ((l_map.abs().sum(dim=1) / 3) < self.cfg.loss.delta_n).unsqueeze(1)  # 차이가 너무 나지 않는것들로만
        normal_loss_map = l_map * valid.float() * normal_mask * dilated_eye_mask
        normal_loss_map = (
                (normal_loss_map.abs().permute(0, 2, 3, 1)) / 2 * 255).detach().cpu().numpy().astype(
            np.uint8)[0]
        # data loss

        lmk = data['landmark'].to(self.device)
        lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
        predicted_landmarks = opdict['landmarks2d']
        scaled_pred_lmk = predicted_landmarks.clone()
        scaled_pred_lmk[:, :, :2] = (scaled_pred_lmk[:, :, :2] + 1) / 2 * img_size
        scaled_gt_lmk = lmk.clone()
        scaled_gt_lmk[:, :, :2] = (scaled_gt_lmk[:, :, :2] + 1) / 2 * img_size
        lmk_loss_map = plot_points(background.detach().cpu().numpy().copy().transpose(1, 2, 0),
                            pts=scaled_pred_lmk[0, :, :2].detach().cpu().numpy(),
                            pts2=scaled_gt_lmk[0, :, :2].detach().cpu().numpy())

        if self.cfg.fit.save_extra_viz:
            loss_catted += [normal_loss_map, uv_loss_map, lmk_loss_map]
            to_be_catted += loss_catted
        catted_uv_I = Image.fromarray(np.concatenate(to_be_catted, axis=1))

        # save single image
        save_imgname = os.path.join(imgsave_path, f"{data['imagename'][0]}.png")
        catted_uv_I.save(save_imgname)