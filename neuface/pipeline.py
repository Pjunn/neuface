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
import wandb

import datetime, sys, copy


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
        else:  # CELEBV_HQ
            self.dataset_identity = seq_path_elements[-1]
            self.data_name = self.dataset_identity

        # Logging
        log_format = '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | - <level>{message}</level>'
        self.logdir = os.path.join(self.cfg.output_dir, self.cfg.fit.dataset, self.data_name, 'logs')
        os.makedirs(self.logdir, exist_ok=True)
        logger.add(os.path.join(self.logdir, 'train.log'), format=log_format)
        shutil.copy(self.cfg.cfg_file, os.path.join(self.cfg.output_dir, self.cfg.fit.dataset, self.data_name, 'config.yaml'))

        if self.cfg.fit.write_summary:
            wandb.login()
            wandb.init(project="neuface")
            wandb.config.update(self.cfg)


    def configure_optimizers(self):
        self.opt = torch.optim.Adam(
            self.deca.E_flame.parameters(),
            lr=self.cfg.fit.lr,
            amsgrad=False)

    @staticmethod
    def transform_points(self, points, tform, points_scale=None, out_scale=None):
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
            self.train_dataset = MEAD(seq_path, useVislmk=self.cfg.loss.useVislmk, is_train=True)
        elif dataset == 'celebv_hq':
            self.train_dataset = CELEBV_HQ(seq_path)
        else:
            raise NotImplementedError

        # Dataloader
        self.fit_dataloader = DataLoader(
            self.train_dataset, batch_size=self.cfg.dataset.batch_size, shuffle=False,
            num_workers=self.cfg.dataset.num_workers,
            drop_last=True
        ) 

        # Data to give training process
        seq_data = [data for data in tqdm(self.fit_dataloader)]
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
        pose_clone[:, :3] = 0

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
        else:
            raise NotImplementedError
        losses['temporal'] = lossfunc.temporal_smooth_loss(codedict['pose'][:, :3], n_views) * self.cfg.loss.temp_weight

        predicted_landmarks = opdict['landmarks2d']
        if self.cfg.loss.lmk > 0:
            if self.cfg.loss.useWlmk:
                losses['data'] = lossfunc.weighted_landmark_loss(predicted_landmarks, lmk) * self.cfg.loss.lmk
            else:
                losses['data'] = lossfunc.landmark_loss(predicted_landmarks, lmk) * self.cfg.loss.lmk

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
        seq_data = self.prepare_eft_data(seq_path, self.cfg.fit.dataset)
        num_batches_per_step = len(self.fit_dataloader) # self.fit_dataloader is set in self.prepare_eft_data()

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
                all_loss.backward()
                self.opt.step()
                
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
            flamesave_path = os.path.join(
                self.cfg.output_dir,
                self.cfg.fit.dataset,
                self.data_name,
                'flame_params'
            )

            if self.cfg.fit.save_fit_img:
                os.makedirs(imgsave_path, exist_ok=True)
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
                fixed_shape = codedict['shape']  # (Optional) Fix beta param

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

                    if self.cfg.fit.save_fit_img:
                        self.save_image_one_for_batch(
                            data=batch, 
                            opdict=opdict, 
                            imgsave_path=imgsave_path, 
                            background_img=True,
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

            logger.info("="*50)
            logger.info(f"Results saved in {self.cfg.output_dir}/{self.cfg.fit.dataset}/{self.data_name}")


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
