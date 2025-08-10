'''
Default config for DECA
'''
from yacs.config import CfgNode as CN
import argparse
import yaml
import time
import os

cfg = CN()

abs_deca_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
cfg.deca_dir = abs_deca_dir
cfg.device = 'cuda'

cfg.pretrained_modelpath = os.path.join(cfg.deca_dir, 'data', 'deca_model.tar')
cfg.output_dir = os.path.join(cfg.deca_dir, 'checkpoint')
cfg.rasterizer_type = 'pytorch3d'

# ---------------------------------------------------------------------------- #
# Options for Face model
# ---------------------------------------------------------------------------- #
cfg.model = CN()
cfg.model.topology_path = os.path.join(cfg.deca_dir, 'data', 'head_template.obj')

# texture data original from http://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_texture_data.zip
cfg.model.dense_template_path = os.path.join(cfg.deca_dir, 'data', 'texture_data_256.npy')
cfg.model.fixed_displacement_path = os.path.join(cfg.deca_dir, 'data', 'fixed_displacement_256.npy')
cfg.model.flame_model_path = os.path.join(cfg.deca_dir, 'data', 'generic_model.pkl') 
cfg.model.flame_lmk_embedding_path = os.path.join(cfg.deca_dir, 'data', 'landmark_embedding.npy') 
cfg.model.face_mask_path = os.path.join(cfg.deca_dir, 'data', 'uv_face_mask.png') 
cfg.model.face_eye_mask_path = os.path.join(cfg.deca_dir, 'data', 'uv_face_eye_mask.png') 
cfg.model.mean_tex_path = os.path.join(cfg.deca_dir, 'data', 'mean_texture.jpg') 
cfg.model.tex_path = os.path.join(cfg.deca_dir, 'data', 'FLAME_albedo_from_BFM.npz')
cfg.model.tex_type = 'BFM' # BFM, FLAME, albedoMM
cfg.model.uv_size = 256
cfg.model.param_list = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
cfg.model.n_shape = 100
cfg.model.n_tex = 50
cfg.model.n_exp = 50
cfg.model.n_cam = 3
cfg.model.n_pose = 6
cfg.model.n_light = 27
cfg.model.use_tex = True
cfg.model.jaw_type = 'aa' # default use axis angle, another option: euler. Note that: aa is not stable in the beginning

# uv loss
cfg.model.VALID_VERTS_NARROW = os.path.join(cfg.deca_dir, 'data', 'uv_valid_verty_noEyes.npy')
cfg.model.VALID_VERTS = os.path.join(cfg.deca_dir, 'data', 'uv_valid_verty_noEyes_noEyeRegion_debug_wEars.npy')
cfg.model.FLAME_UV_COORDS = os.path.join(cfg.deca_dir, 'data', 'flame_uv_coords.npy')

# normal
cfg.model.flame_mask = os.path.join(cfg.deca_dir, 'data', 'FLAME_masks.pkl')
# face recognition model
cfg.model.fr_model_path = os.path.join(cfg.deca_dir, 'data', 'resnet50_ft_weight.pkl')

# details
cfg.model.n_detail = 128
cfg.model.max_z = 0.01

# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.training_data = ['MEAD'] # not used
cfg.dataset.mead_identity = ['W041'] # not used
cfg.dataset.eval_data = ['MEAD'] # not used
cfg.dataset.test_data = ['']
cfg.dataset.batch_size = 140
cfg.dataset.K = 4
cfg.dataset.isSingle = False
cfg.dataset.num_workers = 0
cfg.dataset.image_size = 224
cfg.dataset.scale_min = 1.4
cfg.dataset.scale_max = 1.8
cfg.dataset.trans_scale = 0.
cfg.dataset.view_num = 0
cfg.dataset.static_crop = True
cfg.dataset.is_seg = False
# ---------------------------------------------------------------------------- #
# Options for training
# ---------------------------------------------------------------------------- #
cfg.fit = CN()
cfg.fit.max_steps = 100
cfg.fit.lr = 2e-5 # neuface default = 2e-5, code default: 1e-5
cfg.fit.log_dir = 'logs'
cfg.fit.log_steps = 10
cfg.fit.vis_dir = 'fit_images'
cfg.fit.vis_steps = 20
cfg.fit.write_summary = False
cfg.fit.save_misc = True

cfg.fit.save_fit_img = True # whether to save final inference iamges
cfg.fit.save_train_img = True
cfg.fit.save_fit_flame = False # whether to save FLAME parameters
cfg.fit.save_fit_video = True # whether to save final inference iamges & videos
cfg.fit.save_extra_viz = False # visualize loss map
cfg.fit.checkpoint_steps = 100
cfg.fit.resume = True

cfg.fit.finetune = True
cfg.fit.orig_deca = False
# cfg.fit.result_dir = '/local_data/MEAD_FLAME' # TODO - not used?
cfg.fit.dataset = 'MEAD'

cfg.train = CN() # eft.py엔 쓰이는 곳이 없음.
cfg.train.train_detail = False
cfg.train.max_epochs = 500
cfg.train.max_steps = 100
cfg.train.lr = 1e-4
cfg.train.log_dir = ''
cfg.train.log_steps = 10
cfg.train.vis_dir = 'train_images'
cfg.train.vis_steps = 200
cfg.train.write_summary = False
cfg.train.checkpoint_steps = 500
cfg.train.val_steps = 500
cfg.train.val_vis_dir = 'val_images'
cfg.train.eval_steps = 5000
cfg.train.resume = True
cfg.train.finetune = True

# ---------------------------------------------------------------------------- #
# Options for Losses
# ---------------------------------------------------------------------------- #
cfg.loss = CN()
cfg.loss.lmk = 1.0 # yml 있음 / is it is greater 0, we use 2d landmark loss and this value is for weight of 2d landmark loss

cfg.loss.useWlmk = True # whether to use weighted version of 2d landmark loss
cfg.loss.temporal = True # whether to use temporal loss
cfg.loss.temp_state = 0.0
cfg.loss.temp_2d = 0.0
cfg.loss.temp_3d = 0.0
cfg.loss.temp_verts = 0.0
cfg.loss.temp_weight = 100000.0 # 원랜 5000.0 / weight of temporal loss
cfg.loss.temp_smooth = 5000.0
cfg.loss.useVislmk = False
cfg.loss.eyed = 0.0
cfg.loss.lipd = 0.0
cfg.loss.photo = 2.0

# cfg.loss.spatial = "merge" #[naive, merge] / in yml
# cfg.loss.spatial_detach = False # In yml
# cfg.loss.spatial_huber = False
# cfg.loss.relative = False
# cfg.loss.spatial_down = False # in yml
cfg.loss.spatial_weight = 2.0 # in yml
cfg.loss.spatial_sample = True # in yml
# cfg.loss.spatial_icp = False # in yml

cfg.loss.useSeg = False
cfg.loss.id = 0.2
cfg.loss.id_shape_only = True
cfg.loss.reg_shape = 0.0 # 1e-04
cfg.loss.reg_exp = 0.0 # 1e-04
cfg.loss.reg_tex = 1e-04
cfg.loss.reg_light = 1.
cfg.loss.reg_jaw_pose = 0. #1.
cfg.loss.use_gender_prior = False
cfg.loss.shape_consistency = True

# loss for detail
cfg.loss.detail_consistency = True
cfg.loss.useConstraint = True
cfg.loss.mrf = 5e-2
cfg.loss.photo_D = 2.
cfg.loss.reg_sym = 0.005
cfg.loss.reg_z = 0.005
cfg.loss.reg_diff = 0.005

# uv loss
cfg.loss.uv_map_super = 100.0
cfg.loss.uv_l2 = True

cfg.loss.stricter_uv_mask = False
cfg.loss.delta_uv = 0.00005
cfg.loss.dist_uv = 20

# normal loss
cfg.loss.normal_super = 100.0
cfg.loss.normal_l2 = False
cfg.loss.delta_n = 0.33  # threshold that determines above which loss pixels in the normal loss map are considered outliers
cfg.loss.normal_mask_ksize = 13

cfg.eval=False
cfg.test_seq_path = "not this"


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()

def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--exp_name', type=str, default='debug', help='experiment name')
    parser.add_argument('--user', type=str, default="user", help='user identity')
    parser.add_argument('--test_seq_path', type=str, default=None)
    parser.add_argument('--cuda_device', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--view_num', type=int, default=None)
    parser.add_argument('--seg', type=bool, default=None)
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.cfg_file = None

    cfg.exp_name = "_".join((time.strftime('%m%d', time.localtime()), args.exp_name))
    cfg.user = args.user
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    if args.test_seq_path is not None:
        cfg.test_seq_path = args.test_seq_path

    if args.cuda_device is not None:
        cfg.cuda_device = args.cuda_device

    if args.batch_size is not None:
        cfg.dataset.batch_size = args.batch_size

    if args.dataset is not None:
        cfg.fit.dataset = args.dataset

    if args.view_num is not None:
        cfg.dataset.view_num = args.view_num

    if args.seg is not None:
        cfg.dataset.is_seg = args.seg

    print(f"\nConfigurations" + " -"*50)
    print(f"Config File: {cfg.cfg_file}")
    print(f"Experiment Name: {cfg.exp_name}")
    print(f"User: {cfg.user}")
    print(f"test_seq_path: {cfg.test_seq_path}")
    print(f"cuda_device: {cfg.cuda_device}")
    print(f"batch_size: {cfg.dataset.batch_size}")
    print(f"dataset: {cfg.fit.dataset}")

    return cfg
