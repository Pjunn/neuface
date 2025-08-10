import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as autograd
from functools import reduce
import torchvision.models as models
import cv2
import torchfile
from torch.autograd import Variable
# from . import util
import pdb
from . import utils_spatial_loss as util_s
torch.autograd.set_detect_anomaly(True)
from pytorch3d.transforms import axis_angle_to_quaternion
from pytorch3d.ops import knn_points

import pickle

def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

class Masking(nn.Module):
    def __init__(self, model_cfg):
        super(Masking, self).__init__()
        with open(f'{model_cfg.flame_mask}', 'rb') as f:
            ss = pickle.load(f, encoding='latin1')
            self.masks = Struct(**ss)
        with open(f'{model_cfg.flame_model_path}', 'rb') as f:
            ss = pickle.load(f, encoding='latin1')
            flame_model = Struct(**ss)
        self.dtype = torch.float32
        self.register_buffer('vertices', to_tensor(to_np(flame_model.v_template), dtype=self.dtype))
    def get_mask_eyes(self):
        left = self.masks.left_eyeball
        right = self.masks.right_eyeball

        return np.unique(np.concatenate((left, right)))

    def get_mask_eyes_rendering(self):
        eyes_mask = torch.zeros_like(self.vertices)[None]
        eyes_mask[:, self.get_mask_eyes(), :] = 1.0

        return eyes_mask

    def to_render_mask(self, mask):
        face_mask = torch.zeros_like(self.vertices)[None]
        face_mask[:, mask, :] = 1.0
        return face_mask

    def get_mask_face(self):
        return self.masks.face


class UVLoss():

    def __init__(self, model_cfg, stricter_mask : bool = False, delta_uv=0.00005, delta_nocs=0.0001, dist_uv=15):
        self.delta = delta_uv
        self.delta_nocs = delta_nocs
        self.dist_uv = dist_uv
        self.valid_verts = None
        self.valid_verts_nocs = None
        self.stricter_mask = stricter_mask
        """
        FLAME_UV_COORDS = f'{CODE_BASE}/assets/flame_uv_coords.npy'
        VALID_VERTS_NARROW = f'{CODE_BASE}/assets/uv_valid_verty_noEyes.npy'
        VALID_VERTS = f'{CODE_BASE}/assets/uv_valid_verty_noEyes_noEyeRegion_debug_wEars.npy'
        """
        if self.stricter_mask:
            self.valid_verts = np.load(f'{model_cfg.VALID_VERTS_NARROW}')
        else:
            self.valid_verts = np.load(f'{model_cfg.VALID_VERTS}')
        self.can_uv = torch.from_numpy(np.load(model_cfg.FLAME_UV_COORDS)[self.valid_verts, :]).cuda().unsqueeze(0).float()
        self.can_uv[..., 1] = (self.can_uv[..., 1] * -1) + 1

        self.verts_2d = []
        self.gt_2_verts = None

        self.valid_vertex_index = torch.from_numpy(self.valid_verts).long().cuda()



    def finish_stage1(self, delta_uv_fine=None, dist_uv_fine=None):
        self.verts_2d = torch.cat(self.verts_2d, dim=0)
        if delta_uv_fine is not None:
            self.delta = delta_uv_fine
            self.dist_uv = dist_uv_fine

    def is_next(self):
        self.gt_2_verts = None

    # @torch.compiler.disable
    def compute_corresp(self, gt, selected_frames=None):
        '''
        gt는 uv_map, flame can uv에서 vertex 들의 uv 값과 비교해서 픽셀을 찾는다.
        '''
        self.gt = gt

        gt_uv = gt[:, :2, :, :].permute(0, 2, 3, 1) # (b, h, w, 2)
        gt_uv = gt_uv.reshape(gt_uv.shape[0], -1, 2)  # B x n_pixel x 2
        can_uv = self.can_uv.repeat(gt_uv.shape[0], 1, 1) # 변형 없는 flame uv

        knn_result = knn_points(can_uv, gt_uv)
        pixel_position_width = knn_result.idx % gt.shape[-1]
        pixel_position_height = knn_result.idx // gt.shape[-2]
        self.dists = knn_result.dists.clone()

        self.gt_2_verts = torch.cat([pixel_position_width, pixel_position_height], dim=-1) # 가까운 pixel 위치 저장
        # if selected_frames is None:
        #     self.verts_2d.append(torch.cat([pixel_position_width, pixel_position_height], dim=-1))



    def compute_loss(self, proj_vertices, is_visible_verts_idx=None, selected_frames=None, uv_map=None, l2_loss=False):
        '''
        global 아닐때는 selected frame 없음.
        찾은 픽셀들에 대해서 픽셀 간의 차이 계산
        '''

        if is_visible_verts_idx is not None:
            not_occluded = is_visible_verts_idx[:, self.valid_vertex_index].float()
        else:
            not_occluded = torch.ones_like(self.valid_vertex_index).float().unsqueeze(0)


        if selected_frames is not None:
            gt_2_verts = self.verts_2d[selected_frames, :, :]
        else:
            gt_2_verts = self.gt_2_verts

        valid_proj_v = proj_vertices[:, self.valid_vertex_index, ..., :2]
        v_dist_2d = (gt_2_verts - valid_proj_v)
        if l2_loss:
            uv_loss = (
                ( v_dist_2d/ self.gt.shape[-1]) * (self.dists < self.delta) *
                (v_dist_2d.abs().sum(dim=-1) < self.dist_uv).unsqueeze(-1) *
                not_occluded.unsqueeze(-1)
            ).square().mean() * 100
        else:
            uv_loss = (
                    (v_dist_2d / self.gt.shape[-1]) * (self.dists < self.delta) *
                    (v_dist_2d.abs().sum(dim=-1) < self.dist_uv).unsqueeze(-1) *
                    not_occluded.unsqueeze(-1)
            ).abs().mean()
        return uv_loss


    def compute_loss_lstsq(self, proj_vertices):
        uv_loss = ((self.gt_2_verts / self.gt.shape[-1] - proj_vertices[:, torch.from_numpy(self.valid_verts).long().cuda(), :] /
                    self.gt.shape[-1]) * (self.dists < self.delta)   *
                    (
                               (self.gt_2_verts - proj_vertices[:, torch.from_numpy(self.valid_verts).long().cuda(), :]).abs().sum(
                                   dim=-1) < 30).unsqueeze(-1)
                   )
        #print(uv_loss)
        return uv_loss


def efficient_spatial_consistency_loss(
        verts, 
        orig_verts, 
        face, 
        sampling=True,
        dataset='MEAD'
    ):
    device = verts.device

    # visibility
    if dataset=='MEAD':
        verts = verts.view(-1, 7, 5023, 3) # [frame*view, 5023 (=vertices), 3 (=locdation)] -> [frame, 7 (=view), 5023, 3]
        sim = util_s.visibility_check(orig_verts, face)
        sim = sim.view(-1, 7, 5023) # [frame, view, vertices]
    else: # Celebv_HQ, Nersemble
        verts = verts.view(-1, 1, 5023, 3) # [frame*view, 5023 (=vertices), 3 (=locdation)] -> [frame, 7 (=view), 5023, 3]
        sim = util_s.visibility_check(orig_verts, face)
        sim = sim.view(-1, 1, 5023) # [frame, view, vertices]

    criterion = nn.L1Loss()
    all_loss = 0.0

    # select views for same timestep
    # frame을 sampling 한다.
    if sampling:
        sample_num = 5
        sample_idx = torch.randperm(len(verts))[:sample_num]
        div_factor = len(sample_idx)
    else:
        sample_idx = None
        div_factor = int(verts.shape[0] / 7)

    for verts_frame, sim_frame in zip(verts[sample_idx], sim[sample_idx]):
        # weighted sum the visibility
        # [7, 5023, 3], [7, 5023]
        sim_frame_ = util_s.softmax(sim_frame, verts_frame)
        merged_verts = torch.sum(verts_frame * sim_frame_.unsqueeze(2), axis=0) # 잘보이는 vertics의 위치로 가중합, [5023, 3]

        merged_verts = merged_verts.tile(sim_frame_.shape[0], 1, 1) # [7, 5023, 3]
        all_loss += criterion(merged_verts, verts_frame) # 평균 좌표랑 비슷하게 

    return all_loss / div_factor


def spatial_consistency_loss(
        verts, 
        image_name, 
        faces, 
        relative=False, 
        spatial_detach=True,
        spatial_huber=True, 
        down_weight=False, 
        sampling=True
    ):
    # parse frame id
    device = verts.device
    frame_id=[]
    for img_id in image_name:
        id = img_id.split("_")[1]
        if id not in frame_id:
            frame_id.append(id)

    if spatial_huber:
        criterion = nn.HuberLoss(reduction='sum')
    else:
        criterion = nn.L1Loss()
    # all_loss = torch.Tensor([0]).cuda()
    all_loss = 0.0
    # select 7 views for same timestep
    if sampling == True:
        sample_num = 5
        sample_idx = torch.randperm(len(frame_id))[:sample_num]
        div_factor = len(sample_idx)
    else:
        sample_idx = None
        div_factor = int(verts.shape[0] / 7)

    for bn,id in enumerate(frame_id):
        if sampling:
            if bn not in sample_idx:
                continue
        # frame_names = [ii for ii in image_name if id in ii] #['img_0000_front', 'img_0000_down', 'img_0000_top', 'img_0000_left_30', 'img_0000_left_60', 'img_0000_right_30', 'img_0000_right_60']
        frame_index = [index for index, ii in enumerate(image_name) if id in ii] #[0, 10, 20, 30, 40, 50, 60]
        ref_index = 1

        # visibility check for 7 views
        merged_verts, rotated_verts = util_s.weighted_sum_vertices(verts[frame_index], ref_index, faces, weighted_type="softmax", spatial_detach=spatial_detach)

        if relative:
            pairs = [[4597, 4051],
                     [4597, 3526],
                     [4051, 3526],
                     [3543, 2774],
                     [2774, 2850],
                     [2850, 2845],
                     [2845, 2897],
                     [2897, 3506],
                     [3506, 1794],
                     [1794, 1730],
                     [1730, 1735],
                     [1735, 1657],
                     [1657, 3543],
                     [3543, 3506],
                     [2774, 3506],
                     [1657, 3506]
                     ]
            for i, j in pairs:

                dist = torch.dist(merged_verts[i, :], merged_verts[j, :])
                for v in range(1, 7):
                    all_loss += F.mse_loss(dist, torch.dist(verts[bn*7+v, i, :],verts[bn*7+v, j, :]))

        else:
            # measure vertex loss between rotated vertices
            merged_verts = merged_verts.tile(rotated_verts.shape[0], 1, 1)

            if down_weight:
                downs = torch.ones(merged_verts.shape, device=device)
                downs[0] = 5
                all_loss += torch.mean(torch.abs(merged_verts - rotated_verts) * downs)
            else:
                all_loss += criterion(merged_verts, rotated_verts)

    return all_loss / div_factor
    # return torch.sqrt(((verts1 - verts2)**2).sum(2)).mean(1).mean()


L2_loss = torch.nn.MSELoss(reduction='none')
L1_loss = torch.nn.L1Loss()


def temporal_smooth_loss(r_aa, n_views=1):
    win_size = 2
    T = int(r_aa.shape[0] / n_views)
    r_aa_reshaped = r_aa.reshape(T, n_views, 3)
    r_qt = axis_angle_to_quaternion(r_aa_reshaped)
    r_qt_smooth = r_qt.clone()
    for t in range(win_size, T-win_size):
        conv_sum = torch.mean(r_qt[t-win_size:t+win_size], 0)
        norm_conv = F.normalize(conv_sum, dim=1)
        r_qt_smooth[t] = norm_conv

    l_temp_smooth = torch.mean(L2_loss(r_qt, r_qt_smooth), dim=(0, 2)).sum()
    return l_temp_smooth


def batch_kp_2d_l1_loss(real_2d_kp, predicted_2d_kp, weights=None):
    # TODO - fill the blank
    """Computes the l1 loss between the GT keypoints and the predicted ones

    Args:
        real_2d_kp : (frame*view, 68, 3)
        predicted_2d_kp : (frame*view, 68, 2)
    
    Returns:
        L1 loss between GT and prediction
    """
    if weights is not None:
        real_2d_kp[:, :, 2] = weights[None, :] * real_2d_kp[:, :, 2]
    kp_gt = real_2d_kp.view(-1, 3)
    kp_pred = predicted_2d_kp.contiguous().view(-1, 2)
    vis = kp_gt[:, 2] # 가시성
    k = torch.sum(vis) * 2.0 + 1e-8
    dif_abs = torch.abs(kp_gt[:, :2] - kp_pred).sum(1)
    return torch.matmul(dif_abs, vis) * 1.0 / k # 가시성 기반 정규화, 이미지마다 보이는 keypoint 개수가 다를수 있음.


def landmark_loss(predicted_landmarks, landmarks_gt, weight=1.):
    """Calculate 2D Landmark loss without weight sum

    Without weighted coefficient (same weight for every landmark)

    Args:
        predicted_landmarks () :
            Prediction of DECA decoder 
            shape: [frame * view, 68 (=kpts), 2 (=location)]
        landmarks_gt () : 
            Ground Truth landmarks which was predicted by FAN (or in dataset). 
            shape: [frame * view, 68, 3]
    """
    # (predicted_theta, predicted_verts, predicted_landmarks) = ringnet_outputs[-1]
    # if torch.is_tensor(landmarks_gt) is not True:
    #     real_2d = torch.cat(landmarks_gt).cuda()
    # else:
    #     real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1)).cuda()], dim=-1)
    # real_2d = torch.cat(landmarks_gt).cuda()
    
    # print("\n" + "lossfunc.py / landmark_loss()" + " -" * 50)
    # print(f"predicted_landmarks.shape: {predicted_landmarks.shape}")
    # print(f"landmarks_gt.shape: {landmarks_gt.shape}")

    loss_lmk_2d = batch_kp_2d_l1_loss(landmarks_gt, predicted_landmarks)
    return loss_lmk_2d * weight


def weighted_landmark_loss(predicted_landmarks, landmarks_gt, weight=1.):
    """Calculate 2D Landmark loss with weight sum

    Args:
        predicted_landmarks () :
            Prediction of DECA decoder 
            shape: [frame * view, 68 (=kpts), 2 (=location)]
        landmarks_gt () : 
            Ground Truth landmarks which was predicted by FAN (or in dataset). 
            shape: [frame * view, 68, 3]
    """

    #smaller inner landmark weights
    # (predicted_theta, predicted_verts, predicted_landmarks) = ringnet_outputs[-1]
    # import ipdb; ipdb.set_trace()
    real_2d = landmarks_gt
    weights = torch.ones((68,)).cuda()
    weights[5:7] = 2
    weights[10:12] = 2
    # nose points
    weights[27:36] = 1.5
    weights[30] = 3
    weights[31] = 3
    weights[35] = 3
    # inner mouth
    weights[60:68] = 1.5
    weights[48:60] = 1.5
    weights[48] = 3
    weights[54] = 3

    # print("\n" + "lossfunc.py / weighted_landmark_loss()" + " -" * 50)
    # print(f"predicted_landmarks.shape: {predicted_landmarks.shape}")
    # print(f"landmarks_gt.shape: {landmarks_gt.shape}")

    loss_lmk_2d = batch_kp_2d_l1_loss(real_2d, predicted_landmarks, weights)
    return loss_lmk_2d * weight


def bone_length_loss(self, rollout_joints3d):
    bones = rollout_joints3d[:, :, 1:]
    parents = rollout_joints3d[:, :, SMPL_PARENTS[1:]]
    bone_lengths = torch.norm(bones - parents, dim=-1)
    bone_length_diff = bone_lengths[:, 1:] - bone_lengths[:, :-1]
    loss = 0.5 * torch.sum(bone_length_diff ** 2)
    return loss


def l2_distance(verts1, verts2):
    return torch.sqrt(((verts1 - verts2)**2).sum(2)).mean(1).mean()

### VAE
def kl_loss(texcode):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    mu, logvar = texcode[:,:128], texcode[:,128:]
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return KLD

### ------------------------------------- Losses/Regularizations for shading
# white shading
# uv_mask_tf = tf.expand_dims(tf.expand_dims(tf.constant( self.uv_mask, dtype = tf.float32 ), 0), -1)
# mean_shade = tf.reduce_mean( tf.multiply(shade_300W, uv_mask_tf) , axis=[0,1,2]) * 16384 / 10379
# G_loss_white_shading = 10*norm_loss(mean_shade,  0.99*tf.ones([1, 3], dtype=tf.float32), loss_type = "l2")
def shading_white_loss(shading):
    '''
    regularize lighting: assume lights close to white 
    '''
    # rgb_diff = (shading[:,0] - shading[:,1])**2 + (shading[:,0] - shading[:,2])**2 + (shading[:,1] - shading[:,2])**2
    # rgb_diff = (shading[:,0].mean([1,2]) - shading[:,1].mean([1,2]))**2 + (shading[:,0].mean([1,2]) - shading[:,2].mean([1,2]))**2 + (shading[:,1].mean([1,2]) - shading[:,2].mean([1,2]))**2
    # rgb_diff = (shading.mean([2, 3]) - torch.ones((shading.shape[0], 3)).float().cuda())**2
    rgb_diff = (shading.mean([0, 2, 3]) - 0.99)**2
    return rgb_diff.mean()

def shading_smooth_loss(shading):
    '''
    assume: shading should be smooth
    ref: Lifting AutoEncoders: Unsupervised Learning of a Fully-Disentangled 3D Morphable Model using Deep Non-Rigid Structure from Motion
    '''
    dx = shading[:,:,1:-1,1:] - shading[:,:,1:-1,:-1]
    dy = shading[:,:,1:,1:-1] - shading[:,:,:-1,1:-1]
    gradient_image = (dx**2).mean() + (dy**2).mean()
    return gradient_image.mean()

### ------------------------------------- Losses/Regularizations for albedo
# texture_300W_labels_chromaticity = (texture_300W_labels + 1.0)/2.0
# texture_300W_labels_chromaticity = tf.divide(texture_300W_labels_chromaticity, tf.reduce_sum(texture_300W_labels_chromaticity, axis=[-1], keep_dims=True) + 1e-6)


# w_u = tf.stop_gradient(tf.exp(-15*tf.norm( texture_300W_labels_chromaticity[:, :-1, :, :] - texture_300W_labels_chromaticity[:, 1:, :, :], ord='euclidean', axis=-1, keep_dims=True)) * texture_vis_mask[:, :-1, :, :] )
# G_loss_local_albedo_const_u = tf.reduce_mean(norm_loss( albedo_300W[:, :-1, :, :], albedo_300W[:, 1:, :, :], loss_type = 'l2,1', reduce_mean=False, p=0.8) * w_u) / tf.reduce_sum(w_u+1e-6)

    
# w_v = tf.stop_gradient(tf.exp(-15*tf.norm( texture_300W_labels_chromaticity[:, :, :-1, :] - texture_300W_labels_chromaticity[:, :, 1:, :], ord='euclidean', axis=-1, keep_dims=True)) * texture_vis_mask[:, :, :-1, :] )
# G_loss_local_albedo_const_v = tf.reduce_mean(norm_loss( albedo_300W[:, :, :-1, :], albedo_300W[:, :, 1:, :],  loss_type = 'l2,1', reduce_mean=False, p=0.8) * w_v) / tf.reduce_sum(w_v+1e-6)

# G_loss_local_albedo_const = (G_loss_local_albedo_const_u + G_loss_local_albedo_const_v)*10

def albedo_constancy_loss(albedo, alpha = 15, weight = 1.):
    '''
    for similarity of neighbors
    ref: Self-supervised Multi-level Face Model Learning for Monocular Reconstruction at over 250 Hz
        Towards High-fidelity Nonlinear 3D Face Morphable Model
    '''
    albedo_chromaticity = albedo/(torch.sum(albedo, dim=1, keepdim=True) + 1e-6)
    weight_x = torch.exp(-alpha*(albedo_chromaticity[:,:,1:,:] - albedo_chromaticity[:,:,:-1,:])**2).detach()
    weight_y = torch.exp(-alpha*(albedo_chromaticity[:,:,:,1:] - albedo_chromaticity[:,:,:,:-1])**2).detach()
    albedo_const_loss_x = ((albedo[:,:,1:,:] - albedo[:,:,:-1,:])**2)*weight_x
    albedo_const_loss_y = ((albedo[:,:,:,1:] - albedo[:,:,:,:-1])**2)*weight_y
    
    albedo_constancy_loss = albedo_const_loss_x.mean() + albedo_const_loss_y.mean()
    return albedo_constancy_loss*weight

def albedo_ring_loss(texcode, ring_elements, margin, weight=1.):
        """
            computes ring loss for ring_outputs before FLAME decoder
            Inputs:
              ring_outputs = a list containing N streams of the ring; len(ring_outputs) = N
              Each ring_outputs[i] is a tensor of (batch_size X shape_dim_num)
              Aim is to force each row (same subject) of each stream to produce same shape
              Each row of first N-1 strams are of the same subject and
              the Nth stream is the different subject
        """
        tot_ring_loss = (texcode[0]-texcode[0]).sum()
        diff_stream = texcode[-1]
        count = 0.0
        for i in range(ring_elements - 1):
            for j in range(ring_elements - 1):
                pd = (texcode[i] - texcode[j]).pow(2).sum(1)
                nd = (texcode[i] - diff_stream).pow(2).sum(1)
                tot_ring_loss = torch.add(tot_ring_loss,
                                (torch.nn.functional.relu(margin + pd - nd).mean()))
                count += 1.0

        tot_ring_loss = (1.0/count) * tot_ring_loss
        return tot_ring_loss * weight

def albedo_same_loss(albedo, ring_elements, weight=1.):
        """
            computes ring loss for ring_outputs before FLAME decoder
            Inputs:
              ring_outputs = a list containing N streams of the ring; len(ring_outputs) = N
              Each ring_outputs[i] is a tensor of (batch_size X shape_dim_num)
              Aim is to force each row (same subject) of each stream to produce same shape
              Each row of first N-1 strams are of the same subject and
              the Nth stream is the different subject
        """
        loss = 0
        for i in range(ring_elements - 1):
            for j in range(ring_elements - 1):
                pd = (albedo[i] - albedo[j]).pow(2).mean()
                loss += pd
        loss = loss/ring_elements
        return loss * weight




def eye_dis(landmarks):
    # left eye:  [38,42], [39,41] - 1
    # right eye: [44,48], [45,47] -1
    eye_up = landmarks[:,[37, 38, 43, 44], :]
    eye_bottom = landmarks[:,[41, 40, 47, 46], :]
    dis = torch.sqrt(((eye_up - eye_bottom)**2).sum(2)) #[bz, 4]
    return dis

def eyed_loss(predicted_landmarks, landmarks_gt, weight=1.):
    if torch.is_tensor(landmarks_gt) is not True:
        real_2d = torch.cat(landmarks_gt).cuda()
    else:
        real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1)).cuda()], dim=-1)
    pred_eyed = eye_dis(predicted_landmarks[:,:,:2])
    gt_eyed = eye_dis(real_2d[:,:,:2])

    loss = (pred_eyed - gt_eyed).abs().mean()
    return loss

def lip_dis(landmarks):
    # up inner lip:  [62, 63, 64] - 1
    # down innder lip: [68, 67, 66] -1
    lip_up = landmarks[:,[61, 62, 63], :]
    lip_down = landmarks[:,[67, 66, 65], :]
    dis = torch.sqrt(((lip_up - lip_down)**2).sum(2)) #[bz, 4]
    return dis

def lipd_loss(predicted_landmarks, landmarks_gt, weight=1.):
    if torch.is_tensor(landmarks_gt) is not True:
        real_2d = torch.cat(landmarks_gt).cuda()
    else:
        real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1)).cuda()], dim=-1)
    pred_lipd = lip_dis(predicted_landmarks[:,:,:2])
    gt_lipd = lip_dis(real_2d[:,:,:2])

    loss = (pred_lipd - gt_lipd).abs().mean()
    return loss
    

def landmark_loss_tensor(predicted_landmarks, landmarks_gt, weight=1.):
    # (predicted_theta, predicted_verts, predicted_landmarks) = ringnet_outputs[-1]
    loss_lmk_2d = batch_kp_2d_l1_loss(landmarks_gt, predicted_landmarks)
    return loss_lmk_2d * weight


def ring_loss(ring_outputs, ring_type, margin, weight=1.):
    """
        computes ring loss for ring_outputs before FLAME decoder
        Inputs:
            ring_outputs = a list containing N streams of the ring; len(ring_outputs) = N
            Each ring_outputs[i] is a tensor of (batch_size X shape_dim_num)
            Aim is to force each row (same subject) of each stream to produce same shape
            Each row of first N-1 strams are of the same subject and
            the Nth stream is the different subject
        """
    tot_ring_loss = (ring_outputs[0]-ring_outputs[0]).sum()
    if ring_type == '51':
        diff_stream = ring_outputs[-1]
        count = 0.0
        for i in range(6):
            for j in range(6):
                pd = (ring_outputs[i] - ring_outputs[j]).pow(2).sum(1)
                nd = (ring_outputs[i] - diff_stream).pow(2).sum(1)
                tot_ring_loss = torch.add(tot_ring_loss,
                                (torch.nn.functional.relu(margin + pd - nd).mean()))
                count += 1.0

    elif ring_type == '33':
        perm_code = [(0, 1, 3),
                    (0, 1, 4),
                    (0, 1, 5),
                    (0, 2, 3),
                    (0, 2, 4),
                    (0, 2, 5),
                    (1, 0, 3),
                    (1, 0, 4),
                    (1, 0, 5),
                    (1, 2, 3),
                    (1, 2, 4),
                    (1, 2, 5),
                    (2, 0, 3),
                    (2, 0, 4),
                    (2, 0, 5),
                    (2, 1, 3),
                    (2, 1, 4),
                    (2, 1, 5)]
        count = 0.0
        for i in perm_code:
            pd = (ring_outputs[i[0]] - ring_outputs[i[1]]).pow(2).sum(1)
            nd = (ring_outputs[i[1]] - ring_outputs[i[2]]).pow(2).sum(1)
            tot_ring_loss = torch.add(tot_ring_loss,
                            (torch.nn.functional.relu(margin + pd - nd).mean()))
            count += 1.0

    tot_ring_loss = (1.0/count) * tot_ring_loss

    return tot_ring_loss * weight


######################################## images/features/perceptual
def gradient_dif_loss(prediction, gt):
    prediction_diff_x =  prediction[:,:,1:-1,1:] - prediction[:,:,1:-1,:-1]
    prediction_diff_y =  prediction[:,:,1:,1:-1] - prediction[:,:,1:,1:-1]
    gt_x =  gt[:,:,1:-1,1:] - gt[:,:,1:-1,:-1]
    gt_y =  gt[:,:,1:,1:-1] - gt[:,:,:-1,1:-1]
    diff = torch.mean((prediction_diff_x-gt_x)**2) + torch.mean((prediction_diff_y-gt_y)**2)
    return diff.mean()


def get_laplacian_kernel2d(kernel_size: int):
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size (int): filter size should be odd.

    Returns:
        Tensor: 2D tensor with laplacian filter matrix coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples::

        >>> kornia.image.get_laplacian_kernel2d(3)
        tensor([[ 1.,  1.,  1.],
                [ 1., -8.,  1.],
                [ 1.,  1.,  1.]])

        >>> kornia.image.get_laplacian_kernel2d(5)
        tensor([[  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1., -24.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.]])

    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or \
            kernel_size <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}"
                        .format(kernel_size))

    kernel = torch.ones((kernel_size, kernel_size))
    mid = kernel_size // 2
    kernel[mid, mid] = 1 - kernel_size ** 2
    kernel_2d: torch.Tensor = kernel
    return kernel_2d

def laplacian_hq_loss(prediction, gt):
    # https://torchgeometry.readthedocs.io/en/latest/_modules/kornia/filters/laplacian.html
    b, c, h, w = prediction.shape
    kernel_size = 3
    kernel = get_laplacian_kernel2d(kernel_size).to(prediction.device).to(prediction.dtype)
    kernel = kernel.repeat(c, 1, 1, 1)
    padding = (kernel_size - 1) // 2
    lap_pre = F.conv2d(prediction, kernel, padding=padding, stride=1, groups=c)
    lap_gt = F.conv2d(gt, kernel, padding=padding, stride=1, groups=c)

    return ((lap_pre - lap_gt)**2).mean()


## 
class VGG19FeatLayer(nn.Module):
    def __init__(self):
        super(VGG19FeatLayer, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True).features.eval().cuda()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    def forward(self, x):
        out = {}
        x = x - self.mean
        x = x/self.std
        ci = 1
        ri = 0
        for layer in self.vgg19.children():
            if isinstance(layer, nn.Conv2d):
                ri += 1
                name = 'conv{}_{}'.format(ci, ri)
            elif isinstance(layer, nn.ReLU):
                ri += 1
                name = 'relu{}_{}'.format(ci, ri)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                ri = 0
                name = 'pool_{}'.format(ci)
                ci += 1
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(ci)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            x = layer(x)
            out[name] = x
        # print([x for x in out])
        return out

class IDMRFLoss(nn.Module):
    def __init__(self, featlayer=VGG19FeatLayer):
        super(IDMRFLoss, self).__init__()
        self.featlayer = featlayer()
        self.feat_style_layers = {'relu3_2': 1.0, 'relu4_2': 1.0}
        self.feat_content_layers = {'relu4_2': 1.0}
        self.bias = 1.0
        self.nn_stretch_sigma = 0.5
        self.lambda_style = 1.0
        self.lambda_content = 1.0

    def sum_normalize(self, featmaps):
        reduce_sum = torch.sum(featmaps, dim=1, keepdim=True)
        return featmaps / reduce_sum

    def patch_extraction(self, featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = self.patches_OIHW.size()
        self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return self.patches_OIHW

    def compute_relative_distances(self, cdist):
        epsilon = 1e-5
        div = torch.min(cdist, dim=1, keepdim=True)[0]
        relative_dist = cdist / (div + epsilon)
        return relative_dist

    def exp_norm_relative_dist(self, relative_dist):
        scaled_dist = relative_dist
        dist_before_norm = torch.exp((self.bias - scaled_dist)/self.nn_stretch_sigma)
        self.cs_NCHW = self.sum_normalize(dist_before_norm)
        return self.cs_NCHW

    def mrf_loss(self, gen, tar):
        meanT = torch.mean(tar, 1, keepdim=True)
        gen_feats, tar_feats = gen - meanT, tar - meanT

        gen_feats_norm = torch.norm(gen_feats, p=2, dim=1, keepdim=True)
        tar_feats_norm = torch.norm(tar_feats, p=2, dim=1, keepdim=True)

        gen_normalized = gen_feats / gen_feats_norm
        tar_normalized = tar_feats / tar_feats_norm

        cosine_dist_l = []
        BatchSize = tar.size(0)

        for i in range(BatchSize):
            tar_feat_i = tar_normalized[i:i+1, :, :, :]
            gen_feat_i = gen_normalized[i:i+1, :, :, :]
            patches_OIHW = self.patch_extraction(tar_feat_i)

            cosine_dist_i = F.conv2d(gen_feat_i, patches_OIHW)
            cosine_dist_l.append(cosine_dist_i)
        cosine_dist = torch.cat(cosine_dist_l, dim=0)
        cosine_dist_zero_2_one = - (cosine_dist - 1) / 2
        relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
        rela_dist = self.exp_norm_relative_dist(relative_dist)
        dims_div_mrf = rela_dist.size()
        k_max_nc = torch.max(rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2)[0]
        div_mrf = torch.mean(k_max_nc, dim=1)
        div_mrf_sum = -torch.log(div_mrf)
        div_mrf_sum = torch.sum(div_mrf_sum)
        return div_mrf_sum

    def forward(self, gen, tar):
        ## gen: [bz,3,h,w] rgb [0,1]
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)
        style_loss_list = [self.feat_style_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for layer in self.feat_style_layers]
        self.style_loss = reduce(lambda x, y: x+y, style_loss_list) * self.lambda_style

        content_loss_list = [self.feat_content_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for layer in self.feat_content_layers]
        self.content_loss = reduce(lambda x, y: x+y, content_loss_list) * self.lambda_content

        return self.style_loss + self.content_loss

        # loss = 0
        # for key in self.feat_style_layers.keys():
        #     loss += torch.mean((gen_vgg_feats[key] - tar_vgg_feats[key])**2)
        # return loss


######################################################## vgg16 face

class VGG_16(nn.Module):
    """
    Main Class
    """

    def __init__(self):
        """
fr        Constructor
        """
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2622)
        self.mean = torch.Tensor(np.array([129.1863, 104.7624, 93.5940])/255.).float().view(1, 3, 1, 1).cuda()
        # self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()

    def load_weights(self, path="pretrained/VGG_FACE.t7"):
        """ Function to load luatorch pretrained
        Args:
            path: path for the luatorch pretrained
        """
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, "conv_%d_%d" % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                else:
                    self_layer = getattr(self, "fc%d" % (block))
                    block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]

    def forward(self, x):
        """ Pytorch forward
        Args:
            x: input image (224x224)
        Returns: class logits
        """
        out = {}
        x = x - self.mean
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        out['relu3_2'] = x
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        out['relu4_2'] = x
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, 0.5, self.training)
        x = self.fc8(x)
        out['last'] = x
        return out

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.featlayer = VGG_16().float()
        self.featlayer.load_weights(path="data/face_recognition_model/vgg_face_torch/VGG_FACE.t7")
        self.featlayer = self.featlayer.cuda().eval()
        self.feat_style_layers = {'relu3_2': 1.0, 'relu4_2': 1.0}
        self.feat_content_layers = {'relu4_2': 1.0}
        self.bias = 1.0
        self.nn_stretch_sigma = 0.5
        self.lambda_style = 1.0
        self.lambda_content = 1.0

    def sum_normalize(self, featmaps):
        reduce_sum = torch.sum(featmaps, dim=1, keepdim=True)
        return featmaps / reduce_sum

    def patch_extraction(self, featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = self.patches_OIHW.size()
        self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return self.patches_OIHW

    def compute_relative_distances(self, cdist):
        epsilon = 1e-5
        div = torch.min(cdist, dim=1, keepdim=True)[0]
        relative_dist = cdist / (div + epsilon)
        return relative_dist

    def exp_norm_relative_dist(self, relative_dist):
        scaled_dist = relative_dist
        dist_before_norm = torch.exp((self.bias - scaled_dist)/self.nn_stretch_sigma)
        self.cs_NCHW = self.sum_normalize(dist_before_norm)
        return self.cs_NCHW

    def mrf_loss(self, gen, tar):
        meanT = torch.mean(tar, 1, keepdim=True)
        gen_feats, tar_feats = gen - meanT, tar - meanT

        gen_feats_norm = torch.norm(gen_feats, p=2, dim=1, keepdim=True)
        tar_feats_norm = torch.norm(tar_feats, p=2, dim=1, keepdim=True)

        gen_normalized = gen_feats / gen_feats_norm
        tar_normalized = tar_feats / tar_feats_norm

        cosine_dist_l = []
        BatchSize = tar.size(0)

        for i in range(BatchSize):
            tar_feat_i = tar_normalized[i:i+1, :, :, :]
            gen_feat_i = gen_normalized[i:i+1, :, :, :]
            patches_OIHW = self.patch_extraction(tar_feat_i)

            cosine_dist_i = F.conv2d(gen_feat_i, patches_OIHW)
            cosine_dist_l.append(cosine_dist_i)
        cosine_dist = torch.cat(cosine_dist_l, dim=0)
        cosine_dist_zero_2_one = - (cosine_dist - 1) / 2
        relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
        rela_dist = self.exp_norm_relative_dist(relative_dist)
        dims_div_mrf = rela_dist.size()
        k_max_nc = torch.max(rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2)[0]
        div_mrf = torch.mean(k_max_nc, dim=1)
        div_mrf_sum = -torch.log(div_mrf)
        div_mrf_sum = torch.sum(div_mrf_sum)
        return div_mrf_sum

    def forward(self, gen, tar):
        ## gen: [bz,3,h,w] rgb [0,1]
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)
        style_loss_list = [self.feat_style_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for layer in self.feat_style_layers]
        self.style_loss = reduce(lambda x, y: x+y, style_loss_list) * self.lambda_style

        content_loss_list = [self.feat_content_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for layer in self.feat_content_layers]
        self.content_loss = reduce(lambda x, y: x+y, content_loss_list) * self.lambda_content

        return self.style_loss + self.content_loss
        # loss = 0
        # for key in self.feat_style_layers.keys():
        #     loss += torch.mean((gen_vgg_feats[key] - tar_vgg_feats[key])**2)
        # return loss

##############################################
## ref: https://github.com/cydonia999/VGGFace2-pytorch
from ..models.frnet import resnet50, load_state_dict
class VGGFace2Loss(nn.Module):
    def __init__(self, pretrained_model, pretrained_data='vggface2'):
        super(VGGFace2Loss, self).__init__()
        self.reg_model = resnet50(num_classes=8631, include_top=False).eval().cuda()
        load_state_dict(self.reg_model, pretrained_model)
        self.mean_bgr = torch.tensor([91.4953, 103.8827, 131.0912]).cuda()

    def reg_features(self, x):
        # out = []
        margin=10
        x = x[:,:,margin:224-margin,margin:224-margin]
        # x = F.interpolate(x*2. - 1., [224,224], mode='nearest')
        x = F.interpolate(x*2. - 1., [224,224], mode='bilinear')
        # import ipdb; ipdb.set_trace()
        feature = self.reg_model(x)
        feature = feature.view(x.size(0), -1)
        return feature

    def transform(self, img):
        # import ipdb;ipdb.set_trace()
        img = img[:, [2,1,0], :, :].permute(0,2,3,1) * 255 - self.mean_bgr
        img = img.permute(0,3,1,2)
        return img

    def _cos_metric(self, x1, x2):
        return 1.0 - F.cosine_similarity(x1, x2, dim=1)

    def forward(self, gen, tar, is_crop=True):
        gen = self.transform(gen)
        tar = self.transform(tar)

        gen_out = self.reg_features(gen)
        tar_out = self.reg_features(tar)
        # loss = ((gen_out - tar_out)**2).mean()
        loss = self._cos_metric(gen_out, tar_out).mean()
        return loss