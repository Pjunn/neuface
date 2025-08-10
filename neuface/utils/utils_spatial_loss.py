import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn.functional as F
import pdb
# from torch_batch_svd import svd

torch.autograd.set_detect_anomaly(True)
def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None] # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]
    faces = faces.reshape(-1, 3)
    vertices_faces = vertices_faces.reshape(-1, 3, 3)
    normals.index_add_(0, faces[:, 1].long(),
                       torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(0, faces[:, 2].long(),
                       torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(0, faces[:, 0].long(),
                       torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0]))

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = torch.mean(A, axis=0)
    centroid_B = torch.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = torch.matmul(AA.T, BB)
    U, S, Vt = torch.linalg.svd(H)
    # U, S, Vt = svd(H)
    # Vt=Vt.T
    R = torch.matmul(Vt.T, U.T)
    # special reflection case
    if torch.det(R) < 0:
        Vt[m - 1, :] = Vt[m - 1, :]*(-1)
        R = torch.matmul(Vt.T, U.T)

    # translation
    t = centroid_B.T - torch.matmul(R, centroid_A.T)

    # homogeneous transformation
    T = torch.eye(m + 1).cuda()
    T[:m, :m] = R
    T[:m, m] = t
    return T, R, t

def icp(A, B, overlap_index=None):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]
    if overlap_index==None:
        A_vis, B_vis = A, B
    else:
        A_vis,B_vis = A[overlap_index,:],B[overlap_index,:]

    # make points homogeneous, copy them to maintain the originals
    src_orig = torch.ones((m + 1, A.shape[0])).cuda()
    src = torch.ones((m + 1, A_vis.shape[0])).cuda()
    dst = torch.ones((m + 1, B_vis.shape[0])).cuda()

    src_orig[:m,:]=A.T.clone()
    src[:m, :] = A_vis.T.clone()
    dst[:m, :] = B_vis.T.clone()

    # calculate final transformation
    T, _, _ = best_fit_transform(A_vis, dst[:m, :].T)

    return T, src_orig, dst

def find_overlap_index(A,B):
    overlap=[i for i in A if i in B]
    return overlap

def weighted(sim, vertices):
    sim_ = np.copy(sim)
    sim_total = np.sum(sim_, axis=0)
    for index,_ in enumerate(sim_):
        sim_[index,sim_total == 0] = 1
    sim_total[sim_total == 0] = vertices.shape[0]
    sim_ = sim_ / sim_total

    return sim_[:,:,np.newaxis]

def softmax(sim, vertices, T=0.1):
    softmax = torch.nn.Softmax()
    sim_ = sim.clone()
    sim_[sim_==0]=-100

    sim_ = softmax(sim_.T/T) # [5023, 7], 한 프레임에 대해서 camera로 봤을때 잘보이는 view에 가중치 높게주는 분포 만들기
    sim_ = sim_.T

    return sim_

def find_visible_index_v2(vertices, vert_depth, face):
    # ## too deep depth cannot been seen
    # z = vertices[0, :, 2]
    # deepz_index = np.where(z < -0.1)[0]
    #
    # ## compute normal vectors
    # normals = vertex_normals(torch.Tensor(vertices), torch.Tensor(face).expand(1, -1, -1))
    #
    # ## compute the cosine sim. between ray and normal vector
    # ## compute visible vertices
    # ray = torch.zeros((1, 3, 1)); ray[0, 2, 0] = 1
    #
    # sim = torch.bmm(normals, ray).numpy()
    # vis_index = np.where(sim > 0)[1]
    # temp = np.arange(5023)
    # vis_index_final = [i for i in vis_index if i not in deepz_index]
    # # invis_index = [i for i in temp if i not in vis_index_final]
    # sim[sim <= 0] = 0
    # return vis_index_final, sim[0,:,0]

    ## compute normal vectors
    normals = vertex_normals(torch.Tensor(vertices), torch.Tensor(face).expand(1, -1, -1))
    ## compute the cosine sim. between ray and normal vector
    ## compute visible vertices
    ray = torch.zeros((1, 3, 1)); ray[0, 2, 0] = 1
    invis_index = [index for index, i in enumerate(vert_depth) if i == -1]
    sim=torch.bmm(normals, ray).numpy()
    sim[:,invis_index,:]=0
    sim[sim<0.2] = 0
    vis_index = np.where(sim >= 0.2)[1]
    # sim = torch.bmm(normals[:,vis_index_,:], ray).numpy()
    # vis_index = np.where(sim > 0.2)[1]
    # vis_index_final=vis_index_.detach().cpu().numpy()[vis_index]
    return vis_index, sim[0,:,0]

def find_visible_index(vertices, face):
    ## too deep depth cannot been seen
    z = vertices[0, :, 2]
    deepz_index = np.where(z < -0.1)[0]

    ## compute normal vectors
    normals = vertex_normals(torch.Tensor(vertices), torch.Tensor(face).expand(1, -1, -1))

    ## compute the cosine sim. between ray and normal vector
    ## compute visible vertices
    ray = torch.zeros((1, 3, 1)); ray[0, 2, 0] = 1

    sim = torch.bmm(normals, ray).numpy()
    vis_index = np.where(sim > 0)[1]
    temp = np.arange(5023)
    vis_index_final = [i for i in vis_index if i not in deepz_index]
    # invis_index = [i for i in temp if i not in vis_index_final]
    sim[sim <= 0] = 0
    return vis_index_final, sim[0,:,0]

def visibility_check(vertices, face):
    ## batch and z axis
    batch=vertices.shape[0]
    z = vertices[:, :, 2]

    ## compute normal vectors
    normals = vertex_normals(vertices, face.tile(batch,1,1)) # [batch, 5023, 3]

    ####### For opdict["verts"]
    ## compute cosine similairty between ray and normal vector
    ray = torch.zeros((batch, 3, 1)).cuda()
    ray[:, 2, 0] = 1
    sim = torch.bmm(normals, ray).squeeze(2)  # [batch, 5023, 1], cosine sim score
    sim = sim
    sim_ = sim.clone()
    ## filter out small similarity and too deep depth
    sim_[z < -0.08] = 0  # too deep depth
    sim_[sim < 0.28] = 0  # uncertain region

    ####### For opdict["trans_verts"]
    ## compute cosine similairty between ray and normal vector
    """ray = torch.zeros((batch, 3, 1)).cuda()
    ray[:, 2, 0] = 1
    sim = torch.bmm(normals, ray).squeeze(2) # [batch, 5023, 1], cosine sim score
    sim = sim*(-1)
    sim_ = sim.clone()
    ## filter out small similarity and too deep depth
    sim_[z>10.8]=0 #too deep depth
    sim_[sim<0.25]=0 # uncertain region"""

    return sim_


def weighted_sum_vertices(vertices, front_index,faces, vert_depth=None,weighted_type="weighted", spatial_detach=True):
    # visibility check
    if vert_depth==None:
        sim = visibility_check(vertices, faces)
    else:
        vis_f, sim_f = find_visible_index_v2(vertices, vert_depth[front_index],faces)

    # rotate and align with ref_index frame
    vert_ref = vertices[front_index]
    vis_ref = torch.where(sim[front_index]>0)[0]

    # new vertices will be merged as a single vertices
    new_vertices = torch.zeros(vertices.shape).cuda()

    for i, verts in enumerate(vertices):
        if i==front_index:
            new_vertices[i]=vert_ref
            continue
        vis = torch.where(sim[i] > 0)[0]
        overlap_index = find_overlap_index(vis_ref, vis)

        # run icp and find the new aligned vertices
        if spatial_detach:
            T, _, _ = icp(verts.clone().detach(), vert_ref.clone().detach(), overlap_index)
        else:
            T, _, _ = icp(verts, vert_ref, overlap_index)

        src = torch.ones((verts.shape[1] + 1, verts.shape[0])).cuda()
        src[:verts.shape[1], :] = verts.T.clone()
        vertices_rotation = torch.matmul(T, src)[:3, :]  # aligned vertice to the frontal view

        # save rotated vertices
        vertices_rotation = vertices_rotation.T
        new_vertices[i] = vertices_rotation

    if weighted_type=="weighted":
        sim_ = weighted(sim, vertices)
    else:
        sim_ = softmax(sim, vertices)

    if spatial_detach:
        new_vertices_clone =new_vertices.clone().detach()
        sim_ = sim_.detach()
        weighted_vertices = torch.sum(new_vertices_clone*sim_.unsqueeze(2),axis=0)
    else:
        weighted_vertices = torch.sum(new_vertices * sim_.unsqueeze(2), axis=0)
    return weighted_vertices, new_vertices

def weighted_sum_vertices_naive(vertices, front_index,faces, vert_depth=None,weighted_type="weighted", spatial_detach=True):
    # visibility check
    if vert_depth==None:
        sim = visibility_check(vertices, faces)
    else:
        vis_f, sim_f = find_visible_index_v2(vertices, vert_depth[front_index],faces)

    # rotate and align with ref_index frame
    vert_ref = vertices[front_index]
    vis_ref = torch.where(sim[front_index]>0)[0]

    # new vertices will be merged as a single vertices
    new_vertices = torch.zeros(vertices.shape).cuda()

    for i, verts in enumerate(vertices):
        if i==front_index:
            new_vertices[i]=vert_ref
            continue
        vis = torch.where(sim[i] > 0)[0]
        overlap_index = find_overlap_index(vis_ref, vis)

        # run icp and find the new aligned vertices
        if spatial_detach:
            T, _, _ = icp(verts.clone().detach(), vert_ref.clone().detach(), overlap_index)
        else:
            T, _, _ = icp(verts, vert_ref, overlap_index)

        src = torch.ones((verts.shape[1] + 1, verts.shape[0])).cuda()
        src[:verts.shape[1], :] = verts.T.clone()
        vertices_rotation = torch.matmul(T, src)[:3, :]  # aligned vertice to the frontal view

        # save rotated vertices
        vertices_rotation = vertices_rotation.T
        new_vertices[i] = vertices_rotation

    if weighted_type=="weighted":
        sim_ = weighted(sim, vertices)
    else:
        sim_ = softmax(sim, vertices)

    if spatial_detach:
        new_vertices_clone =new_vertices.clone().detach()
        sim_ = sim_.detach()
    else:
        new_vertices_clone =new_vertices

    # weighted_vertices = torch.sum(new_vertices_clone*sim_.unsqueeze(2),axis=0)

    return vert_ref, new_vertices_clone, sim_

def rotate_vertices(verts, trans_verts, front_index=1):
    # new vertices will be merged as a single vertices
    new_vertices = torch.zeros(verts.shape).cuda()
    new_trans_vertices = torch.zeros(verts.shape).cuda()

    new_vertices[front_index]=verts[front_index]
    new_trans_vertices[front_index] = trans_verts[front_index]

    for i, vert_comb in enumerate(zip(verts,trans_verts)):
        vert, trans_vert = vert_comb[0], vert_comb[1]
        if i == front_index:
            continue
        # run icp and find the new aligned vertices
        T, src, dst = icp(vert, new_vertices[front_index])
        vertices_rotation = torch.matmul(T, src)[:3, :]  # aligned vertice to the frontal view

        # save rotated vertices
        vertices_rotation = vertices_rotation.T
        new_vertices[i] = vertices_rotation

        # run icp and find the new aligned vertices
        T, src, dst = icp(trans_vert, new_trans_vertices[front_index])
        vertices_rotation = torch.matmul(T, src)[:3, :]  # aligned vertice to the frontal view

        # save rotated vertices
        vertices_rotation = vertices_rotation.T
        new_trans_vertices[i] = vertices_rotation

    return new_vertices, new_trans_vertices


def rotate_vertices_single(verts, trans_verts, ref_vertices, ref_transformed_vertices):
    # new vertices will be merged as a single vertices
    new_vertices = torch.zeros(verts.shape).cuda()
    new_trans_vertices = torch.zeros(verts.shape).cuda()

    # run icp and find the new aligned vertices
    T, src, dst = icp(verts[0], ref_vertices[0])
    vertices_rotation = torch.matmul(T, src)[:3, :]  # aligned vertice to the frontal view

    # save rotated vertices
    vertices_rotation = vertices_rotation.T
    new_vertices = vertices_rotation

    # run icp and find the new aligned vertices
    T, src, dst = icp(trans_verts[0], ref_transformed_vertices[0])
    vertices_rotation = torch.matmul(T, src)[:3, :]  # aligned vertice to the frontal view

    # save rotated vertices
    vertices_rotation = vertices_rotation.T
    new_trans_vertices = vertices_rotation

    return new_vertices, new_trans_vertices