import torch
import json
import numpy as np
import os
from tqdm import tqdm
import glob
import pickle
import random

import cv2
import imageio
from .ray_utils import get_ray_directions
from .color_utils import read_image

from .base import BaseDataset
import imutils
import h5py

def load_h5(path):
    fx_input = h5py.File(path, 'r')
    x = fx_input['data'][:]
    fx_input.close()
    return x


def load_models(path):
    models = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            model = os.path.basename(line[:-1])
            model = model[:-15]
            models.append(model)

    return models


trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, t],
    [0, 0, 1, 0],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()
    
 
  
rot_x = lambda th:  np.array([
    [1, 0, 0, 0],
    [0, np.cos(th), -np.sin(th), 0],
    [0, np.sin(th), np.cos(th), 1],
    [0, 0, 0, 1]])

rot_y = lambda th:  np.array([
    [np.cos(th), 0, np.sin(th), 0],
    [0, 1, 0, 0],
    [-np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]])
    
rot_z = lambda th: np.array([
    [np.cos(th),-np.sin(th), 0,0],
    [np.sin(th), np.cos(th), 0,  0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]])


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    #c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    c2w = c2w.numpy()
    return c2w



def read_pickle_file(path):
    objects = []
    with open(path, "rb") as fp:
        while True:
            try:
                obj = pickle.load(fp)
                objects.append(obj)

            except EOFError:
                break

    return objects


def load_dataset(directory):
    # print(directory)

    cam_data_path = os.path.join(directory, "cam_data.pkl")
    cam_data = read_pickle_file(cam_data_path)[0]
    cams = {"width": 1280, "height": 720}

    imgs = {}
    image_dir = os.path.join(directory, "render/")
    images = glob.glob(image_dir + "**/*.png", recursive=True)
    images.sort()

    mask_dir = os.path.join(directory, "mask/")
    depth_dir = os.path.join(directory, "depth/")

    for i in range(len(images)):
        image_current = images[i]
        image_id = os.path.basename(image_current).split(".")[0]
        image_parent_dir = image_current.split("/")[-2]

        cam = cam_data[image_id]["K"]
        [cams["fx"], cams["fy"], cams["cx"], cams["cy"]] = cam
        
        pose = cam_data[image_id]["extrinsics_opencv"]
        pose = np.vstack([pose, np.array([0, 0, 0, 1])])
        pose = np.linalg.inv(pose)
        # print(pose)

        imgs[i] = {
            "camera_id": image_id,
            "t": pose[:3, 3].reshape(3, 1),
            "R": pose[:3, :3],
            "path": images[i],
            "pose": pose
        }

        imgs[i]["mask_path"] = os.path.join(mask_dir, "%s/%s_seg.png" % (image_parent_dir, image_id))
        imgs[i]["depth_path"] = os.path.join(depth_dir, "%s/%s_depth.npz" % (image_parent_dir, image_id))

    return imgs, cams

def pallette_to_labels(mask):
    uniq_vals = np.unique(mask)

    for i in range(len(uniq_vals)):
        mask = np.where(mask == uniq_vals[i], i, mask)

    return mask

def load_data(imgs, cams, max_ind=54, skip=1, res=1):
    all_ids = []
    all_imgs = []
    all_poses = []
    all_depths = []
    all_seg_masks = []

    flip_x = np.eye(4)
    flip_x[2, 2] *= -1
    flip_x[1, 1] *= -1
    flip_x = np.linalg.inv(flip_x)
    
    t = np.array([0.0, -0.5, 4.5]).T
    nerf_w_2_transform_w = np.identity(4)
    nerf_w_2_transform_w[:3, -1] = -t
    
    nerf_w_2_transform_w_1 = np.identity(4)
    nerf_w_2_transform_w_1[:3, -1] = t

    for index in range(0, max_ind, skip):
        all_ids.append(imgs[index]["camera_id"])

        n_image = imageio.imread(imgs[index]["path"]) / 255.0
        h, w = n_image.shape[:2]
        resized_h = round(h * res)
        resized_w = round(w * res)
        n_image = cv2.resize(n_image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        all_imgs.append(n_image)

        
        n_pose = imgs[index]["pose"] 
        all_poses.append(n_pose)

        n_seg_mask = cv2.imread(imgs[index]["mask_path"], cv2.IMREAD_GRAYSCALE)
        n_seg_mask = cv2.resize(n_seg_mask, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        n_seg_mask = pallette_to_labels(n_seg_mask)
        all_seg_masks.append(n_seg_mask)

        #n_depth = np.load(imgs[index]["depth_path"])['arr_0']
        #n_depth = np.where(n_depth == np.inf, 0, n_depth)
        #n_depth = np.where(n_depth > 100, 0, n_depth)
        #n_depth = cv2.resize(n_depth, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        #all_depths.append(n_depth)

    # all_poses = [all_poses[all_ids.index("left_5")]]
    all_imgs = np.array(all_imgs).astype(np.float32)
    all_poses = np.array(all_poses)
    all_depths = np.array(all_depths).astype(np.float32)
    
    i_val = []
    sides = ["back", "bottom", "front", "left", "right", "top"]
    for side_idx in range(len(sides)):
        panel_idx = np.random.randint(1, 10)
        val_camera_id = "%s_%d" % (sides[side_idx], panel_idx)
        val_idx = all_ids.index(val_camera_id)
        i_val.append(val_idx)

    indices = np.arange(len(all_imgs))
    i_train = np.array(list(set(indices).difference(set(i_val))))
    i_test = i_val
    i_split = [i_train, i_val, i_test]

    #render_poses = ([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)
   
    
    
    # center the poses
    
    all_poses[:, :3], _ = center_poses(all_poses[:, :3])

    scale_factor_max = all_poses[:, :3, -1].max()
    scale_factor_min = all_poses[:, :3, -1].min()

    scale_factor = max(scale_factor_max, -scale_factor_min)

    all_poses[:, :3, -1] = all_poses[:, :3, -1] / (scale_factor + 1e-6) * 0.5
    
    # render the canonical path
    input_poses_path = "/home2/shaurya.dewan/NOCs/gradient_condor/gradient_density_siamese_rots/car/car_input_rot.h5"
    canonical_poses_path = "/home2/shaurya.dewan/NOCs/gradient_condor/gradient_density_siamese_rots/car/car_canonical.h5"
    canonical_models_path = "/home2/shaurya.dewan/NOCs/gradient_condor/gradient_density_siamese_rots/car/car_files.txt"
    
    

    input_poses = load_h5(input_poses_path)
    canonical_frames = load_h5(canonical_poses_path)
    canonical_models = load_models(canonical_models_path)
    
    
    for _ in range(len(canonical_models)):
        model_name = canonical_models[_]
        #02958343_3870022ab97149605697f65a2d559457
        #02958343_324434f8eea2839bf63ee8a34069b7c5
        if model_name != "02958343_3870022ab97149605697f65a2d559457":
            continue
        print(_)
        
        canonical_poses = []
        anchor_frame = all_poses[all_ids.index("back_1")]
        canonical_frame = np.identity(4)
        input_frame  = np.identity(4)
        
        canonical_frame[:3,:3] = canonical_frames[_]
        input_frame[:3,:3] = input_poses[_]
        
        
        anchor_frame_1 =  np.linalg.inv(canonical_frame) @ np.linalg.inv(input_frame)
        #anchor_frame_1 = np.linalg.inv(input_frame) 
        num_poses = 200
        

        for i in range(num_poses):
            angle = np.linspace(0, 360, num_poses)[i]
            circular_pose = anchor_frame_1 @ np.linalg.inv(pose_spherical(-angle,0.0,0.)) @ anchor_frame
            canonical_poses.append(circular_pose)
        
        
        all_poses = np.array(canonical_poses)
    
    
    render_poses = []
    return all_imgs, all_poses, render_poses, cams, all_seg_masks, all_depths, i_split


def get_rays_np(H, W, K, c2w, mesh_grid=None):
    # print(K.shape)
    # print(H, W)
    if mesh_grid is None:
        i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    else:
        i, j = mesh_grid
    # print(i.shape, j.shape)
    dirs = np.stack([(i - K[0][2]) / K[0][0], (j - K[1][2]) / K[1][1], np.ones_like(i)], -1)

    # print(dirs.shape)
    # Rotate ray directions from camera frame to the world frame
    # rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3].numpy(), -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # print(dirs.shape, np.transpose(dirs, (2, 0, 1)).shape)
    # print(dirs.shape, "before")
    rays_d = np.transpose(c2w[:3, :3].numpy() @ np.transpose(dirs.reshape(-1, 3), (1, 0)), (1, 0))
    rays_d = rays_d.reshape(H, W, 3)
    # print(rays_d.shape, "after")
    # rays_d = np.transpose(rays_d, (1, 2, 0))
    # print(rays_d.shape)
    rays_d = rays_d / (np.linalg.norm(rays_d, ord=2, axis=-1, keepdims=True) + 1e-6)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return torch.tensor(rays_o), torch.tensor(rays_d)


def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    return poses_centered, pose_avg


class BRICSDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()

        if kwargs.get('read_meta', True):
            self.read_meta(split)

    def read_intrinsics(self):
        imgs, cams = load_dataset(self.root_dir)
        # print(imgs)
        # print(cams)
        scale = 1

        cams["fx"] = fx = cams["fx"] * scale
        cams["fy"] = fy = cams["fy"] * scale
        cams["cx"] = cx = cams["cx"] * scale
        cams["cy"] = cy = cams["cy"] * scale

        rand_key = random.choice(list(imgs))
        test_img = cv2.imread(imgs[rand_key]["path"])
        h, w = test_img.shape[:2]

        cams["height"] = round(h * scale)
        cams["width"] = round(w * scale)
        self.img_wh = (w, h)
        cams["intrinsic_mat"] = np.float32([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        self.K = torch.FloatTensor(cams["intrinsic_mat"])
        self.blender2opencv = np.array([[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1.0, 0], [0, 0, 0, 1.0]], dtype=float)

        all_imgs, all_poses, render_poses, cams, all_seg_masks, all_depths, i_split = load_data(imgs, cams)

        #all_poses[:, :3], _ =  (all_poses[:, :3])
        
        all_poses[:, :3], _ = center_poses(all_poses[:, :3])

        scale_factor_max = all_poses[:, :3, -1].max()
        scale_factor_min = all_poses[:, :3, -1].min()

        scale_factor = max(scale_factor_max, -scale_factor_min)

        all_poses[:, :3, -1] = all_poses[:, :3, -1] / (scale_factor + 1e-6) * 0.5
        
        self.directions = get_ray_directions(h, w, self.K)  # TODO: check if it is correct

        self.all_imgs = all_imgs
        self.all_poses = all_poses
        self.i_split = i_split
        self.all_depths = all_depths
        self.all_seg_masks = all_seg_masks

    def read_meta(self, split):
        self.rays = []
        self.poses = []

        if split == 'trainval':
            frames = self.i_split[0]
            frames += self.i_split[1]
        else:
            if split == 'train':
                frames = self.i_split[0]
            elif split == 'val':
                frames = self.i_split[1]
            elif split == 'test':
                frames = self.i_split[2]

        print(f'Loading {len(frames)} {split} images ...')
        
        
        for idx in frames:
            c2w = self.all_poses[idx]
            # rays_o, rays_d = get_rays_np(img.shape[0], img.shape[1], K,
            #                              (torch.FloatTensor(self.blender2opencv))[:3, :4] @ c2w)
            img = self.all_imgs[idx][:, :, :3]
            mask = self.all_seg_masks[idx]
            mask = mask.reshape(720, 1280, 1)
            img = img[..., :3] * mask + (1.0 - mask)
            img = torch.FloatTensor(img[..., :3].reshape(-1, 3))
            self.rays += [img]
            #c2w = np.matmul(self.blender2opencv[:3, :4], c2w)
            # c2w[:, 1:3] *= -1  # [right up back] to [right down front]
            # pose_radius_scale = 1.5 #scene scale
            # c2w[:, 3] /= np.linalg.norm(c2w[:, 3])/pose_radius_scale
            self.poses += [c2w]

        if len(self.rays) > 0:
            self.rays = torch.stack(self.rays)  # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses)  # (N_images, 3, 4)