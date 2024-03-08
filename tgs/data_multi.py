import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from tgs.utils.config import parse_structured
from tgs.utils.ops import get_intrinsic_from_fov, get_ray_directions, get_rays
from tgs.utils.typing import *


def complete_pts(pts):
    """ completes pts to be (..., 4, 1) """
    if pts.shape[-1] != 1:
        pts = pts[..., None]
    s = pts.shape
    if s[-2] == 4:
        return pts
    if isinstance(pts, np.ndarray):
        return np.concatenate((pts, np.ones((*s[:-2], 1, 1), dtype=pts.dtype)), axis=-2)
    return torch.cat((pts, torch.ones(*s[:-2], 1, 1, dtype=pts.dtype, device=pts.device)), dim=-2)


def complete_trans(T):
    """ completes T to be (..., 4, 4) """
    s = T.shape
    if s[-2:] == (4, 4):
        return T
    T_comp = np.zeros((*s[:-2], 4, 4), dtype=T.dtype) if isinstance(T, np.ndarray) else torch.zeros(*s[:-2], 4, 4, dtype=T.dtype, device=T.device)
    T_comp[..., :3, :s[-1]] = T
    T_comp[..., 3, 3] = 1
    return T_comp


# def _parse_scene_list_single(scene_list_path: str):
#     if scene_list_path.endswith(".json"):
#         with open(scene_list_path) as f:
#             all_scenes = json.loads(f.read())
#     elif scene_list_path.endswith(".txt"):
#         with open(scene_list_path) as f:
#             all_scenes = [p.strip() for p in f.readlines()]
#     else:
#         all_scenes = [scene_list_path]

#     return all_scenes


# def _parse_scene_list(scene_list_path: Union[str, List[str]]):
#     all_scenes = []
#     if isinstance(scene_list_path, str):
#         scene_list_path = [scene_list_path]
#     for scene_list_path_ in scene_list_path:
#         all_scenes += _parse_scene_list_single(scene_list_path_)
#     return all_scenes

def _parse_single_multi_view_scene(multi_view_dir: str, split: str, input_view_list: List[str]):
    all_scenes = []
    all_views = []
    for view in input_view_list:
        # add views from split
        # print('multi_view_dir:', multi_view_dir)
        # print('split:', split)
        # print('view:', view)
        all_views += [Path(multi_view_dir) / split / f"{view}.png"]
    all_scenes.append(all_views)
    return all_scenes


@dataclass
class MultiImageDataModuleConfig:
    # image_list: Any = ""
    multi_view_dir: Any = "" # asssume multi_view_dir contains camera json and folder of images
    split: Any = "train"
    input_view_list: List[str] = field(
        default_factory=lambda: ['0000', '0001', '0003']
    )

    background_color: Tuple[float, float, float] = field(
        default_factory=lambda: (1.0, 1.0, 1.0)
    )

    relative_pose: bool = False
    cond_height: int = 512
    cond_width: int = 512
    cond_camera_distance: float = 1.6
    cond_fovy_deg: float = 40.0
    cond_elevation_deg: float = 0.0
    cond_azimuth_deg: float = 0.0
    num_workers: int = 16

    eval_height: int = 512
    eval_width: int = 512
    eval_batch_size: int = 1
    eval_elevation_deg_lo: float = -30.0
    eval_elevation_deg_hi: float = 60.0
    eval_camera_distance: float = 1.6
    eval_fovy_deg: float = 40.0
    n_test_views: int = 168
    num_views_output: int = 168
    only_3dgs: bool = False


class MultiImageOrbitDataset(Dataset):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: MultiImageDataModuleConfig = parse_structured(MultiImageDataModuleConfig, cfg)

        # import pdb; pdb.set_trace()

        self.n_views = self.cfg.n_test_views
        assert self.n_views % self.cfg.num_views_output == 0

        self.n_imgs = len(self.cfg.input_view_list)
        # self.all_scenes = _parse_scene_list(self.cfg.image_list)
        self.all_scenes = _parse_single_multi_view_scene(self.cfg.multi_view_dir, self.cfg.split, self.cfg.input_view_list)
        # print('self.all_scenes:', self.all_scenes)
        # get image width and height
        img = Image.open(self.all_scenes[0][0])
        self.img_width, self.img_height = img.size

        # condition
        self.cameras_path = Path(self.cfg.multi_view_dir) / "camera_{}.json".format(self.cfg.split)
        # load intrinsic from camera json
        self.cameras = json.load(open(self.cameras_path))
        self.intrinsic_cond = torch.from_numpy(np.array(self.cameras['K']))
        self.intrinsic_normed_cond = self.intrinsic_cond.clone()
        self.intrinsic_normed_cond[..., 0, 2] /= self.img_width
        self.intrinsic_normed_cond[..., 1, 2] /= self.img_height
        self.intrinsic_normed_cond[..., 0, 0] /= self.img_width
        self.intrinsic_normed_cond[..., 1, 1] /= self.img_height

        if self.cfg.relative_pose:
            self.c2w_cond = torch.as_tensor(
                [
                    [0, 0, 1, self.cfg.cond_camera_distance],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                ]
            ).float()
        else:
            cond_elevation = self.cfg.cond_elevation_deg * math.pi / 180
            cond_azimuth = self.cfg.cond_azimuth_deg * math.pi / 180
            cond_camera_position: Float[Tensor, "3"] = torch.as_tensor(
                [
                    self.cfg.cond_camera_distance * np.cos(cond_elevation) * np.cos(cond_azimuth),
                    self.cfg.cond_camera_distance * np.cos(cond_elevation) * np.sin(cond_azimuth),
                    self.cfg.cond_camera_distance * np.sin(cond_elevation),
                ], dtype=torch.float32
            )

            cond_center: Float[Tensor, "3"] = torch.zeros_like(cond_camera_position)
            cond_up: Float[Tensor, "3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)
            cond_lookat: Float[Tensor, "3"] = F.normalize(cond_center - cond_camera_position, dim=-1)
            cond_right: Float[Tensor, "3"] = F.normalize(torch.cross(cond_lookat, cond_up), dim=-1)
            cond_up = F.normalize(torch.cross(cond_right, cond_lookat), dim=-1)
            cond_c2w3x4: Float[Tensor, "3 4"] = torch.cat(
                [torch.stack([cond_right, cond_up, -cond_lookat], dim=-1), cond_camera_position[:, None]],
                dim=-1,
            )
            cond_c2w: Float[Tensor, "4 4"] = torch.cat(
                [cond_c2w3x4, torch.zeros_like(cond_c2w3x4[:1])], dim=0
            )
            cond_c2w[3, 3] = 1.0
            self.c2w_cond = cond_c2w

        G_w_cc = np.array([self.cameras[self.cfg.input_view_list[0]]], dtype=np.float32)
        G_cc_w = np.linalg.inv(G_w_cc)

        # assuming the object is at the origin of the world coordinate frame
        d_w = (-G_w_cc @ complete_pts(np.zeros(3)))[:, [2]]
        S_tgs_w = complete_trans(d_w / self.cfg.cond_camera_distance * np.identity(3))
        self.S_tgs_w = torch.from_numpy(S_tgs_w).float()

        T_tgs_w = torch.tensor(G_cc_w @ S_tgs_w, dtype=torch.float32) @ torch.linalg.inv(self.c2w_cond)
        T_w_tgs = torch.linalg.inv(T_tgs_w).float()
        self.T_w_tgs = T_w_tgs.float()

        # assert check
        # print('self.c2w_cond: ', self.c2w_cond)
        # print('self.T_w_tgs @ G_cc_w @ self.S_tgs_w: ', self.T_w_tgs @ G_cc_w @ self.S_tgs_w)
        assert torch.allclose(self.T_w_tgs @ G_cc_w @ self.S_tgs_w, self.c2w_cond, atol=1e-5)

        # evaluation setup
        n = (1 + np.sqrt(1 + self.n_views * (self.cfg.eval_elevation_deg_hi - self.cfg.eval_elevation_deg_lo) / 90)) / 2
        m = round(self.n_views / n)
        n = round(n)
        # self.cfg.num_views_output is no longer respected, as self.n_views may not be dividable
        self.n_views = self.cfg.num_views_output = m * n
        azimuth_deg: Float[Tensor, "B"] = torch.linspace(0, 360.0, m + 1)[:m]
        elevation_deg = torch.linspace(self.cfg.eval_elevation_deg_lo, self.cfg.eval_elevation_deg_hi, n)
        elevation_deg, azimuth_deg = torch.meshgrid(elevation_deg, azimuth_deg, indexing='ij')
        azimuth_deg = azimuth_deg.reshape(-1)
        elevation_deg = elevation_deg.reshape(-1)

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        camera_positions_norm: Float[Tensor, "B 3"] = torch.stack(
            [
                torch.cos(elevation) * torch.cos(azimuth),
                torch.cos(elevation) * torch.sin(azimuth),
                torch.sin(elevation),
            ],
            dim=-1,
        )
        
        eval_cam_dist = d_w.mean()
        camera_positions = T_w_tgs[:, None, ...] @ complete_pts(camera_positions_norm * eval_cam_dist)
        # camera_positions = self.c2w_cond @ torch.linalg.inv(cc2ws)[:, None, ...].float() @ complete_pts(camera_positions_norm * self.cfg.eval_camera_distance)
        camera_positions = camera_positions[..., :3, 0]

        # default scene center at origin
        center: Float[Tensor, "M B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        # up: Float[Tensor, "M B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
        #     None, :
        # ].repeat(self.n_imgs, self.n_views, 1)
        up = T_w_tgs @ complete_pts(torch.tensor([0, 0, 1], dtype=torch.float32))
        up = up[:, None, :3, 0].expand(-1, self.n_views, -1)

        fovy_deg: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_fovy_deg
        )
        fovy = fovy_deg * math.pi / 180

        lookat: Float[Tensor, "M B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "M B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "M B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[..., None]],
            dim=-1,
        )
        c2w: Float[Tensor, "M B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[..., [0], :])], dim=-2
        )
        c2w[..., 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = (
            0.5 * self.cfg.eval_height / torch.tan(0.5 * fovy)
        )
        directions_unit_focal = get_ray_directions(
            H=self.cfg.eval_height,
            W=self.cfg.eval_width,
            focal=1.0,
        )
        directions: Float[Tensor, "B H W 3"] = directions_unit_focal[
            None, :, :, :
        ].repeat(self.n_views, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )
        # must use normalize=True to normalize directions here
        rays_o = []
        rays_d = []
        for cw2_ in c2w:
            rays_o_, rays_d_ = get_rays(directions, cw2_, keepdim=True)
            rays_o.append(rays_o_)
            rays_d.append(rays_d_)
        rays_o = np.array(rays_o)
        rays_d = np.array(rays_d)
        # rays_o, rays_d = get_rays(directions, c2w, keepdim=True)

        intrinsic: Float[Tensor, "B 3 3"] = get_intrinsic_from_fov(
            self.cfg.eval_fovy_deg * math.pi / 180,
            H=self.cfg.eval_height,
            W=self.cfg.eval_width,
            bs=self.n_views,
        )
        intrinsic_normed: Float[Tensor, "B 3 3"] = intrinsic.clone()
        intrinsic_normed[..., 0, 2] /= self.cfg.eval_width
        intrinsic_normed[..., 1, 2] /= self.cfg.eval_height
        intrinsic_normed[..., 0, 0] /= self.cfg.eval_width
        intrinsic_normed[..., 1, 1] /= self.cfg.eval_height

        self.rays_o, self.rays_d = rays_o, rays_d
        self.intrinsic = intrinsic
        self.intrinsic_normed = intrinsic_normed
        self.c2w = c2w
        self.camera_positions = camera_positions

        self.background_color = torch.as_tensor(self.cfg.background_color)

        # camera_positions_abs = camera_positions_norm * eval_cam_dist
        # center_abs: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions_abs)
        # # default camera up direction as +z
        # up_abs: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
        #     None, :
        # ].repeat(self.n_views, 1)

        # lookat_abs: Float[Tensor, "B 3"] = F.normalize(center_abs - camera_positions_abs, dim=-1)
        # right_abs: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat_abs, up_abs), dim=-1)
        # up_abs = F.normalize(torch.cross(right_abs, lookat_abs), dim=-1)
        # c2w3x4_abs: Float[Tensor, "B 3 4"] = torch.cat(
        #     [torch.stack([right_abs, up_abs, -lookat_abs], dim=-1), camera_positions_abs[..., None]],
        #     dim=-1,
        # )
        # c2w_abs: Float[Tensor, "B 4 4"] = torch.cat(
        #     [c2w3x4_abs, torch.zeros_like(c2w3x4_abs[..., [0], :])], dim=-2
        # )
        # c2w_abs[..., 3, 3] = 1.0

    def __len__(self):
        if self.cfg.only_3dgs:
            return len(self.all_scenes)
        else:
            return len(self.all_scenes) * self.n_views // self.cfg.num_views_output

    def __getitem__(self, index):
        if self.cfg.only_3dgs:
            scene_index = index
            view_index = [0]
        else:
            scene_index = index * self.cfg.num_views_output // self.n_views
            view_start = index % (self.n_views // self.cfg.num_views_output)
            view_index = list(range(self.n_views))[view_start * self.cfg.num_views_output:
                                                   (view_start + 1) * self.cfg.num_views_output]
            
        def _get_view_data(img_path):
            img_cond = torch.from_numpy(
                np.asarray(
                    Image.fromarray(imageio.v2.imread(img_path))
                    .convert("RGBA")
                    .resize((self.cfg.cond_width, self.cfg.cond_height))
                )
                / 255.0
            ).float()
            mask_cond: Float[Tensor, "Hc Wc 1"] = img_cond[:, :, -1:]
            rgb_cond: Float[Tensor, "Hc Wc 3"] = img_cond[
                :, :, :3
            ] * mask_cond + self.background_color[None, None, :] * (1 - mask_cond)

            img_index = os.path.split(img_path)[-1].split('.')[0]

            G_w_cc = np.array(self.cameras[img_index], dtype=np.float32)
            G_cc_w = np.linalg.inv(G_w_cc)
            G_cc_w = torch.from_numpy(G_cc_w).float()
            G_cc_tgs = self.T_w_tgs @ G_cc_w @ self.S_tgs_w

            return rgb_cond, mask_cond, G_cc_tgs

        img_paths = self.all_scenes[scene_index]
        rgbs_cond, masks_cond, poses_cond = zip(*[_get_view_data(p) for p in img_paths])
        rgbs_cond = torch.stack(rgbs_cond, dim=0) # [3, 512, 512, 3]
        masks_cond = torch.stack(masks_cond, dim=0)
        poses_cond = torch.cat(poses_cond, dim=0) # [3, 4, 4]

        rgb_cond = rgbs_cond[0]
        mask_cond = masks_cond[0]

        # print('self.c2w_cond:', self.c2w_cond)
        # print('poses_cond:', poses_cond)

        out = {
            "rgb_cond": rgb_cond.unsqueeze(0),
            "c2w_cond": self.c2w_cond.unsqueeze(0),
            "mask_cond": mask_cond.unsqueeze(0),
            "rgbs_cond": rgbs_cond,
            "masks_cond": masks_cond,
            "poses_cond": poses_cond,
            "intrinsic_cond": self.intrinsic_cond.unsqueeze(0),
            "intrinsic_normed_cond": self.intrinsic_normed_cond.unsqueeze(0),
            "view_index": torch.as_tensor(view_index),
            "rays_o": self.rays_o[scene_index][view_index],
            "rays_d": self.rays_d[scene_index][view_index],
            "intrinsic": self.intrinsic[view_index],
            "intrinsic_normed": self.intrinsic_normed[view_index],
            "c2w": self.c2w[scene_index][view_index],
            "camera_positions": self.camera_positions[scene_index][view_index],
        }
        out["c2w"][..., :3, 1:3] *= -1
        out["c2w_cond"][..., :3, 1:3] *= -1
        instance_id = os.path.split(img_paths[0])[-1].split('.')[0]
        out["index"] = torch.as_tensor(scene_index)
        out["background_color"] = self.background_color
        out["instance_id"] = instance_id
        return out

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch
    

if __name__ == "__main__":
    cfg = {
        "multi_view_dir": "/datasets/paris/load/sapien/USB/100109/start",
        "split": "train",
        "input_view_list": ['0000', '0001', '0003'],
    }

    # MultiImageDataModuleConfig(**cfg)

    dataset = MultiImageOrbitDataset(cfg)
    sample_data = dataset[0]
    # print('sample_data:', sample_data)
    print('sample_data.keys:', sample_data.keys())
    print('sample_data["rgb_cond"].shape:', sample_data["rgb_cond"].shape)
    print('sample_data["c2w_cond"].shape:', sample_data["c2w_cond"].shape)
    print('sample_data["mask_cond"].shape:', sample_data["mask_cond"].shape)
    print('sample_data["rgbs_cond"].shape:', sample_data["rgbs_cond"].shape)
    print('sample_data["masks_cond"].shape:', sample_data["masks_cond"].shape)
    print('sample_data["poses_cond"].shape:', sample_data["poses_cond"].shape)
    print('sample_data["intrinsic_cond"].shape:', sample_data["intrinsic_cond"].shape)
    print('sample_data["intrinsic_normed_cond"].shape:', sample_data["intrinsic_normed_cond"].shape)

    print('sample_data["c2w_cond"]:', sample_data["c2w_cond"])
    print('sample_data["poses_cond"]:', sample_data["poses_cond"])

