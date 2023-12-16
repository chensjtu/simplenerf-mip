# Shree KRISHNAya Namaha
# Predicts sparse depth from the given images
# Authors: Nagabhushan S N, Sai Harsha Mupparaju
# Last Modified: 26/09/2023

import os
import shutil
from pathlib import Path
from typing import List, Optional

import numpy
import pandas
import skimage.io
from scipy.spatial.transform import Rotation

import llff.poses.colmap_read_model as read_model
from colmapUtils.read_write_model import read_images_binary, read_points3d_binary
from database import COLMAPDatabase, array_to_blob

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class ColmapTester:
    def __init__(self, database_dirpath: Path, selected_ids: List[int], frame_nums: numpy.ndarray):
        self.database_dirpath = database_dirpath
        self.sparse_dirpath = self.database_dirpath / 'sparse/0'
        self.selected_ids = selected_ids
        self.frame_nums = frame_nums
        return

    @staticmethod
    def get_quaternions_and_translations(trans_mat: numpy.ndarray):
        rot_mat = trans_mat[:3, :3]
        rotation = Rotation.from_matrix(rot_mat)
        assert type(rotation) == Rotation
        quaternions = rotation.as_quat()
        quaternions = numpy.roll(quaternions, 1)
        quaternions_str = ' '.join(quaternions.astype('str'))
        translations = trans_mat[:3, 3]
        translations_str = ' '.join(translations.astype('str'))
        return quaternions_str, translations_str

    @staticmethod
    def read_image(path: Path) -> numpy.ndarray:
        image = skimage.io.imread(path.as_posix())
        return image

    @staticmethod
    def save_image(path: Path, image: numpy.ndarray):
        path.parent.mkdir(parents=True, exist_ok=True)
        skimage.io.imsave(path.as_posix(), image)
        return

    def load_colmap_data(self):
        camdata = read_model.read_cameras_binary((self.sparse_dirpath / 'cameras.bin').as_posix())
    
        list_of_keys = list(camdata.keys())
        cam = camdata[list_of_keys[0]]
        print( 'Cameras', len(cam))
    
        h, w, f = cam.height, cam.width, cam.params[0]
        hwf = numpy.array([h,w,f]).reshape([3,1])

        imdata = read_model.read_images_binary((self.sparse_dirpath / 'images.bin'), self.frame_nums)
        
        w2c_mats = []
        bottom = numpy.array([0,0,0,1.]).reshape([1,4])
        
        names = [imdata[k].name for k in imdata]
        perm = numpy.argsort(names)
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat()
            t = im.tvec.reshape([3,1])
            m = numpy.concatenate([numpy.concatenate([R, t], 1), bottom], 0)
            w2c_mats.append(m)
        
        w2c_mats = numpy.stack(w2c_mats, 0)
        c2w_mats = numpy.linalg.inv(w2c_mats)
        
        poses = c2w_mats[:, :3, :4].transpose([1,2,0])
        poses = numpy.concatenate([poses, numpy.tile(hwf[..., numpy.newaxis], [1,1,poses.shape[-1]])], 1)

        pts3d = read_model.read_points3d_binary((self.sparse_dirpath / 'points3D.bin'), self.selected_ids)

        # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
        poses = numpy.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
        
        return poses, pts3d, perm

    def get_bounds(self, poses, pts3d, perm):
        pts_arr = []
        vis_arr = []
        print('pts3d', len(pts3d))
        for k in pts3d:
            pts_arr.append(pts3d[k].xyz)
            cams = [0] * poses.shape[-1]
            for ind in pts3d[k].image_ids:
                # if len(cams) < ind - 1:
                #     print('ERROR: the correct camera poses for current points cannot be accessed')
                #     return
                for frame_num, image_id in enumerate(self.frame_nums):
                    if ind == image_id + 1:
                        cams[frame_num] = 1
            vis_arr.append(cams)

        if len(pts_arr) == 0:
            return None

        pts_arr = numpy.array(pts_arr)
        vis_arr = numpy.array(vis_arr)
        print( 'Points', pts_arr.shape, 'Visibility', vis_arr.shape )
        # Points (769, 3) Visibility (769, 2)
        
        zvals = numpy.sum(-(pts_arr[:, numpy.newaxis, :].transpose([2,0,1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
        valid_z = zvals[vis_arr==1]
        print( 'Depth stats', valid_z.min(), valid_z.max(), valid_z.mean() )
        
        bounds = []
        for i in perm:
            vis = vis_arr[:, i]
            zs = zvals[:, i]
            zs = zs[vis==1]
            if zs.size == 0:
                return None
            close_depth, inf_depth = numpy.percentile(zs, .5), numpy.percentile(zs, 99.5)
            print( i, close_depth, inf_depth )
            
            bounds.append([close_depth, inf_depth])
        bounds = numpy.array(bounds).astype(numpy.float32)
        return bounds

    @staticmethod
    def get_poses(images):
        poses = []
        for i in images:
            R = images[i].qvec2rotmat()
            t = images[i].tvec.reshape([3,1])
            bottom = numpy.array([0,0,0,1.]).reshape([1,4])
            w2c = numpy.concatenate([numpy.concatenate([R, t], 1), bottom], 0)
            c2w = numpy.linalg.inv(w2c)
            poses.append(c2w)
        return numpy.array(poses)

    def compute_colmap_depth(self):
        if not (self.sparse_dirpath / 'images.bin').exists():
            return None, None

        images = read_images_binary((self.sparse_dirpath / 'images.bin').as_posix(), self.frame_nums)
        points = read_points3d_binary((self.sparse_dirpath / 'points3D.bin').as_posix(), self.selected_ids)

        Errs = numpy.array([point3D.error for point3D in points.values()])
        Err_mean = numpy.mean(Errs)
        print("Mean Projection Error:", Err_mean)
    
        poses = self.get_poses(images)
        colmap_data = self.load_colmap_data()
        bds_raw = self.get_bounds(*colmap_data)
        print('bds_raw:', bds_raw)
        if bds_raw is None:
            return None, None
    
        data_list = []
        for id_im in range(1, len(images)+1):
            depth_list = []
            coord_list = []
            error_list = []
            weight_list = []
            image_id = self.frame_nums[id_im - 1] + 1
            for i in range(len(images[image_id].xys)):
                point2D = images[image_id].xys[i]
                id_3D = images[image_id].point3D_ids[i]
                if id_3D == -1:
                    continue
                if id_3D not in points.keys():
                    continue
                point3D = points[id_3D].xyz
                depth = (poses[id_im-1,:3,2].T @ (point3D - poses[id_im-1,:3,3]))
                if depth < bds_raw[id_im-1,0] or depth > bds_raw[id_im-1,1]:
                    continue
                err = points[id_3D].error
                weight = 2 * numpy.exp(-(err/Err_mean)**2)
                depth_list.append(depth)
                coord_list.append(point2D)
                error_list.append(err)
                weight_list.append(weight)
            if len(depth_list) > 0:
                print(id_im, len(depth_list), numpy.min(depth_list), numpy.max(depth_list), numpy.mean(depth_list))
                data_list.append({
                    "depth": numpy.array(depth_list),
                    "coord": numpy.array(coord_list),
                    "error": numpy.array(error_list),
                    "weight": numpy.array(weight_list),
                })
            else:
                print(image_id, len(depth_list))

        depth_data_list = []
        for i in range(len(data_list)):
            depth_data_array = numpy.concatenate([data_list[i]['coord'], data_list[i]['depth'][:, None], data_list[i]['error'][:, None], data_list[i]['weight'][:, None]], axis=1)
            depth_data = pandas.DataFrame(depth_data_array, columns=['x', 'y', 'depth', 'reprojection_error', 'weight'])
            depth_data_list.append(depth_data)

        bounds_data = pandas.DataFrame(bds_raw, columns=['near', 'far'])

        return depth_data_list, bounds_data
    
    def estimate_sparse_depth(self):
        depth_data, bounds_data = self.compute_colmap_depth()
        return depth_data, bounds_data
