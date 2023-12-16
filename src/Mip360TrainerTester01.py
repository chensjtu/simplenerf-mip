# Shree KRISHNAya Namaha
# Runs both training and testing on NeRF-LLFF dataset.
# Authors: Nagabhushan S N, Adithyan K V
# Last Modified: 15/06/2023

import datetime
import os
import time
import traceback
from pathlib import Path

import numpy
import pandas
import skimage.io
import skvideo.io

import Tester01 as Tester
import Trainer01 as Trainer

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def read_image(path: Path):
    image = skimage.io.imread(path.as_posix())
    return image


def save_video(path: Path, video: numpy.ndarray):
    if path.exists():
        return
    try:
        skvideo.io.vwrite(path.as_posix(), video,
                          inputdict={'-r': str(15)},
                          outputdict={'-c:v': 'libx264', '-pix_fmt': 'yuv420p'}, verbosity=1)
    except (OSError, NameError):
        pass
    return


def start_training(train_configs: dict):
    root_dirpath = Path('../')
    database_dirpath = root_dirpath / train_configs['database_dirpath']

    # Setup output dirpath
    output_dirpath = root_dirpath / f'runs/training/train{train_configs["train_num"]:04}'
    output_dirpath.mkdir(parents=True, exist_ok=True)
    scene_names = train_configs['data_loader'].get('scene_names', None)
    Trainer.save_configs(output_dirpath, train_configs)
    train_configs['data_loader']['scene_names'] = scene_names

    if train_configs['data_loader']['scene_names'] is None:
        set_num = train_configs['data_loader']['train_set_num']
        video_datapath = database_dirpath / f'train_test_sets/set{set_num:02}/TrainVideosData.csv'
        video_data = pandas.read_csv(video_datapath)
        scene_names = video_data['scene_name'].to_numpy()
    scene_ids = numpy.unique(scene_names)
    train_configs['data_loader']['scene_ids'] = scene_ids
    Trainer.start_training(train_configs)
    return


def start_testing(test_configs: dict):
    root_dirpath = Path('../')
    database_dirpath = root_dirpath / test_configs['database_dirpath']

    output_dirpath = root_dirpath / f"runs/testing/test{test_configs['test_num']:04}"
    output_dirpath.mkdir(parents=True, exist_ok=True)
    Tester.save_configs(output_dirpath, test_configs)

    set_num = test_configs['test_set_num']
    train_video_datapath = database_dirpath / f'train_test_sets/set{set_num:02}/TrainVideosData.csv'
    test_video_datapath = database_dirpath / f'train_test_sets/set{set_num:02}/TestVideosData.csv'
    train_video_data = pandas.read_csv(train_video_datapath)
    test_video_data = pandas.read_csv(test_video_datapath)
    scene_names = test_configs.get('scene_names', test_video_data['scene_name'].to_numpy())
    scene_names = numpy.unique(scene_names)
    scenes_data = {}
    for scene_name in scene_names:
        scene_id = scene_name
        scenes_data[scene_id] = {
            'output_dirname': scene_id,
            'frames_data': {}
        }

        extrinsics_path = database_dirpath / f'all/database_data/{scene_id}/CameraExtrinsics.csv'
        extrinsics = numpy.loadtxt(extrinsics_path.as_posix(), delimiter=',').reshape((-1, 4, 4))
        # Intrinsics and frames required to compute plane sweep volume for conv visibility prediction
        intrinsics_path = database_dirpath / f'all/database_data/{scene_id}/CameraIntrinsics{test_configs["resolution_suffix"]}.csv'
        intrinsics = numpy.loadtxt(intrinsics_path.as_posix(), delimiter=',').reshape((-1, 3, 3))

        test_frame_nums = test_video_data.loc[test_video_data['scene_name'] == scene_name]['pred_frame_num'].to_list()
        train_frame_nums = train_video_data.loc[train_video_data['scene_name'] == scene_name]['pred_frame_num'].to_list()
        frame_nums = numpy.unique(sorted([test_frame_nums + train_frame_nums]))
        for frame_num in frame_nums:
            scenes_data[scene_id]['frames_data'][frame_num] = {
                'extrinsic': extrinsics[frame_num],
                'intrinsic': intrinsics[frame_num],
                'is_train_frame': frame_num in train_frame_nums,
            }
    Tester.start_testing(test_configs, scenes_data, save_depth=True, save_depth_var=True, save_visibility=True)

    # Run QA
    # qa_filepath = Path('./qa/00_Common/src/AllMetrics02_NeRF_LLFF.py')
    # gt_depth_dirpath = Path('../data/DenseNeRF/runs/testing/test2001')
    # cmd = f'python {qa_filepath.absolute().as_posix()} ' \
    #       f'--demo_function_name demo2 ' \
    #       f'--pred_videos_dirpath {output_dirpath.absolute().as_posix()} ' \
    #       f'--database_dirpath {database_dirpath.absolute().as_posix()} ' \
    #       f'--gt_depth_dirpath {gt_depth_dirpath.absolute().as_posix()} ' \
    #       f'--frames_datapath {test_video_datapath.absolute().as_posix()} ' \
    #       f'--pred_frames_dirname predicted_frames ' \
    #       f'--pred_depths_dirname predicted_depths ' \
    #       f'--mask_folder_name {test_configs["qa_masks_dirname"]} '\
    #       f'--resolution_suffix _down4'
    # os.system(cmd)
    return


def start_testing_videos(test_configs: dict):
    root_dirpath = Path('../')
    database_dirpath = root_dirpath / test_configs['database_dirpath']

    output_dirpath = root_dirpath / f"runs/testing/test{test_configs['test_num']:04}"
    output_dirpath.mkdir(parents=True, exist_ok=True)
    Tester.save_configs(output_dirpath, test_configs)

    set_num = test_configs['test_set_num']
    video_datapath = database_dirpath / f'train_test_sets/set{set_num:02}/TestVideosData.csv'
    video_data = pandas.read_csv(video_datapath)
    scene_names = test_configs.get('scene_names', video_data['scene_name'].to_numpy())
    scene_names = numpy.unique(scene_names)

    videos_data = [1, ]
    for video_num in videos_data:
        video_frame_nums_path = database_dirpath / f'train_test_sets/set{set_num:02}/video_poses{video_num:02}/VideoFrameNums.csv'
        if video_frame_nums_path.exists():
            video_frame_nums = numpy.loadtxt(video_frame_nums_path.as_posix(), delimiter=',').astype(int)
        else:
            video_frame_nums = None
        for scene_name in scene_names:
            scenes_data = {}
            scene_id = scene_name
            scenes_data[scene_id] = {
                'output_dirname': scene_id,
                'frames_data': {}
            }

            extrinsics_path = database_dirpath / f'train_test_sets/set{set_num:02}/video_poses{video_num:02}/{scene_id}.csv'
            if not extrinsics_path.exists():
                continue
            extrinsics = numpy.loadtxt(extrinsics_path.as_posix(), delimiter=',').reshape((-1, 4, 4))

            frame_nums = numpy.arange(extrinsics.shape[0] - 1)
            for frame_num in frame_nums:
                scenes_data[scene_id]['frames_data'][frame_num] = {
                    'extrinsic': extrinsics[frame_num + 1]
                }
            output_dir_suffix = f'_video{video_num:02}'
            output_dirpath = Tester.start_testing(test_configs, scenes_data, output_dir_suffix)
            scene_output_dirpath = output_dirpath / f'{scene_id}{output_dir_suffix}'
            if not scene_output_dirpath.exists():
                continue
            pred_frames = [read_image(scene_output_dirpath / f'predicted_frames/{frame_num:04}.png') for frame_num in frame_nums]
            video_frames = numpy.stack(pred_frames)
            if video_frame_nums is not None:
                video_frames = video_frames[video_frame_nums]
            video_output_path = scene_output_dirpath / 'PredictedVideo.mp4'
            save_video(video_output_path, video_frames)
    return


def start_testing_static_videos(test_configs: dict):
    """
    This is for view_dirs visualization
    :param test_configs:
    :return:
    """
    root_dirpath = Path('../')
    database_dirpath = root_dirpath / test_configs['database_dirpath']

    output_dirpath = root_dirpath / f"runs/testing/test{test_configs['test_num']:04}"
    output_dirpath.mkdir(parents=True, exist_ok=True)
    Tester.save_configs(output_dirpath, test_configs)

    set_num = test_configs['test_set_num']
    video_datapath = database_dirpath / f'train_test_sets/set{set_num:02}/TestVideosData.csv'
    video_data = pandas.read_csv(video_datapath)
    scene_names = test_configs.get('scene_names', video_data['scene_name'].to_numpy())
    scene_names = numpy.unique(scene_names)

    videos_data = [1, ]
    for video_num in videos_data:
        video_frame_nums_path = database_dirpath / f'train_test_sets/set{set_num:02}/video_poses{video_num:02}/VideoFrameNums.csv'
        if video_frame_nums_path.exists():
            video_frame_nums = numpy.loadtxt(video_frame_nums_path.as_posix(), delimiter=',').astype(int)
        else:
            video_frame_nums = None
        for scene_name in scene_names:
            scenes_data = {}
            scene_id = scene_name
            scenes_data[scene_id] = {
                'output_dirname': scene_id,
                'frames_data': {}
            }

            extrinsics_path = database_dirpath / f'train_test_sets/set{set_num:02}/video_poses{video_num:02}/{scene_id}.csv'
            if not extrinsics_path.exists():
                continue
            extrinsics = numpy.loadtxt(extrinsics_path.as_posix(), delimiter=',').reshape((-1, 4, 4))

            frame_nums = numpy.arange(extrinsics.shape[0] - 1)
            for frame_num in frame_nums:
                scenes_data[scene_id]['frames_data'][frame_num] = {
                    'extrinsic': extrinsics[0],
                    'extrinsic_viewcam': extrinsics[frame_num + 1],
                }
            output_dir_suffix = f'_video{video_num:02}_static_camera'
            output_dirpath = Tester.start_testing(test_configs, scenes_data, output_dir_suffix)
            scene_output_dirpath = output_dirpath / f'{scene_id}{output_dir_suffix}'
            if not scene_output_dirpath.exists():
                continue
            pred_frames = [read_image(scene_output_dirpath / f'predicted_frames/{frame_num:04}.png') for frame_num in frame_nums]
            video_frames = numpy.stack(pred_frames)
            if video_frame_nums is not None:
                video_frames = video_frames[video_frame_nums]
            video_output_path = scene_output_dirpath / 'StaticCameraVideo.mp4'
            save_video(video_output_path, video_frames)
    return


def demo1():
    train_num = 2011
    test_num = 2011
    scene_names = ['kitchen', 'garden']

    for scene_name in scene_names:
        train_configs = {
            'trainer': f'{this_filename}/{Trainer.this_filename}',
            'train_num': train_num,
            'description': 'Main Model: 5 views',
            'database': 'NeRF_LLFF',
            'database_dirpath': 'sparse_nerf_datasets/mip360_v2_vip_style',
            'data_loader': {
                'data_loader_name': 'NerfLlffDataLoader01',
                'data_preprocessor_name': 'DataPreprocessor01',
                'train_set_num': 1,
                'scene_names': [scene_name],
                'resolution_suffix': '_down4',
                'recenter_camera_poses': True,
                'bd_factor': 0.75,
                'spherify': True,
                'ndc': False,
                'batching': True,
                'downsampling_factor': 1,
                'num_rays': 2048,
                'precrop_fraction': 1,
                'precrop_iterations': -1,
                'sparse_depth': {
                    'dirname': 'DE01',
                    'num_rays': 2048,
                },
            },
            'model': {
                'name': 'SimpleNeRF01',
                'coarse_mlp': {
                    'num_samples': 64,
                    'points_net_depth': 8,
                    'views_net_depth': 1,
                    'points_net_width': 256,
                    'views_net_width': 128,
                    'points_positional_encoding_degree': 10,
                    'views_positional_encoding_degree': 4,
                    'use_view_dirs': True,
                    'view_dependent_rgb': True,
                    'predict_visibility': False,
                },
                'fine_mlp': {
                    'num_samples': 128,
                    'points_net_depth': 8,
                    'views_net_depth': 1,
                    'points_net_width': 256,
                    'views_net_width': 128,
                    'points_positional_encoding_degree': 10,
                    'views_positional_encoding_degree': 4,
                    'use_view_dirs': True,
                    'view_dependent_rgb': True,
                    'predict_visibility': False,
                },
                'points_augmentation': {
                    'coarse_mlp': {
                        'points_net_depth': 8,
                        'views_net_depth': 1,
                        'points_net_width': 256,
                        'views_net_width': 128,
                        'points_positional_encoding_degree': 10,
                        'points_sigma_positional_encoding_degree': 3,
                        'views_positional_encoding_degree': 4,
                        'use_view_dirs': True,
                        'view_dependent_rgb': True,
                        'predict_visibility': False,
                    }
                },
                'views_augmentation': {
                    'coarse_mlp': {
                        'points_net_depth': 8,
                        'views_net_depth': 1,
                        'points_net_width': 256,
                        'views_net_width': 128,
                        'points_positional_encoding_degree': 10,
                        'use_view_dirs': False,
                        'view_dependent_rgb': False,
                        'predict_visibility': False,
                    }
                },
                'chunk': 4*1024,
                'lindisp': False,
                'netchunk': 16*1024,
                'perturb': True,
                'raw_noise_std': 1.0,
                'white_bkgd': False,
            },
            'losses': [
                {
                    'name': 'MSE01',
                    'weight': 1,
                },
                {
                    'name': 'SparseDepthMSE01',
                    'weight': 0.1,
                },
                {
                    'name': 'MSE02',
                    'weight': 1,
                },
                {
                    'name': 'SparseDepthMSE02',
                    'weight': 0.1,
                },
                {
                    'name': 'MSE03',
                    'weight': 1,
                },
                {
                    'name': 'SparseDepthMSE03',
                    'weight': 0.1,
                },
                {
                    'name': 'PointsAugmentationDepthLoss02',
                    'iter_weights': {
                        '0': 0, '10000': 0.1
                    },
                    'rmse_threshold': 0.1,
                    'patch_size': [5, 5],
                },
                {
                    'name': 'ViewsAugmentationDepthLoss02',
                    'iter_weights': {
                        '0': 0, '10000': 0.1
                    },
                    'rmse_threshold': 0.1,
                    'patch_size': [5, 5],
                },
                {
                    'name': 'CoarseFineConsistencyLoss02',
                    'iter_weights': {
                        '0': 0, '10000': 0.1
                    },
                    'rmse_threshold': 0.1,
                    'patch_size': [5, 5],
                },
            ],
            'optimizer': {
                'lr_decayer_name': 'NeRFLearningRateDecayer01',
                'lr_initial': 5e-4,
                'lr_decay': 250,
                'beta1': 0.9,
                'beta2': 0.999,
            },
            'resume_training': True,
            'sub_batch_size': 2048,
            'num_iterations': 100000,
            'validation_interval': 1000000,
            'validation_chunk_size': 64 * 1024,
            'validation_save_loss_maps': False,
            'model_save_interval': 10000,
            'mixed_precision_training': False,
            'seed': numpy.random.randint(1000),
            'device': [2, 3],
        }
        test_configs = {
            'Tester': f'{this_filename}/{Tester.this_filename}',
            'test_num': test_num,
            'test_set_num': 1,
            'train_num': train_num,
            'model_name': 'Model_Iter100000.tar',
            'database_name': 'NeRF_LLFF',
            'database_dirpath': 'sparse_nerf_datasets/mip360_v2_vip_style',
            'qa_masks_dirname': 'VM01',
            'resolution_suffix': train_configs['data_loader']['resolution_suffix'],
            'scene_names': [scene_name],
            'device': [2, 3],
        }
        start_training(train_configs)
        start_testing(test_configs)
        start_testing_videos(test_configs)
        start_testing_static_videos(test_configs)
    return


def demo2():
    train_num = 2021
    test_num = 2021
    scene_names = ['kitchen', 'garden']

    for scene_name in scene_names:
        train_configs = {
            'trainer': f'{this_filename}/{Trainer.this_filename}',
            'train_num': train_num,
            'description': 'Main Model: 4 views',
            'database': 'NeRF_LLFF',
            'database_dirpath': 'sparse_nerf_datasets/mip360_v2_vip_style',
            'data_loader': {
                'data_loader_name': 'NerfLlffDataLoader01',
                'data_preprocessor_name': 'DataPreprocessor01',
                'train_set_num': 1,
                'scene_names': [scene_name],
                'resolution_suffix': '_down4',
                'recenter_camera_poses': True,
                'bd_factor': 0.75,
                'spherify': True,
                'ndc': False,
                'batching': True,
                'downsampling_factor': 1,
                'num_rays': 2048,
                'precrop_fraction': 1,
                'precrop_iterations': -1,
                'sparse_depth': {
                    'dirname': 'DE01',
                    'num_rays': 2048,
                },
            },
            'model': {
                'name': 'SimpleNeRF01',
                'coarse_mlp': {
                    'num_samples': 64,
                    'points_net_depth': 8,
                    'views_net_depth': 1,
                    'points_net_width': 256,
                    'views_net_width': 128,
                    'points_positional_encoding_degree': 10,
                    'views_positional_encoding_degree': 4,
                    'use_view_dirs': True,
                    'view_dependent_rgb': True,
                    'predict_visibility': False,
                },
                'fine_mlp': {
                    'num_samples': 128,
                    'points_net_depth': 8,
                    'views_net_depth': 1,
                    'points_net_width': 256,
                    'views_net_width': 128,
                    'points_positional_encoding_degree': 10,
                    'views_positional_encoding_degree': 4,
                    'use_view_dirs': True,
                    'view_dependent_rgb': True,
                    'predict_visibility': False,
                },
                'points_augmentation': {
                    'coarse_mlp': {
                        'points_net_depth': 8,
                        'views_net_depth': 1,
                        'points_net_width': 256,
                        'views_net_width': 128,
                        'points_positional_encoding_degree': 10,
                        'points_sigma_positional_encoding_degree': 3,
                        'views_positional_encoding_degree': 4,
                        'use_view_dirs': True,
                        'view_dependent_rgb': True,
                        'predict_visibility': False,
                    }
                },
                'views_augmentation': {
                    'coarse_mlp': {
                        'points_net_depth': 8,
                        'views_net_depth': 1,
                        'points_net_width': 256,
                        'views_net_width': 128,
                        'points_positional_encoding_degree': 10,
                        'use_view_dirs': False,
                        'view_dependent_rgb': False,
                        'predict_visibility': False,
                    }
                },
                'chunk': 4*1024,
                'lindisp': False,
                'netchunk': 16*1024,
                'perturb': True,
                'raw_noise_std': 1.0,
                'white_bkgd': False,
            },
            'losses': [
                {
                    'name': 'MSE01',
                    'weight': 1,
                },
                {
                    'name': 'SparseDepthMSE01',
                    'weight': 0.1,
                },
                {
                    'name': 'MSE02',
                    'weight': 1,
                },
                {
                    'name': 'SparseDepthMSE02',
                    'weight': 0.1,
                },
                {
                    'name': 'MSE03',
                    'weight': 1,
                },
                {
                    'name': 'SparseDepthMSE03',
                    'weight': 0.1,
                },
                {
                    'name': 'PointsAugmentationDepthLoss02',
                    'iter_weights': {
                        '0': 0, '10000': 0.1
                    },
                    'rmse_threshold': 0.1,
                    'patch_size': [5, 5],
                },
                {
                    'name': 'ViewsAugmentationDepthLoss02',
                    'iter_weights': {
                        '0': 0, '10000': 0.1
                    },
                    'rmse_threshold': 0.1,
                    'patch_size': [5, 5],
                },
                {
                    'name': 'CoarseFineConsistencyLoss02',
                    'iter_weights': {
                        '0': 0, '10000': 0.1
                    },
                    'rmse_threshold': 0.1,
                    'patch_size': [5, 5],
                },
            ],
            'optimizer': {
                'lr_decayer_name': 'NeRFLearningRateDecayer01',
                'lr_initial': 5e-4,
                'lr_decay': 250,
                'beta1': 0.9,
                'beta2': 0.999,
            },
            'resume_training': True,
            'sub_batch_size': 2048,
            'num_iterations': 100000,
            'validation_interval': 1000000,
            'validation_chunk_size': 64 * 1024,
            'validation_save_loss_maps': False,
            'model_save_interval': 10000,
            'mixed_precision_training': False,
            'seed': numpy.random.randint(1000),
            'device': [2, 3],
        }
        test_configs = {
            'Tester': f'{this_filename}/{Tester.this_filename}',
            'test_num': test_num,
            'test_set_num': 2,
            'train_num': train_num,
            'model_name': 'Model_Iter100000.tar',
            'database_name': 'NeRF_LLFF',
            'database_dirpath': 'sparse_nerf_datasets/mip360_v2_vip_style',
            'qa_masks_dirname': 'VM02',
            'resolution_suffix': train_configs['data_loader']['resolution_suffix'],
            'scene_names': [scene_name],
            'device': [2, 3],
        }
        start_training(train_configs)
        start_testing(test_configs)
        start_testing_videos(test_configs)
        start_testing_static_videos(test_configs)
    return


def main():
    demo1()
    demo2()
    return


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = 'Error: ' + str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
