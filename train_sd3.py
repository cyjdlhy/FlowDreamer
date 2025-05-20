# -*- coding: utf-8 -*-
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import random
import imageio
import os
import torch
import torch.nn as nn
from random import randint
from utils.loss_utils import l1_loss, ssim, tv_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, GenerateCamParams, GuidanceParams
import math
from torchvision.utils import save_image
import torchvision.transforms as T

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
 

# try:
#     from torch.utils.tensorboard import SummaryWriter
#     TENSORBOARD_FOUND = True
# except ImportError:
#     TENSORBOARD_FOUND = False

TENSORBOARD_FOUND = False

def adjust_text_embeddings(embeddings, azimuth, guidance_opt):
    #TODO: add prenerg functions
    text_z_list = []
    weights_list = []
    K = 0
    #for b in range(azimuth):

    text_z_, weights_ = get_pos_neg_text_embeddings(embeddings, azimuth, guidance_opt) 
    K = max(K, weights_.shape[0])
    text_z_list.append(text_z_)
    weights_list.append(weights_)

    # Interleave text_embeddings from different dirs to form a batch
    text_embeddings_embed = []
    text_embeddings_pool = []
    for i in range(K):
        for text_z in text_z_list:
            # if uneven length, pad with the first embedding
            text_embeddings_embed.append(text_z[0][i] if i < len(text_z[0]) else text_z[0][0])
            text_embeddings_pool.append(text_z[1][i] if i < len(text_z[1]) else text_z[1][0])
    text_embeddings_embed = torch.stack(text_embeddings_embed, dim=0) # [B * K, 77, 768]
    text_embeddings_pool = torch.stack(text_embeddings_pool, dim=0) # [B * K, 2048]
    text_embeddings = (text_embeddings_embed, text_embeddings_pool)

    # Interleave weights from different dirs to form a batch
    weights = []
    for i in range(K):
        for weights_ in weights_list:
            weights.append(weights_[i] if i < len(weights_) else torch.zeros_like(weights_[0]))
    weights = torch.stack(weights, dim=0) # [B * K]
    return text_embeddings, weights

def get_pos_neg_text_embeddings(embeddings, azimuth_val, opt):
    if azimuth_val >= -90 and azimuth_val < 90:
        if azimuth_val >= 0:
            r = 1 - azimuth_val / 90
        else:
            r = 1 + azimuth_val / 90
        start_z = embeddings['front']
        end_z = embeddings['side']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z_embed = r * start_z[0] + (1 - r) * end_z[0]
        pos_z_pool = r * start_z[1] + (1 - r) * end_z[1]

        text_z_embed = torch.cat([pos_z_embed, embeddings['front'][0], embeddings['side'][0]], dim=0)
        text_z_pool = torch.cat([pos_z_pool, embeddings['front'][1], embeddings['side'][1]], dim=0)
        text_z = (text_z_embed, text_z_pool)

        if r > 0.8:
            front_neg_w = 0.0 
        else:
            front_neg_w = math.exp(-r * opt.front_decay_factor) * opt.negative_w
        if r < 0.2:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-(1-r) * opt.side_decay_factor) * opt.negative_w

        weights = torch.tensor([1.0, front_neg_w, side_neg_w])
    else:
        if azimuth_val >= 0:
            r = 1 - (azimuth_val - 90) / 90
        else:
            r = 1 + (azimuth_val + 90) / 90
        start_z = embeddings['side']
        end_z = embeddings['back']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z_embed = r * start_z[0] + (1 - r) * end_z[0]
        pos_z_pool = r * start_z[1] + (1 - r) * end_z[1]

        text_z_embed = torch.cat([pos_z_embed, embeddings['side'][0], embeddings['front'][0]], dim=0)
        text_z_pool = torch.cat([pos_z_pool, embeddings['side'][1], embeddings['front'][1]], dim=0)

        text_z = (text_z_embed, text_z_pool)
        front_neg_w = opt.negative_w 
        if r > 0.8:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-r * opt.side_decay_factor) * opt.negative_w / 2

        weights = torch.tensor([1.0, side_neg_w, front_neg_w])
    return text_z, weights.to(text_z_embed.device)

def prepare_embeddings(guidance_opt, guidance):
    embeddings = {}
    # text embeddings (stable-diffusion) and (IF), 在[0]上是[1,33,4096]维度，[1]上是[1,2048]维度
    embeddings['default'] = guidance.get_text_embeds([guidance_opt.text])
    embeddings['uncond'] = guidance.get_text_embeds([guidance_opt.negative])

    for d in ['front', 'side', 'back']:
        embeddings[d] = guidance.get_text_embeds([f"{guidance_opt.text}, {d} view"])
    embeddings['inverse_text'] = guidance.get_text_embeds(guidance_opt.inverse_text) #默认是空的
    return embeddings # 字典里装的是元组

def guidance_setup(guidance_opt):
    if guidance_opt.guidance=="SD3":
        from guidance.sd3_utils import SD3
        guidance = SD3(guidance_opt.g_device, guidance_opt.fp16, guidance_opt.vram_O, 
                          guidance_opt.t_range, guidance_opt.max_t_range, 
                          num_train_timesteps=guidance_opt.num_train_timesteps, 
                          ddim_inv=guidance_opt.ddim_inv,
                          textual_inversion_path = guidance_opt.textual_inversion_path,
                          LoRA_path = guidance_opt.LoRA_path,
                          guidance_opt=guidance_opt)
    else:
        raise ValueError(f'{guidance_opt.guidance} not supported.')
    if guidance is not None:
        for p in guidance.parameters():
            p.requires_grad = False
    embeddings = prepare_embeddings(guidance_opt, guidance)
    return guidance, embeddings

def training(dataset, opt, pipe, gcams, guidance_opt, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, save_video):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gcams, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset._white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=dataset.data_device)
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    #
    save_folder = os.path.join(dataset._model_path,"train_process/")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)  # makedirs
        print('train_process is in :', save_folder)
    #controlnet
    use_control_net = False
    #set up pretrain diffusion models and text_embedings 
    guidance, embeddings = guidance_setup(guidance_opt)   
    viewpoint_stack = None
    viewpoint_stack_around = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    if opt.save_process:
        save_folder_proc = os.path.join(scene.args._model_path,"process_videos/")
        if not os.path.exists(save_folder_proc):
            os.makedirs(save_folder_proc)  # makedirs
        process_view_points = scene.getCircleVideoCameras(batch_size=opt.pro_frames_num,render45=opt.pro_render_45).copy()    
        save_process_iter = opt.iterations // len(process_view_points)
        if save_process_iter == 0:
            save_process_iter = 1
        pro_img_frames = []

    for iteration in range(first_iter, opt.iterations + 1):        
        #TODO: DEBUG NETWORK_GUI
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, guidance_opt.text)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        gaussians.update_feature_learning_rate(iteration)
        gaussians.update_rotation_learning_rate(iteration)
        gaussians.update_scaling_learning_rate(iteration)
        # Every 500 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # progressively relaxing view range    
        if not opt.use_progressive:                
            if iteration >= opt.progressive_view_iter and iteration % opt.scale_up_cameras_iter == 0:
                scene.pose_args.fovy_range[0] = max(scene.pose_args.max_fovy_range[0], scene.pose_args.fovy_range[0] * opt.fovy_scale_up_factor[0])
                scene.pose_args.fovy_range[1] = min(scene.pose_args.max_fovy_range[1], scene.pose_args.fovy_range[1] * opt.fovy_scale_up_factor[1])

                scene.pose_args.radius_range[1] = max(scene.pose_args.max_radius_range[1], scene.pose_args.radius_range[1] * opt.scale_up_factor)
                scene.pose_args.radius_range[0] = max(scene.pose_args.max_radius_range[0], scene.pose_args.radius_range[0] * opt.scale_up_factor)

                scene.pose_args.theta_range[1] = min(scene.pose_args.max_theta_range[1], scene.pose_args.theta_range[1] * opt.phi_scale_up_factor)
                scene.pose_args.theta_range[0] = max(scene.pose_args.max_theta_range[0], scene.pose_args.theta_range[0] * 1/opt.phi_scale_up_factor)

                # opt.reset_resnet_iter = max(500, opt.reset_resnet_iter // 1.25)
                scene.pose_args.phi_range[0] = max(scene.pose_args.max_phi_range[0] , scene.pose_args.phi_range[0] * opt.phi_scale_up_factor)
                scene.pose_args.phi_range[1] = min(scene.pose_args.max_phi_range[1], scene.pose_args.phi_range[1] * opt.phi_scale_up_factor)
                
                print('scale up theta_range to:', scene.pose_args.theta_range)
                print('scale up radius_range to:', scene.pose_args.radius_range)
                print('scale up phi_range to:', scene.pose_args.phi_range)
                print('scale up fovy_range to:', scene.pose_args.fovy_range)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getRandTrainCameras().copy()         
        
        C_batch_size = guidance_opt.C_batch_size
        viewpoint_cams = []
        images = []
        text_z_ = []
        weights_ = []
        depths = []
        alphas = []
        scales = []

        text_z_inverse = ( torch.cat( [embeddings['uncond'][0],embeddings['inverse_text'][0]], dim=0),
                          torch.cat([embeddings['uncond'][1],embeddings['inverse_text'][1]], dim=0) )# 这里要返回的是tuple

        for i in range(C_batch_size):
            try:
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))            
            except:
                viewpoint_stack = scene.getRandTrainCameras().copy()
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
                
            #pred text_z
            azimuth = viewpoint_cam.delta_azimuth
            text_z = [embeddings['uncond']]


            if guidance_opt.perpneg: #在sd3中
                text_z_comp, weights = adjust_text_embeddings(embeddings, azimuth, guidance_opt) # 这里embedding是一个dic，里面装的是tuple
                text_z.append(text_z_comp)
                weights_.append(weights)

            # 在sd3中，暂时pool也进行插值
            else:                
                if azimuth >= -90 and azimuth < 90:
                    if azimuth >= 0:
                        r = 1 - azimuth / 90
                    else:
                        r = 1 + azimuth / 90
                    start_z = embeddings['front'] 
                    end_z = embeddings['side']
                else:
                    if azimuth >= 0:
                        r = 1 - (azimuth - 90) / 90
                    else:
                        r = 1 + (azimuth + 90) / 90
                    start_z = embeddings['side']
                    end_z = embeddings['back']
                
                linear_ebedding = (r * start_z[0] + (1 - r) * end_z[0], r * start_z[1] + (1 - r) * end_z[1])

                text_z.append(linear_ebedding)

            text_z_1 = torch.cat([i[0] for i in text_z], dim=0)
            text_z_2 = torch.cat([i[1] for i in text_z], dim=0)
            text_z = (text_z_1, text_z_2)
            # text_z = torch.cat(text_z, dim=0) 
            text_z_.append(text_z) #四个图像里面 

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, 
                                sh_deg_aug_ratio = dataset.sh_deg_aug_ratio, 
                                bg_aug_ratio = dataset.bg_aug_ratio, 
                                shs_aug_ratio = dataset.shs_aug_ratio, 
                                scale_aug_ratio = dataset.scale_aug_ratio)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            depth, alpha = render_pkg["depth"], render_pkg["alpha"]

            scales.append(render_pkg["scales"])
            images.append(image)
            depths.append(depth)
            alphas.append(alpha)
            viewpoint_cams.append(viewpoint_cams)

        images = torch.stack(images, dim=0)
        depths = torch.stack(depths, dim=0)
        alphas = torch.stack(alphas, dim=0)

        # Loss
        warm_up_rate = 1. - min(iteration/opt.warmup_iter,1.) # 从大到小的一个过程
        guidance_scale = guidance_opt.guidance_scale
        _aslatent = False
        if iteration < opt.geo_iter or random.random()< opt.as_latent_ratio:
            _aslatent=True
        if iteration > opt.use_control_net_iter and (random.random() < guidance_opt.controlnet_ratio):
                use_control_net = True
        
        # 需要对text_z_
        text_z_1_ = torch.stack([i[0] for i in text_z_], dim=1) # torch.Size([2, 4, 333, 4096])
        text_z_2_ = torch.stack([i[1] for i in text_z_], dim=1) # torch.Size([2, 4, 2048])
        text_z_ = (text_z_1_, text_z_2_)

        if guidance_opt.perpneg:
            if guidance_opt.train_step_method == "couple":
                    loss = guidance.train_step_couple_perpneg(text_z_, images, 
                                                        pred_depth=depths, pred_alpha=alphas,
                                                        grad_scale=guidance_opt.lambda_guidance,
                                                        use_control_net = use_control_net ,save_folder = save_folder,  iteration = iteration, warm_up_rate=warm_up_rate, 
                                                        weights = torch.stack(weights_, dim=1), resolution=(gcams.image_h, gcams.image_w),
                                                        guidance_opt=guidance_opt,as_latent=_aslatent, embedding_inverse = text_z_inverse)
            else:
                loss = guidance.train_step_perpneg(text_z_, images, 
                                                    pred_depth=depths, pred_alpha=alphas,
                                                    grad_scale=guidance_opt.lambda_guidance,
                                                    use_control_net = use_control_net ,save_folder = save_folder,  iteration = iteration, warm_up_rate=warm_up_rate, 
                                                    weights = torch.stack(weights_, dim=1), resolution=(gcams.image_h, gcams.image_w),
                                                    guidance_opt=guidance_opt,as_latent=_aslatent, embedding_inverse = text_z_inverse)
        else:
            if guidance_opt.train_step_method == "couple":   
                loss = guidance.train_step_couple(text_z_, images, 
                                        pred_depth=depths, pred_alpha=alphas,
                                        grad_scale=guidance_opt.lambda_guidance,
                                        use_control_net = use_control_net ,save_folder = save_folder,  iteration = iteration, warm_up_rate=warm_up_rate, 
                                        resolution=(gcams.image_h, gcams.image_w),
                                        guidance_opt=guidance_opt,as_latent=_aslatent, embedding_inverse = text_z_inverse)
            else:
                loss = guidance.train_step(text_z_, images, 
                                        pred_depth=depths, pred_alpha=alphas,
                                        grad_scale=guidance_opt.lambda_guidance,
                                        use_control_net = use_control_net ,save_folder = save_folder,  iteration = iteration, warm_up_rate=warm_up_rate, 
                                        resolution=(gcams.image_h, gcams.image_w),
                                        guidance_opt=guidance_opt,as_latent=_aslatent, embedding_inverse = text_z_inverse)
                
        scales = torch.stack(scales, dim=0)

        loss_scale = torch.mean(scales,dim=-1).mean()
        loss_tv = tv_loss(images) + tv_loss(depths) 
        # loss_bin = torch.mean(torch.min(alphas - 0.0001, 1 - alphas))

        loss = loss + opt.lambda_tv * loss_tv + opt.lambda_scale * loss_scale #opt.lambda_tv * loss_tv + opt.lambda_bin * loss_bin + opt.lambda_scale * loss_scale +
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if opt.save_process:
                if iteration % save_process_iter == 0 and len(process_view_points) > 0:
                    viewpoint_cam_p = process_view_points.pop(0)
                    render_p = render(viewpoint_cam_p, gaussians, pipe, background, test=True)
                    img_p = torch.clamp(render_p["render"], 0.0, 1.0) 
                    img_p = img_p.detach().cpu().permute(1,2,0).numpy()
                    img_p = (img_p * 255).round().astype('uint8')
                    pro_img_frames.append(img_p)  

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in testing_iterations):
                if save_video:
                    video_inference(iteration, scene, render, (pipe, background))

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration > opt.opacity_reset_from_iter and iteration % opt.opacity_reset_interval == 0: #or (dataset._white_background and iteration == opt.densify_from_iter)
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene._model_path + "/chkpnt" + str(iteration) + ".pth")

    if opt.save_process:
        imageio.mimwrite(os.path.join(save_folder_proc, "video_rgb.mp4"), pro_img_frames, fps=30, quality=8)

def prepare_output_and_logger(args):    
    if not args._model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        # args._model_path = os.path.join("/mnt/workspace/lhy/projects/ICLR2025/results/dreamflow/", args.workspace)
        args._model_path = os.path.join("output", args.workspace)


    # Set up output folder
    print("Output folder: {}".format(args._model_path))
    os.makedirs(args._model_path, exist_ok = True)

    # copy configs
    if args.opt_path is not None:
        os.system(' '.join(['cp', args.opt_path, os.path.join(args._model_path, 'config.yaml')]))

    with open(os.path.join(args._model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args._model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    # Report test and samples of training set
    if iteration in testing_iterations:
        save_folder = os.path.join(scene.args._model_path,"test_six_views/{}_iteration".format(iteration))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)  # makedirs 创建文件时如果路径不存在会创建这个路径
            print('test views is in :', save_folder)
        torch.cuda.empty_cache()
        config = ({'name': 'test', 'cameras' : scene.getTestCameras()})
        if config['cameras'] and len(config['cameras']) > 0:
            for idx, viewpoint in enumerate(config['cameras']):
                render_out = renderFunc(viewpoint, scene.gaussians, *renderArgs, test=True)
                rgb, depth = render_out["render"],render_out["depth"]
                if depth is not None:
                    depth_norm = depth/depth.max()
                    save_image(depth_norm,os.path.join(save_folder,"render_depth_{}.png".format(viewpoint.uid)))

                image = torch.clamp(rgb, 0.0, 1.0)
                save_image(image,os.path.join(save_folder,"render_view_{}.png".format(viewpoint.uid)))
                if tb_writer:
                    tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.uid), image[None], global_step=iteration)     
            print("\n[ITER {}] Eval Done!".format(iteration))
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def video_inference(iteration, scene : Scene, renderFunc, renderArgs):
    sharp = T.RandomAdjustSharpness(3, p=1.0)

    save_folder = os.path.join(scene.args._model_path,"videos/{}_iteration".format(iteration))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)  # makedirs 
        print('videos is in :', save_folder)
    torch.cuda.empty_cache()
    config = ({'name': 'test', 'cameras' : scene.getCircleVideoCameras()})
    if config['cameras'] and len(config['cameras']) > 0:
        img_frames = []
        depth_frames = []
        print("Generating Video using", len(config['cameras']), "different view points")
        for idx, viewpoint in enumerate(config['cameras']):
            render_out = renderFunc(viewpoint, scene.gaussians, *renderArgs, test=True)
            rgb,depth = render_out["render"],render_out["depth"]
            if depth is not None:
                depth_norm = depth/depth.max()
                depths = torch.clamp(depth_norm, 0.0, 1.0) 
                depths = depths.detach().cpu().permute(1,2,0).numpy()
                depths = (depths * 255).round().astype('uint8')          
                depth_frames.append(depths)    
  
            image = torch.clamp(rgb, 0.0, 1.0) 
            image = image.detach().cpu().permute(1,2,0).numpy()
            image = (image * 255).round().astype('uint8')
            img_frames.append(image)    
            #save_image(image,os.path.join(save_folder,"lora_view_{}.jpg".format(viewpoint.uid)))   
        # Img to Numpy
        imageio.mimwrite(os.path.join(save_folder, "video_rgb_{}.mp4".format(iteration)), img_frames, fps=30, quality=8)
        if len(depth_frames) > 0:
            imageio.mimwrite(os.path.join(save_folder, "video_depth_{}.mp4".format(iteration)), depth_frames, fps=30, quality=8)
        print("\n[ITER {}] Video Save Done!".format(iteration))
    torch.cuda.empty_cache()

def convert_to_float_list(char_list):
    string = ''.join(char_list)
    parts = string.split(' ')
    float_list = [float(part) for part in parts]
    return float_list

if __name__ == "__main__":
    import yaml

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    parser.add_argument('--opt', type=str, default=None)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_ratio", type=int, default=5) # 
    parser.add_argument("--save_ratio", type=int, default=5) # 
    parser.add_argument("--save_video", type=bool, default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # parser.add_argument("--device", type=str, default='cuda')
    # parser.add_argument("--pretrained_model_path", type=str, default=None)

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    gcp = GenerateCamParams(parser)
    gp = GuidanceParams(parser)

    args = parser.parse_args(sys.argv[1:])

    if args.opt is not None:
        with open(args.opt) as f:
            opts = yaml.load(f, Loader=yaml.FullLoader)
        lp.load_yaml(opts.get('ModelParams', None))
        op.load_yaml(opts.get('OptimizationParams', None))
        pp.load_yaml(opts.get('PipelineParams', None))
        gcp.load_yaml(opts.get('GenerateCamParams', None))
        gp.load_yaml(opts.get('GuidanceParams', None))
        
        lp.opt_path = args.opt
        args.port = opts['port']
        args.save_video = opts.get('save_video', True)
        args.seed = opts.get('seed', 0)
        args.device = opts.get('device', 'cuda')

        # override device
        gp.g_device = args.device
        lp.data_device = args.device
        gcp.device = args.device

        # others
        lp.workspace = args.workspace
        

        op.iterations = args.iterations
        op.opacity_reset_from_iter = args.opacity_reset_from_iter

        gp.text = args.text
        gp.inverse_guidance_scale = args.inverse_guidance_scale
        gp.time_schedule = args.time_schedule
        gp.guidance_scale = args.guidance_scale
        gp.inverse_steps = args.inverse_steps
        gp.sample_method = args.sample_method
        gp.train_step_method = args.train_step_method
        gp.perpneg = args.perpneg

        if isinstance(args.radius_range[0], str):
            gcp.radius_range =  convert_to_float_list(args.radius_range) #[5.2, 5.5] #[3.8, 4.5] #[3.0, 3.5]
            gcp.max_radius_range = convert_to_float_list(args.max_radius_range) #[3.5, 5.0]
        gcp.default_radius = args.default_radius # 3.5

        gcp.init_prompt = args.init_prompt
        gcp.init_shape = args.init_shape
        gcp.up_dwon_shift = args.up_dwon_shift

        # pre_trained_model
        # lp.pretrained_model_path = args.pretrained_model_path

    # save iterations
    test_iter = [1] + [k * op.iterations // args.test_ratio for k in range(1, args.test_ratio)] + [op.iterations]
    args.test_iterations = test_iter

    save_iter = [k * op.iterations // args.save_ratio for k in range(1, args.save_ratio)] + [op.iterations]
    # save_iter = [op.iterations]
    args.save_iterations = save_iter

    print('Test iter:', args.test_iterations)
    print('Save iter:', args.save_iterations)

    print("Some important parameters:\n"), 
    print(gp.text, gp.inverse_guidance_scale, gp.time_schedule, gp.guidance_scale, gp.inverse_steps, gp.sample_method,gp.train_step_method)
    print(gcp.init_prompt, gcp.init_shape, gcp.default_radius, gcp.radius_range, gcp.max_radius_range, gcp.up_dwon_shift)
    print("perpneg:",gp.perpneg)
    print("Optimizing " + lp._model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, seed=args.seed)
    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp, op, pp, gcp, gp, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.save_video)

    # All done
    print("\nTraining complete.")
