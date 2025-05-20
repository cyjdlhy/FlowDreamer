from audioop import mul
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import StableDiffusionPipeline, DiffusionPipeline, DDPMScheduler, DDIMScheduler, EulerDiscreteScheduler, \
                      EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, ControlNetModel, \
                      DDIMInverseScheduler, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from os.path import isfile
from pathlib import Path
import os
import random

import torchvision.transforms as T
# suppress partial model loading warning
logging.set_verbosity_error()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
from torch.cuda.amp import custom_bwd, custom_fwd
from .perpneg_utils import weighted_perpendicular_aggregator

# from .sd_step import *
from diffusers import StableDiffusion3Pipeline
import copy

def rgb2sat(img, T=None):
    max_ = torch.max(img, dim=1, keepdim=True).values + 1e-5
    min_ = torch.min(img, dim=1, keepdim=True).values
    sat = (max_ - min_) / max_
    if T is not None:
        sat = (1 - T) * sat
    return sat

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class SD3(nn.Module):
    def __init__(self, device, fp16, vram_O, t_range=[0.02, 0.98], max_t_range=0.98, num_train_timesteps=None, 
                 ddim_inv=False, use_control_net=False, textual_inversion_path = None, 
                 LoRA_path = None, guidance_opt=None):
        super().__init__()

        self.device = device
        self.precision_t = torch.float16 if fp16 else torch.float32

        print(f'[INFO] loading stable diffusion3...')

        model_key = guidance_opt.model_key
        assert model_key is not None

        pipe = StableDiffusion3Pipeline.from_pretrained(model_key, torch_dtype=self.precision_t)
        self.scheduler = pipe.scheduler

        if use_control_net:
            controlnet_model_key = guidance_opt.controlnet_model_key
            self.controlnet_depth = ControlNetModel.from_pretrained(controlnet_model_key,torch_dtype=self.precision_t).to(device)

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            pipe.enable_model_cpu_offload()

        # pipe.enable_xformers_memory_efficient_attention() #这个用在SD3中是一个bug

        pipe = pipe.to(self.device)
        if textual_inversion_path is not None:
            pipe.load_textual_inversion(textual_inversion_path)
            print("load textual inversion in:.{}".format(textual_inversion_path))

        #         
        # if LoRA_path is not None:
        #     from lora_diffusion import tune_lora_scale, patch_pipe
        #     print("load lora in:.{}".format(LoRA_path))
        #     patch_pipe(
        #         pipe,
        #         LoRA_path,
        #         patch_text=True,
        #         patch_ti=True,
        #         patch_unet=True,
        #     )
        #     tune_lora_scale(pipe.unet, 1.00)
        #     tune_lora_scale(pipe.text_encoder, 1.00)

        self.pipe = pipe
        self.vae = pipe.vae
        self.transformer = pipe.transformer #unet换成trainformer了
        
        # 锁住所有参数
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.transformer.parameters():
            p.requires_grad_(False)


        self.num_train_timesteps = num_train_timesteps if num_train_timesteps is not None else self.scheduler.config.num_train_timesteps        
        self.timesteps = torch.flip(self.scheduler.timesteps, dims=(0, )).to(self.device) #越来越大，就是t越大，那么noise的比例就越大
        
        # 为下面的sigmas作准备
        self.noise_scheduler_copy = copy.deepcopy(self.pipe.scheduler)

        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.warmup_step = int(self.num_train_timesteps*(max_t_range-t_range[1]))

        self.noise_temp = None
        self.noise_gen = torch.Generator(self.device)
        self.noise_gen.manual_seed(guidance_opt.noise_seed)

        # self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        self.rgb_latent_factors = torch.tensor([
                    # R       G       B
                    [ 0.298,  0.207,  0.208],
                    [ 0.187,  0.286,  0.173],
                    [-0.158,  0.189,  0.264],
                    [-0.184, -0.271, -0.473]
                ], device=self.device)
        
        # 下面是我自己定义一些参数
        self.step = 1

        print(f'[INFO] loaded stable diffusion!')

    def augmentation(self, *tensors):
        augs = T.Compose([
                        T.RandomHorizontalFlip(p=0.5),
                    ])
        
        channels = [ten.shape[1] for ten in tensors]
        tensors_concat = torch.concat(tensors, dim=1)
        tensors_concat = augs(tensors_concat)

        results = []
        cur_c = 0
        for i in range(len(channels)):
            results.append(tensors_concat[:, cur_c:cur_c + channels[i], ...])
            cur_c += channels[i]
        return (ten for ten in results)

    def noise_prediction(self, latents, noise, 
                           ind_t, ind_prev_t, 
                           text_embeddings=None, cfg=1.0, 
                           delta_t=1, inv_steps=1,
                           is_noisy_latent=False,
                           eta=0.0):

        text_embeddings = ( text_embeddings[0].to(self.precision_t),
                            text_embeddings[1].to(self.precision_t) )

        if cfg <= 1.0:
            uncond_text_embedding = ( text_embeddings[0].reshape(2, -1, text_embeddings[0].shape[-2], text_embeddings[0].shape[-1])[1],
                                      text_embeddings[1].reshape(2, -1, text_embeddings[1].shape[-1])[1] )

        # unet = self.unet

        # 下面的t都是index
        if is_noisy_latent:
            prev_noisy_lat = latents
        else:
            # rcflow的t 和 diffusion中的t 是相反的
            # ind_prev_t = self.num_train_timesteps - ind_prev_t
            prev_t = self.timesteps[ind_prev_t]
            sigma = self.get_sigmas(torch.tensor([prev_t]), n_dim=latents.ndim, dtype=latents.dtype) #n_dim 代表想要输出的sigma.ndim的大小
            prev_noisy_lat = sigma * noise + (1-sigma) * latents

            # prev_noisy_lat = self.timesteps[(self.num_train_timesteps -1 -ind_prev_t)] / self.num_train_timesteps * latents + ( 1- self.timesteps[(self.num_train_timesteps-1-ind_prev_t)]/self.num_train_timesteps) * noise

            # prev_noisy_lat = self.scheduler.add_noise(latents, noise, self.timesteps[ind_prev_t])

        cur_ind_t = ind_prev_t
        cur_noisy_lat = prev_noisy_lat

        pred_scores = []

        for i in range(inv_steps):
            # pred noise
            # cur_noisy_lat_ = self.scheduler.scale_model_input(cur_noisy_lat, self.timesteps[cur_ind_t]).to(self.precision_t)
            
            # cur_ind_t = self.num_train_timesteps -1 - cur_ind_t

            if cfg > 1.0:
                latent_model_input = torch.cat([cur_noisy_lat, cur_noisy_lat])
                
                timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
                    
                unet_output = self.transformer(hidden_states = latent_model_input.to(self.precision_t), 
                                            timestep = timestep_model_input.to(self.precision_t), 
                                            encoder_hidden_states= text_embeddings[0].to(self.precision_t),
                                            pooled_projections = text_embeddings[1].to(self.precision_t),
                                            return_dict=False,
                                                )[0].to(latent_model_input.dtype)
                
                # unet_output = unet(latent_model_input, timestep_model_input, 
                #                 encoder_hidden_states=text_embeddings).sample
                
                uncond, cond = torch.chunk(unet_output, chunks=2)
                
                unet_output = cond + cfg * (uncond - cond) # reverse cfg to enhance the distillation
            else:
                timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(cur_noisy_lat.shape[0], 1).reshape(-1)
                latent_model_input = cur_noisy_lat
                unet_output = self.transformer(hidden_states = latent_model_input.to(self.precision_t), 
                                            timestep = timestep_model_input.to(self.precision_t), 
                                            encoder_hidden_states= uncond_text_embedding[0].to(self.precision_t),
                                            pooled_projections = uncond_text_embedding[1].to(self.precision_t),
                                            return_dict=False,
                                                )[0].to(latent_model_input.dtype)
                
                # unet_output = unet(cur_noisy_lat, timestep_model_input, 
                #                     encoder_hidden_states=uncond_text_embedding).sample

            pred_scores.append((cur_ind_t, unet_output)) # t和 (x1-x0)|t

            # cur_ind_t = self.num_train_timesteps -1 - cur_ind_t
            next_ind_t = min(cur_ind_t + delta_t, ind_t) 
            cur_t, next_t = self.timesteps[cur_ind_t], self.timesteps[next_ind_t]
            
            sigma = self.get_sigmas(torch.tensor([cur_t]), n_dim=latents.ndim, dtype=latents.dtype) #n_dim 代表想要输出的sigma.ndim的大小
            next_sigma = self.get_sigmas(torch.tensor([next_t]), n_dim=latents.ndim, dtype=latents.dtype) #n_dim 代表想要输出的sigma.ndim的大小
            # delta_t_ = next_t-cur_t if isinstance(self.scheduler, DDIMScheduler) else next_ind_t-cur_ind_t
            dt = next_sigma - sigma

            # 朝着noise的方向走, 这里的delta_t_是负数
            cur_noisy_lat = cur_noisy_lat + dt * unet_output

            # cur_noisy_lat = self.sche_func(self.scheduler, unet_output, cur_t, cur_noisy_lat, -delta_t_, eta).prev_sample
            cur_ind_t = next_ind_t

            del unet_output
            torch.cuda.empty_cache()

            if cur_ind_t == ind_t:
                break
        
        # pred_scores 记录着times以及其pred_scores
        return prev_noisy_lat, cur_noisy_lat, pred_scores[::-1] 

    @torch.no_grad()
    def get_text_embeds(self, prompt,resolution=(512, 512)):
        
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            device=self.device,
        )
        
        return prompt_embeds,pooled_prompt_embeds # 


    def get_sigmas(self,timesteps, n_dim=4, dtype=torch.float16):

        sigmas = self.noise_scheduler_copy.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler_copy.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    

    # 简单采用这个x1-x0 - (x1_pred-x0_pred) 为direction
    # VFDS
    def train_step(self, text_embeddings, pred_rgb, pred_depth=None, pred_alpha=None,
                    grad_scale=1,use_control_net=False,
                    save_folder:Path=None, iteration=0, warm_up_rate = 0,
                    resolution=(512, 512), guidance_opt=None,as_latent=False, embedding_inverse = None):
        # print("Now, we use train_step to train our model.")
        pred_rgb, pred_depth, pred_alpha = self.augmentation(pred_rgb, pred_depth, pred_alpha)

        B = pred_rgb.shape[0]
        K = text_embeddings[0].shape[0] - 1

        if as_latent:      
            latents,_ = self.encode_imgs(pred_depth.repeat(1,3,1,1).to(self.precision_t))
        else:
            latents,_ = self.encode_imgs(pred_rgb.to(self.precision_t))
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level

        if self.noise_temp is None:
            # self.noise_temp = torch.randn((latents.shape[0], 16, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen) + 0.1 * torch.randn((1, 16, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)
            self.noise_temp = torch.randn((latents.shape[0], 16, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen)
            self.noise_temp = self.noise_temp.to(latents.dtype)

        if guidance_opt.fix_noise:
            noise = self.noise_temp
        else:
            # noise = torch.randn((latents.shape[0], 16, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen) + 0.1 * torch.randn((1, 16, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)
            noise = torch.randn((latents.shape[0], 16, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen)

        text_embeddings = text_embeddings[0][:, :, ...],text_embeddings[1][:, :, ...]
        text_embeddings = ( text_embeddings[0].reshape(-1, text_embeddings[0].shape[-2], text_embeddings[0].shape[-1]),
                            text_embeddings[1].reshape(-1, text_embeddings[1].shape[-1]) ) # make it k+1, c * t, ...

        inverse_text_embeddings = ( embedding_inverse[0].unsqueeze(1).repeat(1, B, 1, 1).reshape(-1, embedding_inverse[0].shape[-2], embedding_inverse[0].shape[-1]),
                                    embedding_inverse[1].unsqueeze(1).repeat(1, B, 1).reshape(-1, embedding_inverse[1].shape[-1]) )

        if guidance_opt.annealing_intervals:
            current_delta_t =  int(guidance_opt.delta_t + (warm_up_rate)*(guidance_opt.delta_t_start - guidance_opt.delta_t))
        else:
            current_delta_t =  guidance_opt.delta_t

        if guidance_opt.annealing_intervals:
            ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)
        else:
            warm_up_rate = 1
            ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)

        t = self.timesteps[ind_t]
        sigma = self.get_sigmas(t, n_dim=latents.ndim, dtype=latents.dtype) #n_dim 代表想要输出的sigma.ndim的大小

        with torch.no_grad():
           
            latents_noisy = sigma * noise + (1.0 - sigma) * latents
            latent_model_input = latents_noisy[None, :, ...].repeat(2, 1, 1, 1, 1).reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, )
            tt = t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
            # 这里的self.transformer有两个embedding
            unet_output = self.transformer(hidden_states = latent_model_input.to(self.precision_t), 
                                           timestep = tt.to(self.precision_t), 
                                           encoder_hidden_states= text_embeddings[0].to(self.precision_t),
                                           pooled_projections = text_embeddings[1].to(self.precision_t),
                                           return_dict=False,
                                            )[0].to(latent_model_input.dtype)

            unet_output = unet_output.reshape(2, -1, 16, resolution[0] // 8, resolution[1] // 8, )
            noise_pred_uncond, noise_pred_text = unet_output[:1].reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, ), unet_output[1:].reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, )
            delta_VFDS = noise_pred_text - noise_pred_uncond

        pred_grad = noise_pred_uncond + guidance_opt.guidance_scale * delta_VFDS
        grad = torch.nan_to_num(pred_grad - (noise - latents)) * grad_scale
        grad = (grad).detach()
        loss = SpecifyGradient.apply(latents, grad)
              
        if iteration % guidance_opt.vis_interval == 0:
            noise_pred_post = noise_pred_uncond + guidance_opt.guidance_scale* delta_VFDS    
            # lat2rgb = lambda x: torch.clip((x.permute(0,2,3,1) @ self.rgb_latent_factors.to(x.dtype)).permute(0,3,1,2), 0., 1.)
            save_path_iter = os.path.join(save_folder,"iter_{}_step_{}.jpg".format(iteration,t.item()))
            with torch.no_grad():

                grad_abs = torch.abs(grad.detach())
                norm_grad  = F.interpolate((grad_abs / grad_abs.max()).mean(dim=1,keepdim=True), (resolution[0], resolution[1]), mode='bilinear', align_corners=False).repeat(1,3,1,1)

                viz_images = torch.cat([pred_rgb, pred_depth.repeat(1, 3, 1, 1), 
                                        pred_alpha.repeat(1, 3, 1, 1), rgb2sat(pred_rgb, pred_alpha).repeat(1, 3, 1, 1),
                                        norm_grad,],dim=0)


                save_image(viz_images, save_path_iter)

        return loss

    def train_step_perpneg(self, text_embeddings, pred_rgb, pred_depth=None, pred_alpha=None,
                    grad_scale=1,use_control_net=False,
                    save_folder:Path=None, iteration=0, warm_up_rate = 0,
                    resolution=(512, 512), guidance_opt=None,as_latent=False, embedding_inverse = None, weights = None):
        # print("Now, we use train_step to train our model.")
        pred_rgb, pred_depth, pred_alpha = self.augmentation(pred_rgb, pred_depth, pred_alpha)

        B = pred_rgb.shape[0]
        K = text_embeddings[0].shape[0] - 1

        if as_latent:      
            latents,_ = self.encode_imgs(pred_depth.repeat(1,3,1,1).to(self.precision_t))
        else:
            latents,_ = self.encode_imgs(pred_rgb.to(self.precision_t))
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level

        if self.noise_temp is None:
            # self.noise_temp = torch.randn((latents.shape[0], 16, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen) + 0.1 * torch.randn((1, 16, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)
            self.noise_temp = torch.randn((latents.shape[0], 16, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen)
            self.noise_temp = self.noise_temp.to(latents.dtype)

        if guidance_opt.fix_noise:
            noise = self.noise_temp
        else:
            # noise = torch.randn((latents.shape[0], 16, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen) + 0.1 * torch.randn((1, 16, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)
            noise = torch.randn((latents.shape[0], 16, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen)

        weights = weights.reshape(-1)

        text_embeddings = text_embeddings[0][:, :, ...],text_embeddings[1][:, :, ...]
        text_embeddings = ( text_embeddings[0].reshape(-1, text_embeddings[0].shape[-2], text_embeddings[0].shape[-1]),
                            text_embeddings[1].reshape(-1, text_embeddings[1].shape[-1]) ) # make it k+1, c * t, ...

        inverse_text_embeddings = ( embedding_inverse[0].unsqueeze(1).repeat(1, B, 1, 1).reshape(-1, embedding_inverse[0].shape[-2], embedding_inverse[0].shape[-1]),
                                    embedding_inverse[1].unsqueeze(1).repeat(1, B, 1).reshape(-1, embedding_inverse[1].shape[-1]) )

        if guidance_opt.annealing_intervals:
            ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)
        else:
            warm_up_rate = 1
            ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)

        t = self.timesteps[ind_t]
        sigma = self.get_sigmas(t, n_dim=latents.ndim, dtype=latents.dtype) #n_dim 代表想要输出的sigma.ndim的大小

        with torch.no_grad():
           
            latents_noisy = sigma * noise + (1.0 - sigma) * latents
            latent_model_input = latents_noisy[None, :, ...].repeat(1+K, 1, 1, 1, 1).reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, )
            tt = t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
            # 这里的self.transformer有两个embedding
            unet_output = self.transformer(hidden_states = latent_model_input.to(self.precision_t), 
                                           timestep = tt.to(self.precision_t), 
                                           encoder_hidden_states= text_embeddings[0].to(self.precision_t),
                                           pooled_projections = text_embeddings[1].to(self.precision_t),
                                           return_dict=False,
                                            )[0].to(latent_model_input.dtype)

            unet_output = unet_output.reshape(1+K, -1, 16, resolution[0] // 8, resolution[1] // 8, )
            noise_pred_uncond, noise_pred_text = unet_output[:1].reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, ), unet_output[1:].reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, )
            
            delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1)
            
            delta_VFDS = weighted_perpendicular_aggregator(delta_noise_preds,\
                                                            weights,\
                                                            B)   
            
        pred_grad = noise_pred_uncond + guidance_opt.guidance_scale * delta_VFDS
        
        grad = torch.nan_to_num(pred_grad - (noise - latents)) * grad_scale

        grad = (grad).detach()
        # loss_rfds = F.mse_loss(latents - noise, target, reduction="mean") / B
        loss = SpecifyGradient.apply(latents, grad)
              
        if iteration % guidance_opt.vis_interval == 0:
            noise_pred_post = noise_pred_uncond + guidance_opt.guidance_scale* delta_VFDS    
            # lat2rgb = lambda x: torch.clip((x.permute(0,2,3,1) @ self.rgb_latent_factors.to(x.dtype)).permute(0,3,1,2), 0., 1.)
            save_path_iter = os.path.join(save_folder,"iter_{}_step_{}.jpg".format(iteration,t.item()))
            with torch.no_grad():
                # pred_x0_latent_sp = pred_original(self.scheduler, noise_pred_uncond, prev_t, prev_latents_noisy)    
                # pred_x0_latent_pos = pred_original(self.scheduler, noise_pred_post, prev_t, prev_latents_noisy)        
                # pred_x0_pos = self.decode_latents(pred_x0_latent_pos.type(self.precision_t))
                # pred_x0_sp = self.decode_latents(pred_x0_latent_sp.type(self.precision_t))
                # pred_x0_uncond = pred_x0_sp[:1, ...]

                grad_abs = torch.abs(grad.detach())
                norm_grad  = F.interpolate((grad_abs / grad_abs.max()).mean(dim=1,keepdim=True), (resolution[0], resolution[1]), mode='bilinear', align_corners=False).repeat(1,3,1,1)

                # latents_rgb = F.interpolate(lat2rgb(latents), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)
                # latents_sp_rgb = F.interpolate(lat2rgb(pred_x0_latent_sp), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)

                # viz_images = torch.cat([pred_rgb, 
                #                         pred_depth.repeat(1, 3, 1, 1), 
                #                         pred_alpha.repeat(1, 3, 1, 1), 
                #                         rgb2sat(pred_rgb, pred_alpha).repeat(1, 3, 1, 1),
                #                         latents_rgb, latents_sp_rgb, norm_grad,
                #                         pred_x0_sp, pred_x0_pos],dim=0) 

                viz_images = torch.cat([pred_rgb, pred_depth.repeat(1, 3, 1, 1), 
                                        pred_alpha.repeat(1, 3, 1, 1), rgb2sat(pred_rgb, pred_alpha).repeat(1, 3, 1, 1),
                                        norm_grad,],dim=0)
                                        # latents_rgb, norm_grad,],dim=0) 

                save_image(viz_images, save_path_iter)

        return loss

    # 获得在t时刻的梯度
    def ODE_step(self, latents, resolution, inverse_t, inverse_text_embeddings, inverse_guidance_scale, B, K = 0, weights=None, text_embeddings=None):
        
        with torch.no_grad():
            if inverse_guidance_scale > 1.0:

                latents_noisy = latents
                latent_model_input = latents_noisy[None, :, ...].repeat(2, 1, 1, 1, 1).reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, )
                tt = torch.tensor([inverse_t]).reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
                tt = tt.to(self.device)
                # 这里的self.transformer有两个embedding
                unet_output = self.transformer(hidden_states = latent_model_input.to(self.precision_t), 
                                            timestep = tt.to(self.precision_t), 
                                            encoder_hidden_states= inverse_text_embeddings[0].to(self.precision_t),
                                            pooled_projections = inverse_text_embeddings[1].to(self.precision_t),
                                            return_dict=False,
                                                )[0].to(latent_model_input.dtype)

                unet_output = unet_output.reshape(2, -1, 16, resolution[0] // 8, resolution[1] // 8, )
                noise_pred_uncond, noise_pred_text = unet_output[:1].reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, ), unet_output[1:].reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, )
                delta_couple = noise_pred_text - noise_pred_uncond
                pred_grad = noise_pred_uncond + inverse_guidance_scale * delta_couple
            
            else:
                
                latents_noisy = latents
                latent_model_input = latents_noisy[None, :, ...].repeat(1, 1, 1, 1, 1).reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, )
                tt = torch.tensor([inverse_t]).reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
                tt = tt.to(self.device)

                pred_grad = self.transformer(hidden_states = latent_model_input.to(self.precision_t), 
                                            timestep = tt.to(self.precision_t), 
                                            encoder_hidden_states= inverse_text_embeddings[0][B:].to(self.precision_t),
                                            pooled_projections = inverse_text_embeddings[1][B:].to(self.precision_t),
                                            return_dict=False,
                                                )[0].to(latent_model_input.dtype)
            return pred_grad

    def get_inverse_dt_list(self,inverse_steps, time_schedule="original"):
        # 

        inverse_start = torch.tensor(0.0).to(self.device) # 因为self.time_steps[0]

        if time_schedule == 'uniform':
            
            dt = (1000-inverse_start)/(inverse_steps*1000) # 0~1之间
            inverse_dt_list = [dt for i in range(inverse_steps)]
        
        elif time_schedule == 'original':
            inverse_timesteps_list = [inverse_start] # 先加起点
            
            # inverse_steps = inverse_steps
            step_size = len(self.timesteps)//inverse_steps -1 
            for i in range(inverse_steps-1):
                inverse_timesteps_list.append(self.timesteps[(i+1)*step_size])
            inverse_timesteps_list.append(self.timesteps[-1])

            inverse_dt_list = []
            for i in range(inverse_steps):
                inverse_dt_list.append((inverse_timesteps_list[i+1]-inverse_timesteps_list[i])/1000)

        elif time_schedule == 'random':
            # 希望他们的间隔都至少大于0.05
            dt_min = 0.05 # 希望每一个间隔都需要大于 50
            remain_sum = 1 - dt_min * inverse_steps
            numbers = [random.randint(1,remain_sum*1000)/1000 for i in range(inverse_steps)]
            number_sum = sum(numbers)
            inverse_dt_list = []
            for number in numbers:
                inverse_dt_list.append(torch.tensor(number/number_sum * remain_sum + dt_min).to(self.device) )
            # numbers = numbers + dt_min
            # inverse_dt_list = numbers

        return inverse_dt_list

    def train_step_couple_perpneg(self, text_embeddings, pred_rgb, pred_depth=None, pred_alpha=None,
                           grad_scale=1,use_control_net=False,
                           save_folder:Path=None, iteration=0, warm_up_rate = 0, weights = 0, 
                           resolution=(512, 512), guidance_opt=None,as_latent=False, embedding_inverse = None):


        pred_rgb, pred_depth, pred_alpha = self.augmentation(pred_rgb, pred_depth, pred_alpha)

        B = pred_rgb.shape[0]
        K = text_embeddings[0].shape[0] - 1

        if as_latent:      
            latents,_ = self.encode_imgs(pred_depth.repeat(1,3,1,1).to(self.precision_t))
        else:
            latents,_ = self.encode_imgs(pred_rgb.to(self.precision_t))

        weights = weights.reshape(-1)

        text_embeddings = text_embeddings[0][:, :, ...],text_embeddings[1][:, :, ...]
        text_embeddings = ( text_embeddings[0].reshape(-1, text_embeddings[0].shape[-2], text_embeddings[0].shape[-1]),
                            text_embeddings[1].reshape(-1, text_embeddings[1].shape[-1]) ) # make it k+1, c * t, ...

        inverse_text_embeddings = ( embedding_inverse[0].unsqueeze(1).repeat(1, B, 1, 1).reshape(-1, embedding_inverse[0].shape[-2], embedding_inverse[0].shape[-1]),
                                    embedding_inverse[1].unsqueeze(1).repeat(1, B, 1).reshape(-1, embedding_inverse[1].shape[-1]) )

        if guidance_opt.annealing_intervals:
            current_delta_t =  int(guidance_opt.delta_t + (warm_up_rate)*(guidance_opt.delta_t_start - guidance_opt.delta_t))
        else:
            current_delta_t =  guidance_opt.delta_t

        # ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        if guidance_opt.annealing_intervals:
            ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        else:
            warm_up_rate = 1
            ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]

        t = self.timesteps[ind_t]

        ori_latents = latents.clone()

        # 超参数
        inverse_steps = guidance_opt.inverse_steps
        inverse_guidance_scale = guidance_opt.inverse_guidance_scale


        with torch.no_grad():

            inverse_t = torch.tensor(0.0).to(self.device)
            inverse_dt_list = self.get_inverse_dt_list(inverse_steps, guidance_opt.time_schedule)
            # latents = latents[None, :, ...].repeat(1, 1, 1, 1, 1).reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, )

            if guidance_opt.sample_method == "euler":
                for dt in inverse_dt_list:
                    inverse_t = torch.tensor([min(inverse_t,1000.0)], device=self.device) # 限制在0~1000之间
                    pred_grad = self.ODE_step(latents, resolution, inverse_t, inverse_text_embeddings, inverse_guidance_scale, B)
                    latents = latents + pred_grad * dt
                    inverse_t += dt*1000
            
            elif guidance_opt.sample_method == "rk45_2":
                for dt in inverse_dt_list:
                    # inverse_t = inverse_t * 1000                
                    inverse_t = torch.tensor([min(inverse_t,1000.0)], device=self.device) # 限制在0~1000之间
                    k1 = self.ODE_step(latents, resolution, inverse_t, inverse_text_embeddings, inverse_guidance_scale, B)
                    k2 = self.ODE_step(latents + dt * k1, resolution, inverse_t + (dt*1000), inverse_text_embeddings, inverse_guidance_scale, B)
                    # pred_grad = 0.5 * (k1 + k2)
                    latents = latents + 0.5 * dt * (k1 + k2)
                    # 更新inverse_t
                    inverse_t += dt*1000


            elif guidance_opt.sample_method == "rk45_4":

                for dt in inverse_dt_list:
                    inverse_t = torch.tensor([min(inverse_t,1000.0)], device=self.device)
                    k1 = self.ODE_step(latents, resolution, inverse_t, inverse_text_embeddings, inverse_guidance_scale, B)
                    k2 = self.ODE_step(latents + 0.5 * dt * k1, resolution, inverse_t + 0.5 * (dt*1000), inverse_text_embeddings, inverse_guidance_scale, B)
                    k3 = self.ODE_step(latents + 0.5 * dt * k2, resolution, inverse_t + 0.5 * (dt*1000), inverse_text_embeddings, inverse_guidance_scale, B)
                    k4 = self.ODE_step(latents + dt * k3, resolution, inverse_t + (dt*1000), inverse_text_embeddings, inverse_guidance_scale, B)
                    latents = latents + (1/6) * dt * (k1 + 2*k2 + 2*k3 + k4)
                    inverse_t = inverse_t + (dt*1000)

            elif guidance_opt.sample_method == "mid_point":

                for dt in inverse_dt_list:
                    inverse_t = torch.tensor([min(inverse_t,1000.0)], device=self.device)
                    k1 = self.ODE_step(latents, resolution, inverse_t, inverse_text_embeddings, inverse_guidance_scale, B)
                    k2 = self.ODE_step(latents + 0.5 * dt * k1, resolution, inverse_t + 0.5 * (dt*1000), inverse_text_embeddings, inverse_guidance_scale, B)
                    latents = latents + dt * k2
                    inverse_t = inverse_t + (dt*1000)

            noise = latents
            latents = ori_latents

            # 第二次预测，加入cfg, 在perpneg中要考虑这里的K的reshape的问题
            sigma = self.get_sigmas(torch.tensor([t]), n_dim=latents.ndim, dtype=latents.dtype) #n_dim 代表想要输出的sigma.ndim的大小
            
            sample_noise = torch.randn((latents.shape[0], 16, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen)
            sample_noise = sample_noise.to(latents.dtype)
            # noise = 0.7*noise + 0.3*sample_noise
            noise = (1-sigma/2) * noise + sigma/2 * sample_noise


            latents_noisy = sigma * noise + ( 1- sigma) * latents
            latent_model_input = latents_noisy[None, :, ...].repeat(1+K, 1, 1, 1, 1).reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, )
            tt = t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)

            unet_output = self.transformer(hidden_states = latent_model_input.to(self.precision_t), 
                                        timestep = tt.to(self.precision_t), 
                                        encoder_hidden_states= text_embeddings[0].to(self.precision_t),
                                        pooled_projections = text_embeddings[1].to(self.precision_t),
                                        return_dict=False,
                                            )[0].to(latent_model_input.dtype)
    

            unet_output = unet_output.reshape(1+K, -1, 16, resolution[0] // 8, resolution[1] // 8, )
            noise_pred_uncond, noise_pred_text = unet_output[:1].reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, ), unet_output[1:].reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, )
            delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1)

            delta_couple = weighted_perpendicular_aggregator(delta_noise_preds,\
                                                            weights,\
                                                            B)    


        pred_grad = noise_pred_uncond + guidance_opt.guidance_scale * delta_couple
        
        # grad = torch.nan_to_num(grad_scale * pred_grad )
        grad = torch.nan_to_num(pred_grad - (noise - latents) ) * grad_scale

        grad = (grad).detach()
        # loss_rfds = F.mse_loss(latents - noise, target, reduction="mean") / B
        loss = SpecifyGradient.apply(latents, grad)
              
        if iteration % guidance_opt.vis_interval == 0:
            noise_pred_post = noise_pred_uncond + guidance_opt.guidance_scale* delta_couple    
            # lat2rgb = lambda x: torch.clip((x.permute(0,2,3,1) @ self.rgb_latent_factors.to(x.dtype)).permute(0,3,1,2), 0., 1.)
            save_path_iter = os.path.join(save_folder,"iter_{}_step_{}.jpg".format(iteration, int(t.item()) ))
            with torch.no_grad():
                grad_abs = torch.abs(grad.detach())
                norm_grad  = F.interpolate((grad_abs / grad_abs.max()).mean(dim=1,keepdim=True), (resolution[0], resolution[1]), mode='bilinear', align_corners=False).repeat(1,3,1,1)

                viz_images = torch.cat([pred_rgb, pred_depth.repeat(1, 3, 1, 1), 
                                        pred_alpha.repeat(1, 3, 1, 1), rgb2sat(pred_rgb, pred_alpha).repeat(1, 3, 1, 1),
                                        norm_grad,],dim=0) 

                save_image(viz_images, save_path_iter)

        self.step += 1
        return loss

    def train_step_couple(self, text_embeddings, pred_rgb, pred_depth=None, pred_alpha=None,
                           grad_scale=1,use_control_net=False,
                           save_folder:Path=None, iteration=0, warm_up_rate = 0,
                           resolution=(512, 512), guidance_opt=None,as_latent=False, embedding_inverse = None):


        pred_rgb, pred_depth, pred_alpha = self.augmentation(pred_rgb, pred_depth, pred_alpha)

        B = pred_rgb.shape[0]
        K = text_embeddings[0].shape[0] - 1

        if as_latent:      
            latents,_ = self.encode_imgs(pred_depth.repeat(1,3,1,1).to(self.precision_t))
        else:
            latents,_ = self.encode_imgs(pred_rgb.to(self.precision_t))


        text_embeddings = text_embeddings[0][:, :, ...],text_embeddings[1][:, :, ...]
        text_embeddings = ( text_embeddings[0].reshape(-1, text_embeddings[0].shape[-2], text_embeddings[0].shape[-1]),
                            text_embeddings[1].reshape(-1, text_embeddings[1].shape[-1]) ) # make it k+1, c * t, ...

        inverse_text_embeddings = ( embedding_inverse[0].unsqueeze(1).repeat(1, B, 1, 1).reshape(-1, embedding_inverse[0].shape[-2], embedding_inverse[0].shape[-1]),
                                    embedding_inverse[1].unsqueeze(1).repeat(1, B, 1).reshape(-1, embedding_inverse[1].shape[-1]) )

        if guidance_opt.annealing_intervals:
            current_delta_t =  int(guidance_opt.delta_t + (warm_up_rate)*(guidance_opt.delta_t_start - guidance_opt.delta_t))
        else:
            current_delta_t =  guidance_opt.delta_t

        # ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        if guidance_opt.annealing_intervals:
            ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        else:
            warm_up_rate = 1
            ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]


        t = self.timesteps[ind_t]
        
        ori_latents = latents.clone()

        # 超参数
        inverse_steps = guidance_opt.inverse_steps
        inverse_guidance_scale = guidance_opt.inverse_guidance_scale
        # guidance_opt.sample_method
        # print(inverse_guidance_scale)

        with torch.no_grad():

            inverse_t = torch.tensor(0.0).to(self.device)
            inverse_dt_list = self.get_inverse_dt_list(inverse_steps, guidance_opt.time_schedule)
            # latents = latents[None, :, ...].repeat(1, 1, 1, 1, 1).reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, )

            if guidance_opt.sample_method == "euler":
                for dt in inverse_dt_list:
                    inverse_t = torch.tensor([min(inverse_t,1000.0)], device=self.device) # 限制在0~1000之间
                    pred_grad = self.ODE_step(latents, resolution, inverse_t, inverse_text_embeddings, inverse_guidance_scale, B)
                    latents = latents + pred_grad * dt
                    inverse_t += dt*1000
            
            elif guidance_opt.sample_method == "rk45_2":
                for dt in inverse_dt_list:
                    # inverse_t = inverse_t * 1000                
                    inverse_t = torch.tensor([min(inverse_t,1000.0)], device=self.device) # 限制在0~1000之间
                    k1 = self.ODE_step(latents, resolution, inverse_t, inverse_text_embeddings, inverse_guidance_scale, B)
                    k2 = self.ODE_step(latents + dt * k1, resolution, inverse_t + (dt*1000), inverse_text_embeddings, inverse_guidance_scale, B)
                    # pred_grad = 0.5 * (k1 + k2)
                    latents = latents + 0.5 * dt * (k1 + k2)
                    # 更新inverse_t
                    inverse_t += dt*1000


            elif guidance_opt.sample_method == "rk45_4":

                for dt in inverse_dt_list:
                    inverse_t = torch.tensor([min(inverse_t,1000.0)], device=self.device)
                    k1 = self.ODE_step(latents, resolution, inverse_t, inverse_text_embeddings, inverse_guidance_scale, B)
                    k2 = self.ODE_step(latents + 0.5 * dt * k1, resolution, inverse_t + 0.5 * (dt*1000), inverse_text_embeddings, inverse_guidance_scale, B)
                    k3 = self.ODE_step(latents + 0.5 * dt * k2, resolution, inverse_t + 0.5 * (dt*1000), inverse_text_embeddings, inverse_guidance_scale, B)
                    k4 = self.ODE_step(latents + dt * k3, resolution, inverse_t + (dt*1000), inverse_text_embeddings, inverse_guidance_scale, B)
                    latents = latents + (1/6) * dt * (k1 + 2*k2 + 2*k3 + k4)
                    inverse_t = inverse_t + (dt*1000)

            elif guidance_opt.sample_method == "mid_point":

                for dt in inverse_dt_list:
                    inverse_t = torch.tensor([min(inverse_t,1000.0)], device=self.device)
                    k1 = self.ODE_step(latents, resolution, inverse_t, inverse_text_embeddings, inverse_guidance_scale, B)
                    k2 = self.ODE_step(latents + 0.5 * dt * k1, resolution, inverse_t + 0.5 * (dt*1000), inverse_text_embeddings, inverse_guidance_scale, B)
                    latents = latents + dt * k2
                    inverse_t = inverse_t + (dt*1000)

            noise = latents
            latents = ori_latents

            # 第二次预测，加入cfg, 在perpneg中要考虑这里的K的reshape的问题
            sigma = self.get_sigmas(torch.tensor([t]), n_dim=latents.ndim, dtype=latents.dtype) #n_dim 代表想要输出的sigma.ndim的大小
            latents_noisy = sigma * noise + ( 1- sigma) * latents
            latent_model_input = latents_noisy[None, :, ...].repeat(2, 1, 1, 1, 1).reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, )
            tt = t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)

            unet_output = self.transformer(hidden_states = latent_model_input.to(self.precision_t), 
                                        timestep = tt.to(self.precision_t), 
                                        encoder_hidden_states= text_embeddings[0].to(self.precision_t),
                                        pooled_projections = text_embeddings[1].to(self.precision_t),
                                        return_dict=False,
                                            )[0].to(latent_model_input.dtype)
    

            unet_output = unet_output.reshape(2, -1, 16, resolution[0] // 8, resolution[1] // 8, )
            noise_pred_uncond, noise_pred_text = unet_output[:1].reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, ), unet_output[1:].reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, )
            delta_couple = noise_pred_text - noise_pred_uncond

        pred_grad = noise_pred_uncond + guidance_opt.guidance_scale * delta_couple
        
        # grad = torch.nan_to_num(grad_scale * pred_grad )
        grad = torch.nan_to_num(pred_grad - (noise - latents) ) * grad_scale

        grad = (grad).detach()
        # loss_rfds = F.mse_loss(latents - noise, target, reduction="mean") / B
        loss = SpecifyGradient.apply(latents, grad)
              
        if iteration % guidance_opt.vis_interval == 0:
            noise_pred_post = noise_pred_uncond + guidance_opt.guidance_scale* delta_couple    
            # lat2rgb = lambda x: torch.clip((x.permute(0,2,3,1) @ self.rgb_latent_factors.to(x.dtype)).permute(0,3,1,2), 0., 1.)
            save_path_iter = os.path.join(save_folder,"iter_{}_step_{}.jpg".format(iteration, int(t.item()) ))
            with torch.no_grad():

                grad_abs = torch.abs(grad.detach())
                norm_grad  = F.interpolate((grad_abs / grad_abs.max()).mean(dim=1,keepdim=True), (resolution[0], resolution[1]), mode='bilinear', align_corners=False).repeat(1,3,1,1)

                viz_images = torch.cat([pred_rgb, pred_depth.repeat(1, 3, 1, 1), 
                                        pred_alpha.repeat(1, 3, 1, 1), rgb2sat(pred_rgb, pred_alpha).repeat(1, 3, 1, 1),
                                        norm_grad,],dim=0) 

                save_image(viz_images, save_path_iter)

        self.step += 1
        return loss


    def decode_latents(self, latents):
        target_dtype = latents.dtype
        latents = latents / self.vae.config.scaling_factor

        imgs = self.vae.decode(latents.to(self.vae.dtype)).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs.to(target_dtype)

    def encode_imgs(self, imgs):
        target_dtype = imgs.dtype
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs.to(self.vae.dtype)).latent_dist
        kl_divergence = posterior.kl()

        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents.to(target_dtype), kl_divergence
    
