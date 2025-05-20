export CUDA_VISIBLE_DEVICES=0
python train_sd3.py --opt 'configs_others_2/base.yaml' --text 'A nike shoe, highly detailed, photorealistic.' \
    --init_prompt 'a shoe.' --workspace 'nike_shoe' \
    --default_radius 2.5 --radius_range "3.7 4.0" --max_radius_range "2.5 3.5" --up_dwon_shift 0.05 


export CUDA_VISIBLE_DEVICES=0
python train_sd3.py --opt 'configs_others_2/base.yaml' --text 'A LEGO car made of colorful interlocking bricks, highly detailed, photorealistic.' \
    --init_prompt 'a car.' --workspace 'LEGO_car' \
    --default_radius 2.5 --radius_range "3.2 4.0" --max_radius_range "2.5 3.5" --up_dwon_shift 0.05


export CUDA_VISIBLE_DEVICES=0
python train_sd3.py --opt 'configs_others_2/base.yaml' --text 'A Beretta 92 pistol, highly detailed, photorealistic.' \
    --init_prompt 'A handgun.' --workspace 'Beretta' \
    --default_radius 2.5 --radius_range "3.7 4.0" --max_radius_range "2.5 3.5" --up_dwon_shift 0.05 


export CUDA_VISIBLE_DEVICES=0
python train_sd3.py --opt 'configs_others_2/base_csd.yaml' --text 'Dragon, head, HDR, photorealistic, 8K.' \
    --init_prompt 'a head.' --workspace 'Dragon' 

export CUDA_VISIBLE_DEVICES=0
python train_sd3.py --opt 'configs_others_2/base.yaml' --text 'A LEGO yachts made of colorful interlocking bricks, highly detailed, photorealistic.' \
    --init_prompt 'a yachts.' --workspace 'yachts' \
    --default_radius 2.2 --radius_range "2.9 3.2" --max_radius_range "2.2 3.0" --up_dwon_shift 0.05 


export CUDA_VISIBLE_DEVICES=0
python train_sd3.py --opt 'configs_others_2/base.yaml' --text 'a cake filled with Oreos, highly detailed, photorealistic.' \
    --init_prompt 'a cake.' --workspace 'cake' \
    --default_radius 2.7 --radius_range "3.9 4.2" --max_radius_range "2.7 3.7" --up_dwon_shift 0.05 


export CUDA_VISIBLE_DEVICES=0
python train_sd3.py --opt 'configs_others_2/base.yaml' --text 'A cyberpunk cyborg, white hair, head, photorealistic, 8K, HDR.' \
    --init_prompt 'a woman head.' --workspace 'cyborg' 
