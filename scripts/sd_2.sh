# python wanda/wanda.py --target 'Van Gogh' --skill_ratio 0.01 --model_id stabilityai/stable-diffusion-2
# python wanda/wanda.py --target 'Monet' --skill_ratio 0.01 --model_id stabilityai/stable-diffusion-2
# python wanda/wanda.py --target 'Pablo Picasso' --skill_ratio 0.01 --model_id stabilityai/stable-diffusion-2
# python wanda/wanda.py --target 'Leonardo Da Vinci' --skill_ratio 0.01 --model_id stabilityai/stable-diffusion-2
# python wanda/wanda.py --target 'Salvador Dali' --skill_ratio 0.01 --model_id stabilityai/stable-diffusion-2


# python wanda/save_union_over_time.py --target 'Van Gogh' --skill_ratio 0.01 --timesteps 10 --select_ratio 0.0 --model_id stabilityai/stable-diffusion-2
# python wanda/save_union_over_time.py --target 'Monet' --skill_ratio 0.01 --timesteps 10 --select_ratio 0.0 --model_id stabilityai/stable-diffusion-2
# python wanda/save_union_over_time.py --target 'Pablo Picasso' --skill_ratio 0.01 --timesteps 10 --select_ratio 0.0 --model_id stabilityai/stable-diffusion-2
# python wanda/save_union_over_time.py --target 'Leonardo Da Vinci' --skill_ratio 0.01 --timesteps 10 --select_ratio 0.0 --model_id stabilityai/stable-diffusion-2
# python wanda/save_union_over_time.py --target 'Salvador Dali' --skill_ratio 0.01 --timesteps 10 --select_ratio 0.0 --model_id stabilityai/stable-diffusion-2

#  python benchmarking/artist_erasure.py --target "Van Gogh" --hook_module 'unet' --baseline 'concept-prune'  --model_id 'stabilityai/stable-diffusion-2' --gpu 1 --ckpt_name "results/results_seed_0/stable-diffusion/stabilityai/stable-diffusion-2/Van Gogh/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt"
# python benchmarking/artist_erasure.py --target "Monet" --hook_module 'unet' --baseline 'concept-prune'  --model_id 'stabilityai/stable-diffusion-2' --gpu 1 --ckpt_name "results/results_seed_0/stable-diffusion/stabilityai/stable-diffusion-2/Monet/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt"
# python benchmarking/artist_erasure.py --target "Pablo Picasso" --hook_module 'unet' --baseline 'concept-prune'  --model_id 'stabilityai/stable-diffusion-2' --gpu 1 --ckpt_name "results/results_seed_0/stable-diffusion/stabilityai/stable-diffusion-2/Pablo Picasso/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt"
# python benchmarking/artist_erasure.py --target "Salvador Dali" --hook_module 'unet' --baseline 'concept-prune'  --model_id 'stabilityai/stable-diffusion-2' --gpu 1 --ckpt_name "results/results_seed_0/stable-diffusion/stabilityai/stable-diffusion-2/Salvador Dali/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt"
# python benchmarking/artist_erasure.py --target "Leonardo Da Vinci" --hook_module 'unet' --baseline 'concept-prune'  --model_id 'stabilityai/stable-diffusion-2' --gpu 1 --ckpt_name "results/results_seed_0/stable-diffusion/stabilityai/stable-diffusion-2/Leonardo Da Vinci/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt"


python benchmarking/artist_erasure.py --target "Leonardo Da Vinci" --hook_module 'unet' --baseline 'uce'  --model_id 'stabilityai/stable-diffusion-2-1-base' --gpu 1 --ckpt_name '../unified-concept-editing/models/erased-leonardo da vinci-towards_art-preserve_true-sd_2_1-method_replace.pt'
python benchmarking/artist_erasure.py --target "Pablo Picasso" --hook_module 'unet' --baseline 'uce'  --model_id 'stabilityai/stable-diffusion-2-1-base' --gpu 1 --ckpt_name '../unified-concept-editing/models/erased-pablo picasso-towards_art-preserve_true-sd_2_1-method_replace.pt'
python benchmarking/artist_erasure.py --target "Van Gogh" --hook_module 'unet' --baseline 'uce'  --model_id 'stabilityai/stable-diffusion-2-1-base' --gpu 1 --ckpt_name '../unified-concept-editing/models/erased-van gogh-towards_art-preserve_true-sd_2_1-method_replace.pt'
python benchmarking/artist_erasure.py --target "Salvador Dali" --hook_module 'unet' --baseline 'uce'  --model_id 'stabilityai/stable-diffusion-2-1-base' --gpu 1 --ckpt_name '../unified-concept-editing/models/erased-salvador dali-towards_art-preserve_true-sd_2_1-method_replace.pt'
python benchmarking/artist_erasure.py --target "Monet" --hook_module 'unet' --baseline 'uce'  --model_id 'stabilityai/stable-diffusion-2-1-base' --gpu 1 --ckpt_name '../unified-concept-editing/models/erased-monet-towards_art-preserve_true-sd_2_1-method_replace.pt'