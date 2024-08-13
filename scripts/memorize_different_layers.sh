# python wanda/wanda.py --target 'memorize_0' --skill_ratio 0.01 --target_file datasets/memorize_0.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'unet' --gpu 6
# python wanda/wanda.py --target 'coco_memorize_0' --skill_ratio 0.01 --target_file datasets/memorize_0.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'unet' --gpu 6

# python wanda/wanda.py --target 'memorize_0' --skill_ratio 0.01 --target_file datasets/memorize_0.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'attn_val' --gpu 6
# python wanda/wanda.py --target 'memorize_0' --skill_ratio 0.01 --target_file datasets/memorize_0.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'text' --gpu 6
# python wanda/wanda.py --target 'memorize_1' --skill_ratio 0.01 --target_file datasets/memorize_0.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'attn_key' --gpu 6


# python wanda/save_union_over_time.py --target 'memorize_0' --skill_ratio 0.01 --target_file datasets/memorize_0.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'unet' --gpu 6 --timesteps 10 --select_ratio 0.0
# python wanda/save_union_over_time.py --target 'coco_memorize_0' --skill_ratio 0.01 --target_file datasets/memorize_0.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'unet' --gpu 6 --select_ratio 0.0 --timesteps 10
# python wanda/save_union_over_time.py --target 'memorize_0' --skill_ratio 0.01 --target_file datasets/memorize_0.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'text' --gpu 6 --select_ratio 0.0 --timesteps 1
# python wanda/save_union_over_time.py --target 'memorize_0' --skill_ratio 0.01 --target_file datasets/memorize_1.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'attn_key' --gpu 6 --select_ratio 0.0 --timesteps 10

# python benchmarking/inference_mem.py --target memorize_0 --baseline 'concept-prune' --hook_module unet --model_id 'runwayml/stable-diffusion-v1-5' --ckpt_name 'results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/memorize_0/checkpoints/random_skill_ratio_0.01_timesteps_10_threshold0.0.pt'


# python benchmarking/inference_mem.py --target memorize_0 --baseline 'concept-prune' --hook_module attn_val --model_id 'runwayml/stable-diffusion-v1-5'
# python benchmarking/inference_mem.py --target memorize_0 --baseline 'concept-prune' --hook_module text --model_id 'runwayml/stable-diffusion-v1-5'
# python benchmarking/inference_mem.py --target memorize_1 --baseline 'concept-prune' --hook_module attn_key --model_id 'runwayml/stable-diffusion-v1-5'
# python wanda/wanda.py --target 'coco_memorize_0' --skill_ratio 0.01 --target_file datasets/memorize_0.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'unet' --gpu 6
# python wanda/wanda.py --target 'coco_memorize_1' --skill_ratio 0.01 --target_file datasets/memorize_1.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'unet' --gpu 6
# python wanda/wanda.py --target 'coco_memorize_2' --skill_ratio 0.01 --target_file datasets/memorize_2.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'unet' --gpu 6
# python wanda/wanda.py --target 'coco_memorize_3' --skill_ratio 0.01 --target_file datasets/memorize_3.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'unet' --gpu 6
# python wanda/wanda.py --target 'coco_memorize_4' --skill_ratio 0.01 --target_file datasets/memorize_4.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'unet' --gpu 6

# python wanda/wanda.py --target 'memorize_0' --skill_ratio 0.01 --target_file datasets/memorize_0.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'unet' --gpu 6
# python wanda/wanda.py --target 'memorize_1' --skill_ratio 0.01 --target_file datasets/memorize_1.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'unet' --gpu 6
# python wanda/wanda.py --target 'memorize_2' --skill_ratio 0.01 --target_file datasets/memorize_2.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'unet' --gpu 6
# python wanda/wanda.py --target 'memorize_3' --skill_ratio 0.01 --target_file datasets/memorize_3.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'unet' --gpu 6
# python wanda/wanda.py --target 'memorize_4' --skill_ratio 0.01 --target_file datasets/memorize_4.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'unet' --gpu 6

# python wanda/wanda.py --target 'tv_memorize_0' --skill_ratio 0.01 --target_file datasets/tv_prompts_0.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'unet' --gpu 6
# python wanda/wanda.py --target 'tv_memorize_1' --skill_ratio 0.01 --target_file datasets/tv_prompts_1.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'unet' --gpu 6
# python wanda/wanda.py --target 'tv_memorize_2' --skill_ratio 0.01 --target_file datasets/tv_prompts_2.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'unet' --gpu 6
# python wanda/wanda.py --target 'tv_memorize_3' --skill_ratio 0.01 --target_file datasets/tv_prompts_3.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'unet' --gpu 6
# python wanda/wanda.py --target 'tv_memorize_4' --skill_ratio 0.01 --target_file datasets/tv_prompts_4.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'unet' --gpu 6

# python wanda/wanda.py --target 'mv_memorize_0' --skill_ratio 0.01 --target_file datasets/mv_prompts_0.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'unet' --gpu 6
# python wanda/wanda.py --target 'mv_memorize_1' --skill_ratio 0.01 --target_file datasets/mv_prompts_1.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'unet' --gpu 6
# python wanda/wanda.py --target 'mv_memorize_2' --skill_ratio 0.01 --target_file datasets/mv_prompts_2.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'unet' --gpu 6
# python wanda/wanda.py --target 'mv_memorize_3' --skill_ratio 0.01 --target_file datasets/mv_prompts_3.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'unet' --gpu 6
# python wanda/wanda.py --target 'mv_memorize_4' --skill_ratio 0.01 --target_file datasets/mv_prompts_4.txt --model_id 'runwayml/stable-diffusion-v1-5' --hook_module 'unet' --gpu 6