# python wanda/wanda.py --target 'Van Gogh' --skill_ratio 0.01
# python wanda/wanda.py --target 'Monet' --skill_ratio 0.01
# python wanda/wanda.py --target 'Pablo Picasso' --skill_ratio 0.01
# python wanda/wanda.py --target 'Leonardo Da Vinci' --skill_ratio 0.01
# python wanda/wanda.py --target 'Salvador Dali' --skill_ratio 0.01
# python wanda/wanda.py --target 'parachute' --skill_ratio 0.02
# python wanda/wanda.py --target 'cassette player' --skill_ratio 0.02
# python wanda/wanda.py --target 'golf ball' --skill_ratio 0.02
# python wanda/wanda.py --target 'gas pump' --skill_ratio 0.02
# python wanda/wanda.py --target 'english springer' --skill_ratio 0.02
# python wanda/wanda.py --target 'tench' --skill_ratio 0.02
# python wanda/wanda.py --target 'chain saw' --skill_ratio 0.02
# python wanda/wanda.py --target 'french horn' --skill_ratio 0.02
# python wanda/wanda.py --target 'parachute' --skill_ratio 0.01
# python wanda/wanda.py --target 'cassette player' --skill_ratio 0.01
# python wanda/wanda.py --target 'golf ball' --skill_ratio 0.01
# python wanda/wanda.py --target 'gas pump' --skill_ratio 0.01
# python wanda/wanda.py --target 'english springer' --skill_ratio 0.01
# python wanda/wanda.py --target 'tench' --skill_ratio 0.01
# python wanda/wanda.py --target 'chain saw' --skill_ratio 0.01
# python wanda/wanda.py --target 'french horn' --skill_ratio 0.01
# python wanda/wanda.py --target 'church' --skill_ratio 0.02
# python wanda/wanda.py --target 'garbage truck' --skill_ratio 0.02
python wanda/wanda.py --target 'cluster_14' --skill_ratio 0.01
python wanda/wanda.py --target 'cluster_5' --skill_ratio 0.01
python wanda/wanda.py --target 'cluster_15' --skill_ratio 0.01
# python wanda/save_union_over_time.py --target 'Monet' --skill_ratio 0.01 --timesteps 10 --select_ratio 0.0
# python wanda/save_union_over_time.py --target 'Pablo Picasso' --skill_ratio 0.01 --timesteps 10 --select_ratio 0.0
# python wanda/save_union_over_time.py --target 'Leonardo Da Vinci' --skill_ratio 0.01 --timesteps 10 --select_ratio 0.0
# python wanda/save_union_over_time.py --target 'Salvador Dali' --skill_ratio 0.01 --timesteps 10 --select_ratio 0.0
# python wanda/save_union_over_time.py --target 'parachute' --skill_ratio 0.02 --timesteps 10 --select_ratio 0.0
# python wanda/save_union_over_time.py --target 'cassette player' --skill_ratio 0.02 --timesteps 10 --select_ratio 0.0
# python wanda/save_union_over_time.py --target 'golf ball' --skill_ratio 0.02 --timesteps 10 --select_ratio 0.0
# python wanda/save_union_over_time.py --target 'gas pump' --skill_ratio 0.02 --timesteps 10 --select_ratio 0.0
# python wanda/save_union_over_time.py --target 'english springer' --skill_ratio 0.02 --timesteps 10 --select_ratio 0.0
# python wanda/save_union_over_time.py --target 'tench' --skill_ratio 0.02 --timesteps 10 --select_ratio 0.0
# python wanda/save_union_over_time.py --target 'chain saw' --skill_ratio 0.02 --timesteps 10 --select_ratio 0.0
# python wanda/save_union_over_time.py --target 'french horn' --skill_ratio 0.02 --timesteps 10 --select_ratio 0.0
# python wanda/save_union_over_time.py --target 'parachute' --skill_ratio 0.01 --timesteps 10 --select_ratio 0.0
# python wanda/save_union_over_time.py --target 'cassette player' --skill_ratio 0.01 --timesteps 10 --select_ratio 0.0
# python wanda/save_union_over_time.py --target 'golf ball' --skill_ratio 0.01 --timesteps 10 --select_ratio 0.0
# python wanda/save_union_over_time.py --target 'gas pump' --skill_ratio 0.01 --timesteps 10 --select_ratio 0.0
# python wanda/save_union_over_time.py --target 'english springer' --skill_ratio 0.01 --timesteps 10 --select_ratio 0.0
# python wanda/save_union_over_time.py --target 'tench' --skill_ratio 0.01 --timesteps 10 --select_ratio 0.0
# python wanda/save_union_over_time.py --target 'chain saw' --skill_ratio 0.01 --timesteps 10 --select_ratio 0.0
# python wanda/save_union_over_time.py --target 'church' --skill_ratio 0.02 --timesteps 10 --select_ratio 0.0
# python wanda/save_union_over_time.py --target 'garbage truck' --skill_ratio 0.02 --timesteps 10 --select_ratio 0.0

# python benchmarking/object_erase.py --target parachute --baseline concept-prune --removal_mode erase
# python benchmarking/object_erase.py --target 'cassette player' --baseline concept-prune --removal_mode erase
# python benchmarking/object_erase.py --target 'golf ball' --baseline concept-prune --removal_mode erase
# python benchmarking/object_erase.py --target 'gas pump' --baseline concept-prune --removal_mode erase
# python benchmarking/object_erase.py --target 'english springer' --baseline concept-prune --removal_mode erase
# python benchmarking/object_erase.py --target 'tench' --baseline concept-prune --removal_mode erase
# python benchmarking/object_erase.py --target 'chain saw' --baseline concept-prune --removal_mode erase
# python benchmarking/object_erase.py --target 'french horn' --baseline concept-prune --removal_mode erase

# python benchmarking/object_erase.py --target church --baseline concept-prune --removal_mode erase --ckpt_name 'results/results_seed_0/stable-diffusion/CompVis/stable-diffusion-v1-4/church/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt'
# python benchmarking/object_erase.py --target 'garbage truck' --baseline concept-prune --removal_mode erase --ckpt_name 'results/results_seed_0/stable-diffusion/CompVis/stable-diffusion-v1-4/garbage truck/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt'

# python benchmarking/object_erase.py --target church --baseline concept-prune --removal_mode keep --ckpt_name 'results/results_seed_0/stable-diffusion/CompVis/stable-diffusion-v1-4/church/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt'
# python benchmarking/object_erase.py --target 'garbage truck' --baseline concept-prune --removal_mode keep --ckpt_name 'results/results_seed_0/stable-diffusion/CompVis/stable-diffusion-v1-4/garbage truck/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt'

# python benchmarking/object_erase.py --target parachute --baseline concept-prune --removal_mode keep
# python benchmarking/object_erase.py --target 'cassette player' --baseline concept-prune --removal_mode keep
# python benchmarking/object_erase.py --target 'golf ball' --baseline concept-prune --removal_mode keep
# python benchmarking/object_erase.py --target 'gas pump' --baseline concept-prune --removal_mode keep
# python benchmarking/object_erase.py --target 'english springer' --baseline concept-prune --removal_mode keep
# python benchmarking/object_erase.py --target 'tench' --baseline concept-prune --removal_mode keep
# python benchmarking/object_erase.py --target 'chain saw' --baseline concept-prune --removal_mode keep
# python benchmarking/object_erase.py --target 'french horn' --baseline concept-prune --removal_mode keep


# python benchmarking/artist_erasure.py --target "Van Gogh" --hook_module 'text' --baseline 'concept-prune' --benchmarking_result_path "results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/"
# python benchmarking/artist_erasure.py --target "Van Gogh" --hook_module unet-ffn-1 --baseline 'concept-prune' --benchmarking_result_path "results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/"