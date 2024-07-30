python wanda/wanda.py --target 'Monet' --skill_ratio 0.02 
python wanda/wanda.py --target 'Pablo Picasso' --skill_ratio 0.02
python wanda/wanda.py --target 'Leonardo Da Vinci' --skill_ratio 0.02
python wanda/wanda.py --target 'Salvador Dali' --skill_ratio 0.02
python wanda/wanda.py --target 'parachute' --skill_ratio 0.02
python wanda/wanda.py --target 'cassette player' --skill_ratio 0.02
python wanda/wanda.py --target 'golf ball' --skill_ratio 0.02
python wanda/wanda.py --target 'gas pump' --skill_ratio 0.02
python wanda/wanda.py --target 'english springer' --skill_ratio 0.02
python wanda/wanda.py --target 'tench' --skill_ratio 0.02
python wanda/wanda.py --target 'chain saw' --skill_ratio 0.02
python wanda/wanda.py --target 'french horn' --skill_ratio 0.02
python wanda/wanda.py --target 'parachute' --skill_ratio 0.02
python wanda/wanda.py --target 'cassette player' --skill_ratio 0.02
python wanda/wanda.py --target 'golf ball' --skill_ratio 0.02
python wanda/wanda.py --target 'gas pump' --skill_ratio 0.02
python wanda/wanda.py --target 'english springer' --skill_ratio 0.02
python wanda/wanda.py --target 'tench' --skill_ratio 0.02
python wanda/wanda.py --target 'chain saw' --skill_ratio 0.02
python wanda/wanda.py --target 'french horn' --skill_ratio 0.02
python wanda/wanda.py --target 'naked' --skill_ratio 0.02

python wanda/save_union_over_time.py --target 'Monet' --skill_ratio 0.02 --timesteps 10 --select_ratio 0.0
python wanda/save_union_over_time.py --target 'Pablo Picasso' --skill_ratio 0.02 --timesteps 10 --select_ratio 0.0
python wanda/save_union_over_time.py --target 'Leonardo Da Vinci' --skill_ratio 0.02 --timesteps 10 --select_ratio 0.0
python wanda/save_union_over_time.py --target 'Salvador Dali' --skill_ratio 0.02 --timesteps 10 --select_ratio 0.0
python wanda/save_union_over_time.py --target 'parachute' --skill_ratio 0.02 --timesteps 10 --select_ratio 0.0
python wanda/save_union_over_time.py --target 'cassette player' --skill_ratio 0.02 --timesteps 10 --select_ratio 0.0
python wanda/save_union_over_time.py --target 'golf ball' --skill_ratio 0.02 --timesteps 10 --select_ratio 0.0
python wanda/save_union_over_time.py --target 'gas pump' --skill_ratio 0.02 --timesteps 10 --select_ratio 0.0
python wanda/save_union_over_time.py --target 'english springer' --skill_ratio 0.02 --timesteps 10 --select_ratio 0.0
python wanda/save_union_over_time.py --target 'tench' --skill_ratio 0.02 --timesteps 10 --select_ratio 0.0
python wanda/save_union_over_time.py --target 'chain saw' --skill_ratio 0.02 --timesteps 10 --select_ratio 0.0
python wanda/save_union_over_time.py --target 'french horn' --skill_ratio 0.02 --timesteps 10 --select_ratio 0.0
python wanda/save_union_over_time.py --target 'naked' --skill_ratio 0.02 --timesteps 10 --select_ratio 0.0

python benchmarking/artist_erasure.py --target "Van Gogh" --hook_module 'unet-ffn-1' --baseline 'concept-prune' --benchmarking_result_path "results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/" --gpu 1
python benchmarking/artist_erasure.py --target "Monet" --hook_module 'unet-ffn-1' --baseline 'concept-prune' --benchmarking_result_path "results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/" --gpu 1
python benchmarking/artist_erasure.py --target "Pablo Picasso" --hook_module 'unet-ffn-1' --baseline 'concept-prune' --benchmarking_result_path "results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/"
python benchmarking/artist_erasure.py --target "Leonardo Da Vinci" --hook_module 'unet-ffn-1' --baseline 'concept-prune' --benchmarking_result_path "results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/"
python benchmarking/artist_erasure.py --target "Salvador Dali" --hook_module 'unet-ffn-1' --baseline 'concept-prune' --benchmarking_result_path "results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/"
python benchmarking/object_erase.py --target "parachute" --hook_module 'unet-ffn-1' --baseline 'concept-prune' --benchmarking_result_path "results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/" --removal_mode erase
python benchmarking/object_erase.py --target "cassette player" --hook_module 'unet-ffn-1' --baseline 'concept-prune' --benchmarking_result_path "results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/" --removal_mode erase
python benchmarking/object_erase.py --target "golf ball" --hook_module 'unet-ffn-1' --baseline 'concept-prune' --benchmarking_result_path "results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/" --removal_mode erase
python benchmarking/object_erase.py --target "gas pump" --hook_module 'unet-ffn-1' --baseline 'concept-prune' --benchmarking_result_path "results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/" --removal_mode erase
python benchmarking/object_erase.py --target "english springer" --hook_module 'unet-ffn-1' --baseline 'concept-prune' --benchmarking_result_path "results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/" --removal_mode erase
python benchmarking/object_erase.py --target "tench" --hook_module 'unet-ffn-1' --baseline 'concept-prune' --benchmarking_result_path "results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/" --removal_mode erase
python benchmarking/object_erase.py --target "chain saw" --hook_module 'unet-ffn-1' --baseline 'concept-prune' --benchmarking_result_path "results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/" --removal_mode erase
python benchmarking/object_erase.py --target "french horn" --hook_module 'unet-ffn-1' --baseline 'concept-prune' --benchmarking_result_path "results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/" --removal_mode erase