import os
import sys
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

uce_models_dict = {
    'Van Gogh': '../unified-concept-editing/models/erased-van gogh-towards_art-preserve_true-sd_1_4-method_replace.pt',
    'Monet': '../unified-concept-editing/models/erased-claude monet-towards_art-preserve_true-sd_1_4-method_replace.pt',
    'Pablo Picasso': '../unified-concept-editing/models/erased-pablo picasso-towards_art-preserve_true-sd_1_4-method_replace.pt',
    'Salvador Dali': '../unified-concept-editing/models/erased-salvador dali-towards_art-preserve_true-sd_1_4-method_replace.pt',
    'Leonardo Da Vinci': '../unified-concept-editing/models/erased-leonardo da vinci-towards_art-preserve_true-sd_1_4-method_replace.pt',
}

concept_ablation_dict = {
    'Van Gogh': '../concept-ablation/diffusers/logs_ablation/vangogh/delta.bin',
    'Monet': '../concept-ablation/diffusers/logs_ablation/vangogh/delta.bin',
    'Pablo Picasso': '../concept-ablation/diffusers/logs_ablation/vangogh/delta.bin',
    'Salvador Dali': '../concept-ablation/diffusers/logs_ablation/vangogh/delta.bin',
    'Leonardo Da Vinci': '../concept-ablation/diffusers/logs_ablation/vangogh/delta.bin',
}

esd_models_dict = {
    'Van Gogh': '../erasing/models/diffusers-VanGogh-ESDx1-UNET.pt',
    'Monet': '../erasing/models/compvis-word_ClaudeMonet-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05/diffusers-word_ClaudeMonet-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05.pt',
    'Pablo Picasso': '../erasing/models/compvis-word_PabloPicasso-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05/diffusers-word_PabloPicasso-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05.pt',
    'Salvador Dali': '../erasing/models/compvis-word_SalvadorDali-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05/diffusers-word_SalvadorDali-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05.pt',
    'Leonardo Da Vinci': '../erasing/models/compvis-word_LeonardoDaVinci-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05/diffusers-word_LeonardoDaVinci-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05.pt',
}

best_ckpt_dict = {
    'Van Gogh': "results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Van Gogh/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'Monet': "results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Monet/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'Pablo Picasso': "results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Pablo Picasso/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'Salvador Dali': "results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Salvador Dali/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'Leonardo Da Vinci': "results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Leonardo Da Vinci/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'naked': 'results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/naked/checkpoints/skill_ratio_0.01_timesteps_9_threshold0.0.pt',
    'parachute': 'results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/parachute/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'cassette player': 'results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/cassette player/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'golf ball': 'results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/golf ball/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'french horn': 'results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/french horn/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'garbage truck': 'results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/garbage truck/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'chain saw': 'results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/chain saw/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'english springer': 'results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/english springer/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'tench': 'results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/tench/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'gas pump': 'results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/gas pump/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'church': 'results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/church/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'female': "results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/female/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'male': "results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/male/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'memorize_0': 'results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/memorize_0/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt',
}

best_ckpt_dict_text = {
    'Van Gogh': "results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Van Gogh/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt",
    'Monet': "results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Monet/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt",
    'Pablo Picasso': "results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Pablo Picasso/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt",
    'Salvador Dali': "results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Salvador Dalih/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt",
    'Leonardo Da Vinci': "results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Leonardo Da Vinci/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt",
    'naked': 'results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/naked/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt',
    'parachute': 'results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/parachute/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt',
    'cassette player': 'results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/cassette player/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt',
    'golf ball': 'results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/golf ball/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt',
    'french horn': 'results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/french horn/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt',
    'garbage truck': 'results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/garbage truck/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt',
    'chain saw': 'results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/chain saw/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt',
    'english springer': 'results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/english springer/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt',
    'tench': 'results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/tench/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt',
    'gas pump': 'results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/gas pump/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt',
    'female': "results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/female/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt",
    'male': "results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/male/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt",
    'memorize_0': 'results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/memorize_0/checkpoints/skill_ratio_0.01_timesteps_1_threshold0.0.pt',
}


best_ckpt_dict_ffn_1 = {
    'Van Gogh': "results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Van Gogh/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'Monet': "results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Monet/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt",
    'Pablo Picasso': "results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Pablo Picasso/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt",
    'Salvador Dali': "results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Salvador Dali/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt",
    'Leonardo Da Vinci': "results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Leonardo Da Vinci/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt",
    'naked': 'results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/naked/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'parachute': 'results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/parachute/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'cassette player': 'results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/cassette player/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'golf ball': 'results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/golf ball/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'french horn': 'results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/french horn/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'garbage truck': 'results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/garbage truck/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'chain saw': 'results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/chain saw/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'english springer': 'results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/english springer/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'tench': 'results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/tench/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'gas pump': 'results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/gas pump/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'female': "results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/female/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'male': "results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/male/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'memorize_0': 'results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/memorize_0/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt',
}

best_ckpt_dict_attn_key = {
    'Van Gogh': "results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Van Gogh/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'Monet': "results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Monet/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'Pablo Picasso': "results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Pablo Picasso/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'Salvador Dali': "results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Salvador Dali/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'Leonardo Da Vinci': "results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Leonardo Da Vinci/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'naked': 'results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/naked/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt',
    'parachute': 'results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/parachute/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'cassette player': 'results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/cassette player/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'golf ball': 'results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/golf ball/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'french horn': 'results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/french horn/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'garbage truck': 'results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/garbage truck/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'chain saw': 'results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/chain saw/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'english springer': 'results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/english springer/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'tench': 'results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/tench/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'gas pump': 'results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/gas pump/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'female': "results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/female/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'male': "results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/male/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'memorize_0': 'results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/memorize_0/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt',
}

best_ckpt_dict_attn_val = {
    'Van Gogh': "results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Van Gogh/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'Monet': "results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Monet/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt",
    'Pablo Picasso': "results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Pablo Picasso/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt",
    'Salvador Dali': "results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Salvador Dali/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt",
    'Leonardo Da Vinci': "results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Leonardo Da Vinci/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt",
    'naked': 'results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/naked/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'parachute': 'results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/parachute/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'cassette player': 'results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/cassette player/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'golf ball': 'results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/golf ball/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'french horn': 'results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/french horn/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'garbage truck': 'results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/garbage truck/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'chain saw': 'results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/chain saw/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'english springer': 'results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/english springer/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'tench': 'results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/tench/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'gas pump': 'results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/gas pump/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'female': "results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/female/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'male': "results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/male/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'memorize_0': 'results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/memorize_0/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt',
}

all_models_dict = {
    'uce': uce_models_dict,
    'concept-ablation': concept_ablation_dict,
    'esd': esd_models_dict,
    'concept-prune': best_ckpt_dict
}


def load_models(args, ckpt_name=None):

    if ckpt_name is None:
        if args.hook_module == 'text':
            all_models_dict['concept-prune'] = best_ckpt_dict_text
        elif args.hook_module == 'unet-ffn-1':
            all_models_dict['concept-prune'] = best_ckpt_dict_ffn_1
        elif args.hook_module == 'attn_key':
            all_models_dict['concept-prune'] = best_ckpt_dict_attn_key
        elif args.hook_module == 'attn_val':
            all_models_dict['concept-prune'] = best_ckpt_dict_attn_val
    else:
        all_models_dict['concept-prune'] = {args.target: ckpt_name}

    print(f"Loading model from {all_models_dict[args.baseline][args.target]}")


    if args.hook_module in ['unet', 'unet-ffn-1', 'attn_key', 'attn_val']:
        if args.baseline in ['uce', 'esd']:
            # load a baseline model and fine tune it
            print(f"Loading model from {all_models_dict[args.baseline][args.target]}")
            unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet", torch_dtype=torch.float16)
            unet.load_state_dict(torch.load(all_models_dict[args.baseline][args.target]))
            remover_model = StableDiffusionPipeline.from_pretrained(args.model_id, unet=unet, torch_dtype=torch.float16)
            remover_model = remover_model.to(args.gpu)
        elif args.baseline == 'concept-prune':
            unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet", torch_dtype=torch.float16)
            unet.load_state_dict(torch.load(all_models_dict[args.baseline][args.target]))
            remover_model = StableDiffusionPipeline.from_pretrained(args.model_id, unet=unet, torch_dtype=torch.float16)
            remover_model = remover_model.to(args.gpu)
        elif args.baseline == 'concept-ablation':
            remover_model = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
            remover_model = remover_model.to(args.gpu)
            model_path = os.path.join(all_models_dict[args.baseline][args.target])
            print(f"Loading model from {model_path}")
            model_ckpt = torch.load(model_path)
            if 'text_encoder' in model_ckpt:
                remover_model.text_encoder.load_state_dict(model_ckpt['text_encoder'])
            for name, params in remover_model.unet.named_parameters():
                if name in model_ckpt['unet']:
                    params.data.copy_(model_ckpt['unet'][f'{name}'])
        elif args.baseline == 'baseline':
            remover_model = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
            remover_model = remover_model.to(args.gpu)
        else:
            raise ValueError(f"Invalid baseline: {args.baseline}")
    elif args.hook_module == 'text':
        # only concept-prune is supported for erasing wuth text encoder editing
        remover_model = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)
        remover_model.text_encoder.load_state_dict(torch.load(all_models_dict[args.baseline][args.target]))
        remover_model = remover_model.to(args.gpu)

    return remover_model