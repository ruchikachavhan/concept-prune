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
    'Monet': '../concept-ablation/diffusers/logs_ablation/monet/delta.bin',
    'Pablo Picasso': '../concept-ablation/diffusers/logs_ablation/picasso/delta.bin',
    'Salvador Dali': '../concept-ablation/diffusers/logs_ablation/salvador_dali/delta.bin',
    'Leonardo Da Vinci': '../concept-ablation/diffusers/logs_ablation/davinci/delta.bin',
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
    'Monet': "results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Van Gogh/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'Pablo Picasso': "results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Van Gogh/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'Salvador Dali': "results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Van Gogh/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'Leonardo Da Vinci': "results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Van Gogh/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'naked': 'results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/naked/checkpoints/skill_ratio_0.01_timesteps_9_threshold0.0.pt',
    'parachute': 'results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/parachute/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt',
    'female': "results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/female/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'male': "results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/male/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt"
}

all_models_dict = {
    'uce': uce_models_dict,
    'concept-ablation': concept_ablation_dict,
    'esd': esd_models_dict,
    'concept-prune': best_ckpt_dict
}


def load_models(args):

    if args.baseline in ['uce', 'esd']:
        # load a baseline model and fine tune it
        unet = UNet2DConditionModel.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="unet", torch_dtype=torch.float16)
        unet.load_state_dict(torch.load(all_models_dict[args.baseline][args.target]))
        remover_model = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', unet=unet, torch_dtype=torch.float16)
        remover_model = remover_model.to(args.gpu)
    elif args.baseline == 'concept-prune':
        unet = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="unet", torch_dtype=torch.float16)
        unet.load_state_dict(torch.load(all_models_dict[args.baseline][args.target]))
        remover_model = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', unet=unet, torch_dtype=torch.float16)
        remover_model = remover_model.to(args.gpu)
    elif args.baseline == 'concept-ablation':
        remover_model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        remover_model = remover_model.to(args.gpu)
        model_path = os.path.join(all_models_dict[args.baseline][args.target])
        print(f"Loading model from {model_path}")
        model_ckpt = torch.load(model_path)
        if 'text_encoder' in model_ckpt:
            remover_model.text_encoder.load_state_dict(model_ckpt['text_encoder'])
        for name, params in remover_model.unet.named_parameters():
            if name in model_ckpt['unet']:
                params.data.copy_(model_ckpt['unet'][f'{name}'])
    else:
        raise ValueError(f"Invalid baseline: {args.baseline}")

    return remover_model