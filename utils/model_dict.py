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
}

all_models_dict = {
    'uce': uce_models_dict,
    'concept-ablation': concept_ablation_dict,
    'esd': esd_models_dict,
    'concept-prune': best_ckpt_dict
}