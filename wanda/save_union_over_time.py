import os
import json
import scipy
import pickle
import torch
import sys
import tqdm
import numpy as np
from argparse import ArgumentParser
sys.path.append(os.getcwd())
from utils import get_prompts, Config, get_sd_model
from neuron_receivers import Wanda
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

def input_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--dbg', type=bool, default=None)
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--base', type=str, default=None)
    parser.add_argument('--skill_ratio', type=float, default=None)
    parser.add_argument('--timesteps', type=int, default=None)
    parser.add_argument('--select_ratio', type=float, default=None)
    return parser.parse_args()


def main():
    args = Config('configs/wanda_config.yaml')
    cmd_args = input_args()
    # iterate over the args and update the config
    for key, value in vars(cmd_args).items():
        if value is not None:
            print(f"Updating {key} with {value}")
            setattr(args, key, value)
    args.configure()
    

    # load pipeline
    model = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
    model = model.to('cuda')

    # get names of layers
    weights_shape = {}
    for name, module in model.unet.named_modules():
        if isinstance(module, torch.nn.Linear) and 'ff.net' in name and not 'proj' in name:
            weights_shape[name] = module.weight.shape

    # sort keys in alphabetical order and ensure that the order is consistent
    weights_shape = dict(sorted(weights_shape.items()))
    layer_names = list(weights_shape.keys())
    print("Layer names: ", layer_names)
    print("Weights shape: ", weights_shape)
    masks = {}
    union_concepts = {}
    path = args.skilled_neuron_path

    for l in range(args.n_layers):
        union_concepts[layer_names[l]] = np.zeros(weights_shape[layer_names[l]])
        union_concepts[layer_names[l]] = scipy.sparse.csr_matrix(union_concepts[layer_names[l]])
        for t in range(0, args.timesteps):
            with open(os.path.join(path, f'timestep_{t}_layer_{l}.pkl'), 'rb') as f:
                # load sparse matrix
                indices = pickle.load(f)
                # take union
                # out of the sparse matrix, only select 50% elements that are 1
                indices = indices.toarray()
                union_concepts[layer_names[l]] += scipy.sparse.csr_matrix(indices)
        union_concepts[layer_names[l]] = union_concepts[layer_names[l]] > (args.select_ratio * args.timesteps)
        array = union_concepts[layer_names[l]].astype('bool').astype('int')
        array = array.toarray()
        print("Layer: ", l, layer_names[l], "Density of skilled neurons: ", np.mean(array))
        masks[layer_names[l]] = array
    
    # apply masks to the model weights and save the model
    print("Applying masks to the model")
    for name, module in model.unet.named_modules():
        if isinstance(module, torch.nn.Linear) and 'ff.net' in name and not 'proj' in name:
            weight = module.weight.data.clone().detach().cpu()
            weight *= (1- masks[name])
            weight = weight.to(torch.float16)
            module.weight.data = weight

    # save the model
    ckpt_name = os.path.join(args.checkpoint_path, f'skill_ratio_{args.skill_ratio}_timesteps_{args.timesteps}_threshold{args.select_ratio}.pt')
    torch.save(model.unet.state_dict(), ckpt_name)

    print("Model saved at: ", ckpt_name)
    del model
    # test the model

    _, target_prompts = get_prompts(args)

    # load the masked model
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet", torch_dtype=torch.float16)
    unet.load_state_dict(torch.load(ckpt_name))
    model = StableDiffusionPipeline.from_pretrained(args.model_id, unet=unet, torch_dtype=torch.float16)
    model = model.to('cuda')

    # test new model on the target prompts
    for ann_target in target_prompts:
        print("Testing on target prompt: ", ann_target)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        out = model(ann_target).images[0]
        out.save(os.path.join(args.checkpoint_path, 'test_images', f'{ann_target}.png'))


if __name__ == '__main__':
    main()

