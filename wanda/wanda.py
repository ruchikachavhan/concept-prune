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
from diffusers.models.activations import LoRACompatibleLinear
from transformers.models.clip.modeling_clip import CLIPMLP

def input_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--dbg', type=bool, default=None)
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--base', type=str, default=None)
    parser.add_argument('--skill_ratio', type=float, default=None)
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

    print("Arguments: ", args.__dict__)
    base_prompts, target_prompts = get_prompts(args)
    print("Base prompts: ", base_prompts)
    print("Target prompts: ", target_prompts)

    # Model
    model, num_layers, replace_fn = get_sd_model(args)
    args.replace_fn = replace_fn
    print("Replace fn: ", replace_fn)
    model = model.to(args.gpu)
    print("Model: ", model)

    # get the absolute value of FFN weights in the second layer
    abs_weights = {}
    layer_names = []

    if args.hook_module == 'unet':
        for name, module in model.unet.named_modules():
            if isinstance(module, LoRACompatibleLinear) and 'ff.net' in name and not 'proj' in name:
                layer_names.append(name)
                weight = module.weight.detach()
                abs_weights[name] = weight.abs().cpu()
                print("Storing absolute value of: ", name, module.weight.shape)

    elif args.hook_module == 'text':
        for name, module in model.text_encoder.named_modules():
            if isinstance(module, CLIPMLP) and 'mlp' in name and 'encoder.layers' in name:
                layer_names.append(name)
                weight = module.fc2.weight.detach().clone()
                abs_weights[name] = weight.abs().cpu()
                print("Storing absolute value of: ", name, module.weight.shape)
    # sort the layer names so that mid block is before up block
    layer_names.sort()
    print("Layer names: ", layer_names, len(layer_names))
               
    # Make two separate norm calculator classes for base and adj prompts
    neuron_receiver_base = Wanda(args.seed, args.timesteps, num_layers, replace_fn = args.replace_fn, keep_nsfw = args.keep_nsfw, hook_module=args.hook_module)
    neuron_receiver_target = Wanda(args.seed, args.timesteps, num_layers, replace_fn = args.replace_fn, keep_nsfw = args.keep_nsfw, hook_module=args.hook_module)

    if not os.path.exists(os.path.join(args.res_path, 'base_norms.pt')):
        # Saving norm values
        iter = 0
        for ann, ann_target in tqdm.tqdm(zip(base_prompts, target_prompts)):
            if iter >= 3 and args.dbg:
                break
            print("text: ", ann, ann_target)
           
            neuron_receiver_base.reset_time_layer()
            out = neuron_receiver_base.observe_activation(model, ann)

            neuron_receiver_target.reset_time_layer()
            out_target = neuron_receiver_target.observe_activation(model, ann_target)

            # save images
            if iter < 5:
                print("Saving images", os.path.join(args.img_save_path, f'base_{iter}.jpg'))
                out.save(os.path.join(args.img_save_path, f'base_{iter}.jpg'))
                out_target.save(os.path.join(args.img_save_path, f'target_{iter}.jpg'))
            
            iter += 1
        
        # get the norms
        if args.hook_module == 'unet':
            act_norms_base = neuron_receiver_base.activation_norm.get_column_norms()
            act_norms_target = neuron_receiver_target.activation_norm.get_column_norms()
            # save
            neuron_receiver_base.activation_norm.save(os.path.join(args.res_path, 'base_norms.pt'))
            neuron_receiver_target.activation_norm.save(os.path.join(args.res_path, 'target_norms.pt'))
        elif args.hook_module == 'text':
            # fix timesteps to 1 because for text encoder, we do only one forward pass, hack for loading timestep wise 
            args.timesteps = 1
            act_norms_base, act_norms_target = {}, {}
            for t in range(args.timesteps):
                act_norms_base[t] = {}
                act_norms_target[t] = {}
                for l in range(num_layers):
                    act_norms_base[t][l] = neuron_receiver_base.activation_norm[l].get_column_norms()
                    act_norms_target[t][l] = neuron_receiver_target.activation_norm[l].get_column_norms()

            # save
            torch.save(act_norms_base, os.path.join(args.res_path, 'base_norms.pt'))
            torch.save(act_norms_target, os.path.join(args.res_path, 'target_norms.pt'))
        
    else:
        act_norms_base = torch.load(os.path.join(args.res_path, 'base_norms.pt'))
        act_norms_target = torch.load(os.path.join(args.res_path, 'target_norms.pt'))


    sparsity_ratio = args.skill_ratio
    # hack for the score calculation to be the same
    timesteps = 1 if args.hook_module == 'text' else args.timesteps

    for t in range(timesteps):
        for l in range(num_layers):
            print("Time step: ", t, "Layer: ", l)
            print("Base norm shape: ", act_norms_base[t][l].shape)
            print("Target norm shape: ", act_norms_target[t][l].shape)
            print("Weight shape: ", abs_weights[layer_names[l]].shape)

            # wanda score is W.abs() * A
            metric_base = abs_weights[layer_names[l]] * act_norms_base[t][l]
            metric_target = abs_weights[layer_names[l]] * act_norms_target[t][l]

            # check for inf values here
            if torch.isinf(metric_base).any():
                print("Inf values in metric base")

            # do row-wise sorting for base in descending order
            _, sorted_idx = torch.sort(metric_base, dim=1, descending=True)
            pruned_indx = sorted_idx[:, :int(sparsity_ratio * metric_base.shape[1])].numpy()
        
            # do row-wise sorting for adj
            binary_mask_target = torch.zeros_like(abs_weights[layer_names[l]])
            _, sorted_idx_target = torch.sort(metric_target, dim=1, descending=True)
            pruned_indx_target = sorted_idx_target[:, :int(sparsity_ratio * metric_target.shape[1])].numpy()
            binary_mask_target.scatter_(1, torch.tensor(pruned_indx_target), 1)

            # make a binary mask of the size of weights 
            binary_mask = torch.zeros_like(abs_weights[layer_names[l]])
            diff = metric_target > metric_base
            binary_mask = diff * binary_mask_target
            binary_mask = binary_mask.float()

            # convert binary mask to array
            binary_mask = binary_mask.cpu().numpy().astype(int)
            binary_mask = scipy.sparse.csr_matrix(binary_mask)
            print("Binary mask density: ", np.mean(binary_mask.toarray()))

            # save in pickle file
            with open(os.path.join(args.skilled_neuron_path, f'timestep_{t}_layer_{l}.pkl'), 'wb') as f:
                pickle.dump(binary_mask, f) 

if __name__ == '__main__':
    main()