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
from utils import get_prompts, Config, get_sd_model, TimeLayerColumnNorm
from neuron_receivers import Wanda
from transformers.models.clip.modeling_clip import CLIPMLP
from torch.profiler import profile, record_function, ProfilerActivity
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

def input_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--dbg', type=bool, default=None)
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--base', type=str, default=None)
    parser.add_argument('--model_id', type=str, default=None)
    parser.add_argument('--skill_ratio', type=float, default=None)
    parser.add_argument('--target_file', type=str, default=None)
    parser.add_argument('--hook_module', type=str, default=None)
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
    args.benchmarking_result_path = os.path.join(args.benchmarking_result_path, 'concept-prune', 'benchmarking', 'concept_erase')
    if not os.path.exists(args.benchmarking_result_path):
        os.makedirs(args.benchmarking_result_path)
    print("Arguments: ", args.__dict__)
    base_prompts, target_prompts = get_prompts(args)
    print("Base prompts: ", base_prompts)
    print("Target prompts: ", target_prompts)

    # Model
    model, num_layers, replace_fn = get_sd_model(args)
    args.replace_fn = replace_fn
    print("Replace fn: ", replace_fn)
    model = model.to(args.gpu)
    print("Model: ", model.unet)

    # get the absolute value of FFN weights in the second layer
    abs_weights = {}
    layer_names = []
    weights_shape = {}
    if args.hook_module == 'unet':
        for name, module in model.unet.named_modules():
            if isinstance(module, torch.nn.Linear) and 'ff.net' in name and not 'proj' in name:
                layer_names.append(name)
                weight = module.weight.detach()
                abs_weights[name] = weight.abs().cpu()
                print("Storing absolute value of: ", name, module.weight.shape)
                weights_shape[name] = module.weight.shape
        # sort the layer names so that mid block is before up block
        layer_names.sort()
    
    elif args.hook_module == 'unet-ffn-1':
        for name, module in model.unet.named_modules():
            if isinstance(module, torch.nn.Linear) and 'ff.net' in name and 'proj' in name:
                layer_names.append(name)
                weight = module.weight.detach()
                abs_weights[name] = weight.abs().cpu()
                print("Storing absolute value of: ", name, module.weight.shape)
        # sort the layer names so that mid block is before up block
        layer_names.sort()

    elif args.hook_module == 'attn_key':
        for name, module in model.unet.named_modules():
            # Key of Cross attention (attn2)
            if isinstance(module, torch.nn.Linear) and 'attn2' in name and 'to_k' in name:
                layer_names.append(name)
                weight = module.weight.detach()
                abs_weights[name] = weight.abs().cpu()
                print("Storing absolute value of: ", name, module.weight.shape)
        # sort the layer names so that mid block is before up block
        layer_names.sort()
    
    elif args.hook_module == 'attn_val':
        for name, module in model.unet.named_modules():
            # Key of Cross attention (attn2)
            if isinstance(module, torch.nn.Linear) and 'attn2' in name and 'to_v' in name:
                layer_names.append(name)
                weight = module.weight.detach()
                abs_weights[name] = weight.abs().cpu()
                print("Storing absolute value of: ", name, module.weight.shape)
        # sort the layer names so that mid block is before up block
        layer_names.sort()

    elif args.hook_module == 'text':
        for name, module in model.text_encoder.named_modules():
            if isinstance(module, CLIPMLP) and 'mlp' in name and 'encoder.layers' in name:
                layer_names.append(name)
                weight = module.fc2.weight.detach().clone()
                abs_weights[name] = weight.abs().cpu()
                print("Storing absolute value of: ", name, module.fc2.weight.shape)
    
    print("Layer names: ", layer_names, len(layer_names))
    # Make two separate norm calculator classes for base and adj prompts
    neuron_receiver_base = Wanda(args.seed, args.timesteps, num_layers, replace_fn = args.replace_fn, keep_nsfw = args.keep_nsfw, hook_module=args.hook_module)
    neuron_receiver_target = Wanda(args.seed, args.timesteps, num_layers, replace_fn = args.replace_fn, keep_nsfw = args.keep_nsfw, hook_module=args.hook_module)

    iter = 0

    for ann, ann_target in tqdm.tqdm(zip(base_prompts, target_prompts)):

            if iter >= 3 and args.dbg:
                break
            print("text: ", ann, ann_target)

            # if target concept is gender, then we select random seeds
            if args.target_type == 'gender':
                seed = torch.randint(0, 250, (1,)).item()
                neuron_receiver_base.seed = seed
                neuron_receiver_target.seed = seed
                print("Seed: ", seed)
        
            neuron_receiver_base.reset_time_layer()
            out = neuron_receiver_base.observe_activation(model, ann)

            neuron_receiver_target.reset_time_layer()
            out_target = neuron_receiver_target.observe_activation(model, ann_target)

            act_norms_base = neuron_receiver_base.activation_norm.get_column_norms()
            act_norms_target = neuron_receiver_target.activation_norm.get_column_norms()

            # clear activation norm for the next iteration
            neuron_receiver_base.activation_norm = TimeLayerColumnNorm(args.timesteps, num_layers)
            neuron_receiver_target.activation_norm = TimeLayerColumnNorm(args.timesteps, num_layers)

            all_masks = {}
            for t in range(1):
                all_masks[t] = {}
                for l in range(num_layers):
                    print("Time step: ", t, "Layer: ", l)
                    # print("Base norm shape: ", act_norms_base[t][l])
                    print("Target norm shape: ", act_norms_target[t][l])
                    print("Weight shape: ", abs_weights[layer_names[l]].shape)

                    # wanda score is W.abs() * A
                    metric_base = abs_weights[layer_names[l]] * act_norms_base[t][l]
                    metric_target = abs_weights[layer_names[l]] * act_norms_target[t][l]

                    # check for inf values here
                    if torch.isinf(metric_base).any():
                        print("Inf values in metric base")

                    # do row-wise sorting for base in descending order
                    _, sorted_idx = torch.sort(metric_base, dim=1, descending=True)
                    pruned_indx = sorted_idx[:, :int(args.skill_ratio * metric_base.shape[1])].numpy()
                
                    # do row-wise sorting for adj
                    binary_mask_target = torch.zeros_like(abs_weights[layer_names[l]])
                    _, sorted_idx_target = torch.sort(metric_target, dim=1, descending=True)
                    pruned_indx_target = sorted_idx_target[:, :int(args.skill_ratio * metric_target.shape[1])].numpy()
                    binary_mask_target.scatter_(1, torch.tensor(pruned_indx_target), 1)

                    # make a binary mask of the size of weights 
                    binary_mask = torch.zeros_like(abs_weights[layer_names[l]])
                    diff = metric_target > metric_base
                    binary_mask = diff * binary_mask_target
                    # binary_mask_base = torch.zeros_like(abs_weights[layer_names[l]])
                    # binary_mask_base.scatter_(1, torch.tensor(pruned_indx), 1)
                    # binary_mask = binary_mask_base
                    binary_mask = binary_mask.float()
                    # print("Binary mask: ", binary_mask.shape, binary_mask.mean())
                    all_masks[t][l] = binary_mask
                    print("Binary mask: ", binary_mask.shape, binary_mask.mean())

            union_concepts = {}
            masks = {}
            for l in range(args.n_layers):
                union_concepts[layer_names[l]] = np.zeros(weights_shape[layer_names[l]])
                union_concepts[layer_names[l]] = scipy.sparse.csr_matrix(union_concepts[layer_names[l]])
                for t in range(0, 1):
                    # load sparse matrix
                    indices = all_masks[t][l]
                    # take union
                    # indices = indices.toarray()
                    print("Indices shape: ", indices.shape, indices.mean())
                    union_concepts[layer_names[l]] += scipy.sparse.csr_matrix(indices)
                union_concepts[layer_names[l]] = union_concepts[layer_names[l]] > 0.0
                array = union_concepts[layer_names[l]].astype('bool').astype('int')
                array = array.toarray()
                masks[layer_names[l]] = array       
                
            # apply masks to the model weights and save the model
            new_model = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
            
            print("Applying masks to the model")
            for name, module in new_model.unet.named_modules():
                if isinstance(module, torch.nn.Linear) and 'ff.net' in name and not 'proj' in name:
                    weight = module.weight.data.clone().detach().cpu()
                    weight *= (1- masks[name])
                    weight = weight.to(torch.float16)
                    module.weight.data = weight

            new_model = new_model.to(args.gpu)
            # evaluate the model
            print("Evaluating the model")
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            new_out = new_model(ann_target).images[0]
            # save the images
            new_out.save(os.path.join(args.benchmarking_result_path, f'img_{iter}.jpg'))
            iter += 1


if __name__ == '__main__':
    main()