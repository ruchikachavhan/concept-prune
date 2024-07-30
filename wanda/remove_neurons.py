import os
import json
import scipy
import pickle
import torch
import sys
import tqdm
import numpy as np
from argparse import ArgumentParser
from PIL import Image, ImageFilter, ImageDraw
sys.path.append(os.getcwd())
from utils import get_prompts, Config, get_sd_model
from neuron_receivers import NeuronRemover
from transformers.models.clip.modeling_clip import CLIPMLP


# if nsfw is on, blur the image a 100 times
def blur_image(image, is_nsfw):
    if is_nsfw:
        for i in range(100):
            image = image.filter(ImageFilter.BLUR)
    return image

def remove_skilled_neurons(target_prompts, model, neuron_remover, args, save_path):
    iter = 0
    for ann_target in target_prompts:
        if iter >= 2 and args.dbg:
            break
        print("text: ", ann_target)

        # fix seed before running the model
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        out_base = model(ann_target).images[0]

        neuron_remover.reset_time_layer()
        out_target = neuron_remover.observe_activation(model, ann_target)

        # stitch the images to keep them side by side
        out_base = out_base.resize((256, 256))
        out_target = out_target.resize((256, 256))
        
        # make bigger image to keep both images side by side with white space in between
        new_im = Image.new('RGB', (530, 290))    
        new_im.paste(out_base, (0,40))
        new_im.paste(out_target, (275,40))
        # write the prompt on the image
        draw = ImageDraw.Draw(new_im)
        draw.text((80, 15), ann_target, (255, 255, 255))
        draw.text((350, 15), 'w/o skilled neurons', (255, 255, 255))
        print("Saving image in: ", os.path.join(save_path, f'img_{iter}_{ann_target}.jpg'))
        new_im.save(os.path.join(save_path, f'img_{iter}_{ann_target}.jpg'))
        iter += 1


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
    model = model.to(args.gpu)

    neuron_remover = NeuronRemover(seed=args.seed, path_expert_indx=args.skilled_neuron_path,
                                      T=args.timesteps, n_layers=num_layers, replace_fn=replace_fn, hook_module=args.hook_module)
    print("Neuron remover: ", neuron_remover)

    remove_skilled_neurons(target_prompts, model, neuron_remover, args, args.after_removal_results)

if __name__ == '__main__':
    main()