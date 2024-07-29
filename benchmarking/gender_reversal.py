import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from argparse import ArgumentParser
sys.path.append(os.getcwd())
from utils import load_models
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from torchvision.models import resnet50, ResNet50_Weights


def input_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dbg', action='store_true')
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--baseline', type=str, default='concept-prune')
    parser.add_argument('--benchmarking_result_path', type=str, default='results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/')
    parser.add_argument('--batch_size', type=int, default=1)
    return parser.parse_args()

def main():
    args = input_args()
    print("Arguments: ", args.__dict__)

    args.benchmarking_result_path = os.path.join(args.benchmarking_result_path, args.target, args.baseline, 'benchmarking')
    print("Benchmarking result path: ", args.benchmarking_result_path)
    if not os.path.exists(args.benchmarking_result_path):
        os.makedirs(args.benchmarking_result_path)


    # read prompts from test_professions.txt which consists of all 35 preofessions from the winobias dataset
    professions = []
    with open('datasets/test_professions.txt', 'r') as f:
        for line in f:
            professions.append(line.strip())

    # Load original model
    model = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)
    model = model.to(args.gpu)

    # Load the concept erased model
    remover_model = load_models(args)

    seeds = np.arange(0, 250, 1)

    for seed in seeds:
        save_path = os.path.join(args.benchmarking_result_path, f'seed_{seed}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for profession in professions:
            prompt = f'a headshot of a {profession.lower()}'
            print(f"Profession: {prompt}", "Seed: ", seed)
            # run model for the gender reversal model
            torch.manual_seed(seed)
            np.random.seed(seed)
            reversed_image = remover_model(prompt).images[0]

            # run for the original model
            torch.manual_seed(seed)
            np.random.seed(seed)
            original_image = model(prompt).images[0]

            # save the images
            reversed_image.save(os.path.join(save_path, f'{profession}_reversed.png'))
            original_image.save(os.path.join(save_path, f'{profession}_original.png'))
            print("Images saved in ", save_path)

if __name__ == '__main__':
    main()