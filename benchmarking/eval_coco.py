import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from argparse import ArgumentParser
sys.path.append(os.getcwd())
from utils import load_models, coco_dataset
from diffusers import StableDiffusionPipeline, UNet2DConditionModel


# COCO dataset class
class COCODataset(torch.utils.data.Dataset):
    def __init__(self, imgs, anns):
        self.imgs = imgs
        self.anns = anns

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        ann = self.anns[idx]
        return img, ann 
    
def input_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--dbg', type=bool, default=None)
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--baseline', type=str, default=None)
    parser.add_argument('--benchmarking_result_path', type=str, default='results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/')
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_args()

def main():
    args = input_args()
    print("Arguments: ", args.__dict__)

    args.benchmarking_result_path = os.path.join(args.benchmarking_result_path, args.target, args.baseline, 'benchmarking', 'eval_coco')
    print("Benchmarking result path: ", args.benchmarking_result_path)
    if not os.path.exists(args.benchmarking_result_path):
        os.makedirs(args.benchmarking_result_path)


    # Load datasets
    imgs, anns = coco_dataset('../COCO-vqa', 'val', 30000)
    dataset = COCODataset(imgs, anns)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load the concept erased model
    remover_model = load_models(args)

    # test model on dataloader
    for iter, (img, prompt) in enumerate(dataloader):

        # check if image is present in putput path
        if os.path.exists(os.path.join(args.benchmarking_result_path, f"original_{iter * args.batch_size}.png")):
            print(f"Skipping iteration {iter}")
            continue

        print("Iteration number", iter, prompt)
        prompt = [p for p in prompt]

        # fix seed before running the model
        torch.manual_seed(0)
        np.random.seed(0)

        images = remover_model(prompt).images
        for i, image in enumerate(images):
            image.save(os.path.join(args.benchmarking_result_path, f"removed_{iter * args.batch_size + i}.png"))

if __name__ == '__main__':
    main()

    

