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
# import clip from transformers
from transformers import AutoProcessor, CLIPModel, AutoTokenizer


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
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--dbg', action='store_true')
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--baseline', type=str, default=None)
    parser.add_argument('--hook_module', type=str, default='unet')
    parser.add_argument('--ckpt_name', type=str, default=None)
    parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--benchmarking_result_path', type=str, default='results/results_seed_0/stable-diffusion/')
    parser.add_argument('--batch_size', type=int, default=16)
    return parser.parse_args()

def main():
    args = input_args()
    print("Arguments: ", args.__dict__)

    args.benchmarking_result_path = os.path.join(args.benchmarking_result_path, args.model_id, args.target, args.baseline, 'benchmarking', 'concept_erase')
    print("Benchmarking result path: ", args.benchmarking_result_path)
    if not os.path.exists(args.benchmarking_result_path):
        os.makedirs(args.benchmarking_result_path)


    # Load datasets
    imgs, anns = coco_dataset('../COCO-vqa', 'val', 480)
    dataset = COCODataset(imgs, anns)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load the concept erased model
    remover_model = load_models(args, args.ckpt_name)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # test model on dataloader
    average_sim = 0
    for iter, (img, prompt) in enumerate(dataloader):
        if iter > 2 and args.dbg:
            break

        # check if image is present in putput path
        if os.path.exists(os.path.join(args.benchmarking_result_path, f"img_{iter * args.batch_size}.jpg")):
            print(f"Skipping iteration {iter}")
            # continue
            # read the images
            images = []
            for i in range(args.batch_size):
                image = Image.open(os.path.join(args.benchmarking_result_path, f"img_{iter * args.batch_size + i}.jpg"))
                images.append(image)
        else:

            print("Iteration number", iter, prompt)
            prompt = [p for p in prompt]

            # fix seed before running the model
            torch.manual_seed(0)
            np.random.seed(0)
            images = remover_model(prompt).images
            for i, image in enumerate(images):
                image.save(os.path.join(args.benchmarking_result_path, f"removed_{iter * args.batch_size + i}.jpg"))
        
        for i, image in enumerate(images):
            # image features
            inputs = processor(images=image, return_tensors="pt")
            image_features = clip_model.get_image_features(**inputs)
            inputs = tokenizer(prompt[i], return_tensors="pt", padding=True)
            text_features = clip_model.get_text_features(**inputs)

            # normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # calculate similarity
            sim = (image_features @ text_features.T)
            average_sim += sim.item()
            print(f"Similarity: {sim}")

    print("Average similarity: ", average_sim / len(dataloader))
    # save 
    results = {"average_similarity": average_sim / len(dataloader)}
    # p = args.ckpt_name.split('/')[-1].split('.pt')[0] if args.ckpt_name else args.baseline
    p = 'all_coco'
    print("Saving results to ", os.path.join(args.benchmarking_result_path, f'{p}_results.json'))
    with open(os.path.join(args.benchmarking_result_path, f'{p}_results.json'), 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()

    

