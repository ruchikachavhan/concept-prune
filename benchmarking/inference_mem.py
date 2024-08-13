import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import glob
from tqdm import tqdm
import logging
import open_clip
from argparse import ArgumentParser
sys.path.append(os.getcwd())
from utils import load_models
from torchvision import transforms
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from typing import Any, Mapping, Iterable, Union, List, Callable, Optional
from benchmarking_utils import set_benchmarking_path
from diffusers.pipelines.stable_diffusion import safety_checker

def sc(self, clip_input, images):
    return images, [False for i in images]

safety_checker.StableDiffusionSafetyChecker.forward = sc
safety_checker_ = safety_checker.StableDiffusionSafetyChecker


def read_jsonlines(filename: str) -> Iterable[Mapping[str, Any]]:
    """Yields an iterable of Python dicts after reading jsonlines from the input file."""
    print(f"Reading JSON lines from {filename}")
    file_size = os.path.getsize(filename)
    with open(filename) as fp:
        for line in tqdm(
            fp.readlines(), desc=f"Reading JSON lines from {filename}", unit="lines"
        ):
            try:
                example = json.loads(line)
                yield example
            except json.JSONDecodeError as ex:
                logging.error(f'Input text: "{line}"')
                logging.error(ex.args)
                raise ex
            
def load_jsonlines(filename: str) -> List[Mapping[str, Any]]:
    """Returns a list of Python dicts after reading jsonlines from the input file."""
    return list(read_jsonlines(filename))

### credit: https://github.com/somepago/DCR
def measure_SSCD_similarity(gt_images, images, model, device):
    ret_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    gt_images = torch.stack([ret_transform(x.convert("RGB")) for x in gt_images]).to(
        device
    )
    images = torch.stack([ret_transform(x.convert("RGB")) for x in images]).to(device)

    with torch.no_grad():
        feat_1 = model(gt_images).clone()
        feat_1 = nn.functional.normalize(feat_1, dim=1, p=2)

        feat_2 = model(images).clone()
        feat_2 = nn.functional.normalize(feat_2, dim=1, p=2)

        return torch.mm(feat_1, feat_2.T)


def measure_CLIP_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    with torch.no_grad():
        img_batch = [clip_preprocess(i).unsqueeze(0) for i in images]
        img_batch = torch.concatenate(img_batch).to(device)
        image_features = model.encode_image(img_batch)

        text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return (image_features @ text_features.T).mean(-1)
    
def input_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dbg', type=bool, default=None)
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--baseline', type=str, default=None)
    parser.add_argument('--hook_module', type=str, default='unet')
    parser.add_argument('--ckpt_name', type=str, default=None)
    parser.add_argument('--dataset_path', type=str, default='../diffusion_memorization/sdv1_500_mem_groundtruth/')
    parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--res_path', type=str, default='results/results_seed_0/stable-diffusion/')
    parser.add_argument("--reference_model_pretrain", default="laion2b_s12b_b42k")
    return parser.parse_args()

    
def main():
    args = input_args()

    args.benchmarking_result_path = set_benchmarking_path(args)
    args.benchmarking_result_path = os.path.join(args.benchmarking_result_path, args.model_id, args.target, args.baseline, 'benchmarking', 'concept_erase')
    print("Benchmarking result path: ", args.benchmarking_result_path)
    if not os.path.exists(args.benchmarking_result_path):
        os.makedirs(args.benchmarking_result_path)

    dataset_path = args.dataset_path
    if args.dataset_path == '../diffusion_memorization/sdv1_500_mem_groundtruth/':
        dataset = load_jsonlines(f"{dataset_path}/sdv1_500_mem_groundtruth.jsonl")
    elif args.dataset_path == '../diffusion_memorization/imagenette_duplicated/':
        dataset = load_jsonlines(f"{dataset_path}/imagenette_duplicated.jsonl")
    elif args.dataset_path == '../diffusion_memorization/memorized_sd2/':
        dataset = load_jsonlines(f"{dataset_path}/memorized_sd2.jsonl")
    else:
        raise ValueError(f"Dataset path {args.dataset_path} not supported")

    print(f"Number of images: {len(dataset)}")

    if args.dataset_path in ['../diffusion_memorization/sdv1_500_mem_groundtruth/', '../diffusion_memorization/memorized_sd2/']:
        extension = 'png'
    else:
        extension = 'JPEG'

    args.reference_model = 'ViT-g-14'

    # Load the remover model
    remover_model = load_models(args, args.ckpt_name)
    print("Remover model: ", remover_model)

    # Loading the models for similarity measurement
    sim_model = torch.jit.load("../diffusion_memorization/sscd_disc_large.torchscript.pt").to(args.gpu)
    ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
            args.reference_model,
            pretrained=args.reference_model_pretrain,
            device=args.gpu,
        )
    ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    gen_images, gt_images = [], []
    results = {}
    iter = 0
    for data in dataset:
        prompt = data['caption']
        im_index = data['index']
        print(f"Prompt: {prompt}")
         

        if os.path.exists(f"{args.benchmarking_result_path}/img_{iter}.jpg") == False:
                seed = args.seed
                torch.manual_seed(seed)
                np.random.seed(seed)
                print(f"Seed: {seed}", im_index)
                image = remover_model(prompt, safety_checker = safety_checker_).images[0]
                image.save(f"{args.benchmarking_result_path}/{iter}_{im_index}.png")
        else:
            print(f"Image already exists at {args.benchmarking_result_path}/img_{iter}.png")
            image = Image.open(f"{args.benchmarking_result_path}/img_{iter}.jpg")
        iter +=1
        gen_images = [image]
        # read ground truth images
        gt_folder = f"{dataset_path}/gt_images/{im_index}"
        # read all images in gt_image folder
        gt_images = [Image.open(filename) for filename in glob.glob(f"{gt_folder}/*.{extension}")]
        print(f"Ground truth images: {gt_images}")
        SSCD_sim = measure_SSCD_similarity(gt_images, gen_images, sim_model, args.gpu)

        gt_image = gt_images[SSCD_sim.argmax(dim=0)[0].item()]
        
        SSCD_sim_max = SSCD_sim.max(0)[0].item()
        SSCD_sim_min = SSCD_sim.min(0)[0].item()
        SSCD_sim = SSCD_sim.mean(0).item()
        print(f"SSCD similarity: {SSCD_sim}", {SSCD_sim_max}, {SSCD_sim_min})
        
        sims = measure_CLIP_similarity(
                [gt_image] + gen_images,
                prompt,
                ref_model,
                ref_clip_preprocess,
                ref_tokenizer,
                args.gpu,
            )
        gt_clip_score = sims[0:1].mean().item()
        gen_clip_score = sims[1:].mean().item()

        results[im_index] = {
            'prompt': prompt,
            'SSCD_sim': SSCD_sim,
            'gt_clip_score': gt_clip_score,
            'gen_clip_score': gen_clip_score,
            'SSCD_sim_max': SSCD_sim_max,
            'SSCD_sim_min': SSCD_sim_min
        }

        print(f"GT clip score: {gt_clip_score}", f"Gen clip score: {gen_clip_score}")
        
    # save the results
    avg_sscd, avg_sscd_min, avg_sscd_max = 0, 0, 0
    avg_gt_clip_score = 0
    avg_gen_clip_score = 0
    for key, value in results.items():
        avg_sscd += value['SSCD_sim']
        avg_sscd_min += value['SSCD_sim_min']
        avg_sscd_max += value['SSCD_sim_max']
        avg_gt_clip_score += value['gt_clip_score']
        avg_gen_clip_score += value['gen_clip_score']
    avg_sscd /= len(results)
    avg_gt_clip_score /= len(results)
    avg_gen_clip_score /= len(results)

    results['average'] = {
        'avg_sscd': avg_sscd,
        'avg_sscd_min': avg_sscd_min,
        'avg_sscd_max': avg_sscd_max,
        'avg_gt_clip_score': avg_gt_clip_score,
        'avg_gen_clip_score': avg_gen_clip_score
    }

    print("Results: ", results)
    # save the results
    with open(f"{args.benchmarking_result_path}/random_results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()