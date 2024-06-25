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

def input_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--dbg', type=bool, default=None)
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--baseline', type=str, default=None)
    parser.add_argument('--benchmarking_result_path', type=str, default='results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/')
    return parser.parse_args()

class ArtistDataset(torch.utils.data.Dataset):
    def __init__(self, prompts):
        self.prompts = prompts['prompt'].tolist()
        self.seeds = prompts['evaluation_seed'].tolist()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        seed = self.seeds[idx]
        return prompt,  seed

def main():
    args = input_args()
    print("Arguments: ", args.__dict__)

    args.benchmarking_result_path = os.path.join(args.benchmarking_result_path, args.target, args.baseline, 'benchmarking')
    print("Benchmarking result path: ", args.benchmarking_result_path)
    if not os.path.exists(args.benchmarking_result_path):
        os.makedirs(args.benchmarking_result_path)

    if not os.path.exists(f'{args.benchmarking_result_path}/concept_erase'):
        os.makedirs(f'{args.benchmarking_result_path}/concept_erase')

    # Load datasets
    data = pd.read_csv(f'datasets/test_{args.target}.csv')
    dataset = ArtistDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Run the model if the folder is empty
    if len(os.listdir(f'{args.benchmarking_result_path}/concept_erase')) == 0:
        print("Saving images after removal of concept")
        
        # Load original SD model
        model = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)
        model = model.to(args.gpu)

        # Load concept erased model
        remover_model = load_models(args)
        print("Remover model: ", remover_model)

        for i, (prompt, seed) in enumerate(dataloader):
            prompt = prompt[0]
            print(f"Prompt: {prompt}, Seed: {seed}")
            
            torch.manual_seed(seed[0])
            np.random.seed(seed[0])
            removal_images = remover_model(prompt).images[0]

            torch.manual_seed(seed[0])
            np.random.seed(seed[0])
            original_images = model(prompt).images[0]

            # save both images
            removal_images.save(f'{args.benchmarking_result_path}/concept_erase/removal_{i}.jpg')
            original_images.save(f'{args.benchmarking_result_path}/concept_erase/original_{i}.jpg')
    
    print("Calculating CLIP scores for the images")
    from transformers import CLIPProcessor, CLIPModel
    # calculate CLIP scores for the images
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load the images
    removal_images = [Image.open(f'{args.benchmarking_result_path}/concept_erase/removal_{i}.jpg') for i in range(len(data))]
    original_images = [Image.open(f'{args.benchmarking_result_path}/concept_erase/original_{i}.jpg') for i in range(len(data))]

    # Calculate the CLIP similarity between generated images after removal and prompt
    scores, similarity = [], []
    for iter, (prompt, seed) in enumerate(dataloader):
        prompt = prompt[0]
        print(f"Prompt: {prompt}, Seed: {seed}")
        removal_image = removal_images[iter]
        original_image = original_images[iter]

        # encode the text
        text_inputs = clip_processor(prompt, return_tensors="pt", padding=True)
        text_features = clip_model.get_text_features(**text_inputs)

        # encode the images
        image_inputs = clip_processor(images=original_image, return_tensors="pt")
        orig_image_features = clip_model.get_image_features(**image_inputs)

        image_inputs = clip_processor(images=removal_image, return_tensors="pt")
        removal_image_features = clip_model.get_image_features(**image_inputs)
        similarity_orig = torch.nn.functional.cosine_similarity(text_features, orig_image_features)
        similarity_removed = torch.nn.functional.cosine_similarity(text_features, removal_image_features)

        similarity.append(similarity_removed.item())
        score = 1 if similarity_removed < similarity_orig else 0
        scores.append(score)

    avg_similarity = np.mean(similarity)
    avg_score = np.mean(scores)

    print(f"Average similarity between prompt and generated image after removal: {avg_similarity}")
    print(f"Average score between prompt and generated image after removal: {avg_score}")

    # Save the scores
    results = {}
    results['avg_similarity'] = avg_similarity
    results['avg_score'] = avg_score
    results['std_similarity'] = np.std(similarity)
    results['std_score'] = np.std(scores)
    with open(f'{args.benchmarking_result_path}/concept_erase/clip_scores.json', 'w') as f:
        json.dump(results, f)

        
if __name__ == '__main__':
    main()

