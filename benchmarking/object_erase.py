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
    parser.add_argument('--baseline', type=str, default=None)
    parser.add_argument('--benchmarking_result_path', type=str, default='results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/')
    parser.add_argument('--removal_mode', type=str, default=None, choices=['erase', 'keep'])
    parser.add_argument('--hook_module', type=str, default='unet')
    parser.add_argument('--batch_size', type=int, default=1)
    return parser.parse_args()

# Dataset class to test concept erasure
class CustomDatasetErasure(torch.utils.data.Dataset):
    def __init__(self, data, concepts_to_remove):
        self.prompts = data['prompt']
        self.concepts_to_remove = concepts_to_remove
        self.seeds = data['evaluation_seed']
        try:
            self.labels = data['class']
        except:
            self.labels = data['label_str']

        # select only prompts that have the concept to remove
        self.prompts = [(self.prompts[i], self.seeds[i], concepts_to_remove) for i in range(len(self.prompts)) if concepts_to_remove.lower() == self.labels[i].lower()]
        
        

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        prompt = self.prompts[idx][0]
        prompt = prompt.replace("an image", "a photo")
        seed = self.prompts[idx][1]
        label = self.prompts[idx][2].lower()
        return prompt, seed, label
    
# Dataset class to test concept keeping
class CustomDatasetKeep(torch.utils.data.Dataset):
    def __init__(self, data, concepts_to_remove):
        self.dataset = data['prompt']
        self.concepts_to_remove = concepts_to_remove
        self.seeds = data['evaluation_seed']
        try:
            self.labels = data['class']
        except:
            self.labels = data['label_str']
        self.dataset = [(self.dataset[i], self.seeds[i], self.labels[i].lower()) for i in range(len(self.dataset)) if concepts_to_remove.lower() != self.labels[i].lower()]

        # select only 500 prompts per class
        labels_dict = {}
        self.prompts = []
        for i in range(len(self.dataset)):
            label = self.labels[i].lower()
            if label == concepts_to_remove:
                continue
            if label not in labels_dict:
                labels_dict[label] = 1
                self.prompts.append((self.dataset[i][0], self.dataset[i][1], self.dataset[i][2]))
            elif labels_dict[label] < 500:
                labels_dict[label] += 1
                self.prompts.append((self.dataset[i][0], self.dataset[i][1], self.dataset[i][2]))
            print(labels_dict)
        print(f"Number of prompts: {len(self.prompts)}")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx][0]
        prompt = prompt.replace("an image", "a photo")
        seed = self.prompts[idx][1]
        label = self.prompts[idx][2].lower()
        return prompt, seed, label
    

def main():
    args = input_args()
    print("Arguments: ", args.__dict__)

    args.benchmarking_result_path = os.path.join(args.benchmarking_result_path, args.target, args.baseline, 'benchmarking', f'concept_{args.removal_mode}')
    print("Benchmarking result path: ", args.benchmarking_result_path)
    if not os.path.exists(args.benchmarking_result_path):
        os.makedirs(args.benchmarking_result_path)

    if not os.path.exists(f'{args.benchmarking_result_path}'):
        os.makedirs(f'{args.benchmarking_result_path}')

    # Load dataset
    data = pd.read_csv(f'datasets/imagenette.csv')
    if args.removal_mode == 'erase':
        dataloader = torch.utils.data.DataLoader(CustomDatasetErasure(data, args.target), batch_size=args.batch_size, shuffle=False)
    else:
        dataloader = torch.utils.data.DataLoader(CustomDatasetKeep(data, args.target), batch_size=args.batch_size, shuffle=False)

    print("Number of prompts: ", len(dataloader))

    # Load the concept erased model
    remover_model = load_models(args)

    # Pre-trained ResNet50 
    weights = ResNet50_Weights.DEFAULT
    classifier = resnet50(weights=weights)
    classifier = classifier.to(args.gpu)
    classifier.eval()

    preprocess = weights.transforms()

    # test model on dataloader
    avg_acc = 0
    for iter, (prompt, seed, label) in enumerate(dataloader):
        if args.dbg and iter > 10:
            break

        print(f"Prompt: {prompt}, Seed: {seed}, Label: {label}")
        prompt = prompt[0]
        label = label[0]
        torch.manual_seed(seed[0])
        np.random.seed(seed[0])
        removal_images = remover_model(prompt)
        # save images
        for i, image in enumerate(removal_images.images):
            image.save(os.path.join(args.benchmarking_result_path, f"removed_{iter * args.batch_size + i}.png"))

        # evaluation using resnet50
        for i, image in enumerate(removal_images.images):
            image = preprocess(image).unsqueeze(0)
            image = image.to(args.gpu)
            with torch.no_grad():
                output = classifier(image)
                
            s, indices = torch.topk(output, 1)
            indices = indices.cpu().numpy()
            pred_labels = [weights.meta["categories"][idx] for idx in indices[0]]
            print(f"Predicted labels: {pred_labels}")
            pred_labels = [l.lower() for l in pred_labels]

            if label in pred_labels:
                avg_acc += 1

    print("Object predicted in: %d/%d images" % (avg_acc, len(dataloader)))
    print(f"Average accuracy: {avg_acc / len(dataloader)}")
    results = {"average_accuracy": avg_acc / len(dataloader)}
    with open(os.path.join(args.benchmarking_result_path, f"results_{args.removal_mode}.json"), 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()

        



