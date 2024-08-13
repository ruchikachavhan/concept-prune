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

imgs, anns = coco_dataset('../COCO-vqa', 'val', 480)

# list all prompts in the dataset
prompts = []
for ann in anns:
    prompts.append(ann)
    print(ann)

# save the prompts in a txt file
with open('datasets/coco_prompts.txt', 'w') as f:
    for prompt in prompts:
        f.write(prompt + '\n')