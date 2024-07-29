import os
import sys
import torch
import numpy as np

things = []
with open(os.path.join('datasets', 'things.txt'), 'r') as f:
    things = f.readlines()
things = [thing.strip() for thing in things]

humans = []
with open(os.path.join('datasets', 'humans.txt'), 'r') as f:
    humans = f.readlines()
humans = [human.strip() for human in humans]

common_scenes = []
with open(os.path.join('datasets', 'common_scenes.txt'), 'r') as f:
    common_scenes = f.readlines()
common_scenes = [scene.strip() for scene in common_scenes]

scenes_names = ['road', 'park', 'beach', 'street', 'house', 'statue', 'tree', 'forest']
scenes_church = ['on a road', 'in the park', 'near the beach', ' on the street', 'in front of a house', 'in front of a statue', 'near a tree', 'in a forest']

places_names = ['road', 'park', 'street', 'white car', 'blue car',  'black car', 'white automobile',  'black automobile',  'blue automobile', 'truck', 'tree', 'forest', 'house']
places_truck = ['on the road', 'in the park', 'on the street', 'near a white car', 'near a blue car',  'near a black car', 'near a white automobile', 'near a black automobile', 'near a blue automobile', 'near a big truck', 'near a tree', 'in a forest', 'in front of a house']

# List of males
male_humans = [
    "Man",
    "Boy",
    "person",
    "guy",
    "Father",
    "Son",
    "Brother",
    "Husband",
    "Uncle",
    "Nephew",
    "Grandfather",
    "King"
]

# List of females
female_humans = [
    "Woman",
    "Girl",
    "female",
    "lady",
    "Mother",
    "Daughter",
    "Sister",
    "Wife",
    "Aunt",
    "Niece",
    "Grandmother",
    "Queen"
]

file_names_dict = {
    'art': 'things',
    'naked': 'humans',
    'Cassette Player': 'object',
    'Chain Saw': 'object',
    'Church': 'object',
    'Gas Pump': 'object',
    'Tench': 'object',
    'Garbage Truck': 'object',
    'English Springer': 'object',
    'Golf Ball': 'object',
    'Parachute': 'object',
    'French Horn': 'object', 
}

def get_prompts(args):

    target = args.target
    target_type = args.target_type

    if target_type == 'naked':
        base_prompts = [f'a photo of a {thing}' for thing in humans]
        target_prompts = [f'a photo of a {target} {thing}' for thing in humans]
        args.keep_nsfw = True
    
    elif target_type == 'art':
        base_prompts = [f'a photo of a {thing}' for thing in things]
        target_prompts = [f'a {thing} in the style of {target}' for thing in things]
    
    elif target_type == 'object':
        if target in ["parachute", "cassette player", "golf ball", "gas pump", "english springer", "tench", "chain saw", "french horn"]:
            base_prompts = [f'a {thing}' for thing in common_scenes]
            target_prompts = [f'a {target} in a {thing}' for thing in common_scenes]

        elif target in ['church']:
            base_prompts = [f'a {thing}' for thing in scenes_names]
            target_prompts = [f'a {target} {thing}' for thing in scenes_church]

        elif target in ['garbage truck']:
            base_prompts = [f'a {thing}' for thing in places_names]
            target_prompts = [f'a {target} {thing}' for thing in places_truck]

    elif target in ['male']:
        base_prompts = [f'a headshot of a {thing}' for thing in male_humans]
        target_prompts = [f'a headshot of a {thing}' for thing in female_humans]
    
    elif target in ['female']:
        base_prompts = [f'a headshot of a {thing}' for thing in female_humans]
        target_prompts = [f'a headshot of a {thing}' for thing in male_humans]

    
    # elif target in ['memorize'] or target.startswith('memorize'):
    #     base_prompts = ['' for _ in things]
    #     target_prompts = [f'{thing}' for thing in things]

    return base_prompts, target_prompts