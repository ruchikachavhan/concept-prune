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

coco_dataset_prompts_1 = [
    'A bicycle replica with a clock as the front wheel',
    "A bathroom with a toilet, sink, and shower.",
    'Two women waiting at a bench next to a street',
    'Two huskys hanging out of the car windows.',
    'A kitchen is shown with wooden cabinets and a wooden celling.',
    'A black and white photo of an older man skiing',
    'A dog and a goat with their noses touching at fence.',
    'A fireplace with a fire built in it.',
    'A couple of birds fly through a blue cloudy sky.',
    'A group of people preparing food in a kitchen.'
]

coco_dataset_prompts_2 = [
    "A group of people sitting around a table with food",
    "A man riding a motorcycle on a city street",
    "A woman cutting vegetables in a kitchen",
    "A young boy holding a baseball bat",
    "A dog jumping to catch a frisbee in a park",
    "A person riding a bike down a mountain trail",
    "A fruit stand at an outdoor market",
    "A group of people working on laptops in an office",
    "A cat sleeping on a couch",
    "People sunbathing and playing in the ocean at a beach"
]

coco_dataset_prompts_3 = [
    "A man sitting on a bench in a park",
    "A woman holding an umbrella in the rain",
    "A couple walking their dog on a leash",
    "A family having a picnic on a grassy field",
    "A person skiing down a snowy slope",
    "A bus driving down a busy street",
    "A group of friends taking a selfie",
    "A construction worker wearing a hard hat",
    "A street performer playing a guitar",
    "A boat sailing on a calm lake"
]

coco_dataset_prompts_4 = [
    "A man cooking on a barbecue grill",
    "A person reading a book in a library",
    "A woman painting on a canvas",
    "A group of children playing soccer",
    "A train moving along a railway track",
    "A person surfing a large wave",
    "A couple dancing at a party",
    "A cyclist riding through a city park",
    "A chef preparing food in a restaurant kitchen",
    "A person hiking up a mountain trail"
]

coco_dataset_prompts_0 = [
    "A farmer driving a tractor in a field",
    "A person walking through a forest trail",
    "A group of people exercising in a gym",
    "A woman practicing yoga on a mat",
    "A man fishing at a riverbank",
    "A person standing at a bus stop",
    "A family decorating a Christmas tree",
    "A person snowboarding down a hill",
    "A person playing the piano in a concert hall",
    "A group of people riding horses on a trail"
]

all_coco_prompts = [
    coco_dataset_prompts_0,
    coco_dataset_prompts_1,
    coco_dataset_prompts_2,
    coco_dataset_prompts_3,
    coco_dataset_prompts_4
]

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

    
    elif target in ['memorize'] or target.startswith('memorize'):
        # read target_file from args
        target_file = args.target_file
        assert os.path.exists(target_file), f"Target file {target_file} does not exist"
        with open(target_file, 'r') as f:
            target_prompts = f.readlines()
        target_prompts = [prompt.strip() for prompt in target_prompts]
        base_prompts = ['' for _ in target_prompts]
        # base_prompts = [p for p in coco_dataset_prompts]
        # base_prompts = base_prompts[:len(target_prompts)]
    
    elif target.startswith('coco_memorize'):
        # read target_file from args
        target_file = args.target_file
        assert os.path.exists(target_file), f"Target file {target_file} does not exist"
        with open(target_file, 'r') as f:
            target_prompts = f.readlines()
        target_prompts = [prompt.strip() for prompt in target_prompts]
        # base_prompts = ['' for _ in target_prompts]
        # select a random prompt from coco_dataset_prompts
        base_prompts = [p for p in all_coco_prompts[int(target[-1])]]
        # base_prompts = base_prompts[:len(target_prompts)]
    
    elif target.startswith('tv_memorize') or target.startswith('mv_memorize'):
        # read target_file from args
        target_file = args.target_file
        assert os.path.exists(target_file), f"Target file {target_file} does not exist"
        with open(target_file, 'r') as f:
            target_prompts = f.readlines()
        target_prompts = [prompt.strip() for prompt in target_prompts]
        base_prompts = ['' for _ in target_prompts]
    
    elif target.startswith('cluster'):
        target_file = f'../cluster_dms/{target}.txt'
        assert os.path.exists(target_file), f"Target file {target_file} does not exist"
        with open(target_file, 'r') as f:
            target_prompts = f.readlines()
        target_prompts = [prompt.strip() for prompt in target_prompts]
        base_prompts = ['' for _ in target_prompts]


    return base_prompts, target_prompts