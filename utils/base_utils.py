import os
import sys
import yaml
import torch
import json
import numpy as np
from diffusers.models.activations import GEGLU
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers.models.clip.modeling_clip import CLIPMLP

def make_dirs(args):
    # Making all directories for the experiment
    if not os.path.exists('test_images'):
        os.makedirs('test_images')

    if not os.path.exists(args.res_path):
        os.makedirs(args.res_path)
    
    if not os.path.exists(args.img_save_path):
        os.makedirs(args.img_save_path)

    if not os.path.exists(args.skilled_neuron_path):
        os.makedirs(args.skilled_neuron_path)

    if not os.path.exists(args.after_removal_results):
        os.makedirs(args.after_removal_results)

    if not os.path.exists(args.union_masks):
        os.makedirs(args.union_masks)
    
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
        os.mkdir(os.path.join(args.checkpoint_path, 'test_images'))
    
    if not os.path.exists(args.benchmarking_result_path):
        os.makedirs(args.benchmarking_result_path)
    

def get_sd_model(args):
    if args.model_id in ['runwayml/stable-diffusion-v1-5', 'CompVis/stable-diffusion-v1-4']:
        print("Loading from pre-trained model", args.model_id)
        model = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)

    if args.hook_module == 'unet':
        num_layers = args.n_layers
        replace_fn = GEGLU
    elif args.hook_module == 'text':
        num_layers = 12
        replace_fn = CLIPMLP

    return model, num_layers, replace_fn

def coco_dataset(data_path, split, num_images=1000):
    with open(os.path.join(data_path, f'annotations/captions_{split}2014.json')) as f:
        data = json.load(f)
    data = data['annotations']
    # select 30k images randomly
    np.random.seed(0)
    np.random.shuffle(data)
    data = data[:num_images]
    imgs = [os.path.join(data_path, f'{split}2014', 'COCO_' + split + '2014_' + str(ann['image_id']).zfill(12) + '.jpg') for ann in data]
    anns = [ann['caption'] for ann in data]
    return imgs, anns


target_types_dict = {
    'painting': 'art',
    'Van Gogh': 'art',
    'Monet': 'art',
    'Pablo Picasso': 'art',
    'Salvador Dali': 'art',
    'Leonardo da Vinci': 'art',
    'naked': 'naked',
    'cassette player': 'object',
    'chain saw': 'object',
    'church': 'object',
    'gas pump': 'object',
    'tench': 'object',
    'garbage truck': 'object',
    'english springer': 'object',
    'golf ball': 'object',
    'parachute': 'object',
    'french horn': 'object',    
    'female': 'gender',
    'male': 'gender'
}

class Config:
    def __init__(self, path):
        # Load config file
        with open(path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.config = config
        for key, value in config.items():
            setattr(self, key, value)
        
        if self.hook_module == 'unet':
            self.res_path = f'results/results_seed_{self.seed}' + '/' + self.res_path.split('/')[1]
        elif self.hook_module == 'text':
            self.res_path = f'results_CLIP/results_seed_{self.seed}' + '/' + self.res_path.split('/')[1]

    def configure(self): 
        self.target_type = target_types_dict[self.target]    
        self.res_path = os.path.join(self.res_path, self.model_id, self.target)
        self.img_save_path = os.path.join(self.res_path, 'images')
        self.skilled_neuron_path = os.path.join(self.res_path, 'skilled_neurons', str(self.skill_ratio))
        self.after_removal_results = os.path.join(self.res_path, 'after_removal_results', str(self.skill_ratio))
        self.union_masks = os.path.join(self.res_path, 'union_masks', str(self.skill_ratio))
        self.checkpoint_path = os.path.join(self.res_path, 'checkpoints')
        self.benchmarking_result_path = os.path.join(self.res_path)

        # make experiment folders
        make_dirs(self)
        
    def __repr__(self):
        for key, value in self.config.items():
            if value is not None:
                print(f"{key}: {value}")
    
class Average:
    '''
    Class to measure average of a set of values
    for all timesteps and layers
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
class StandardDev:
    def __init__(self):
        self.n = 0
        self.mean = 0
        self.M2 = 0

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def variance(self):
        if self.n < 2:
            return float('nan')
        else:
            return self.M2 / (self.n - 1)

    def stddev(self):
        return self.variance() ** 0.5


class StatMeter:
    '''
    Class to measure average and standard deviation of a set of values
    for all timesteps and layers
    '''
    def __init__(self, T, n_layers):
        self.reset()
        self.results = {}
        self.results['time_steps'] = {}
        self.T = T
        self.n_layers = n_layers
        for t in range(T):
            self.results['time_steps'][t] = {}
            for i in range(n_layers):
                self.results['time_steps'][t][i] = {}
                self.results['time_steps'][t][i]['avg'] = Average()
                self.results['time_steps'][t][i]['std'] = StandardDev()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, t, n_layer):
        self.results['time_steps'][t][n_layer]['avg'].update(val)
        self.results['time_steps'][t][n_layer]['std'].update(val)
        
    
    def save(self, path):
        for t in range(self.T):
            for i in range(self.n_layers):
                self.results['time_steps'][t][i]['avg'] = self.results['time_steps'][t][i]['avg'].avg
                self.results['time_steps'][t][i]['std'] = self.results['time_steps'][t][i]['std'].stddev()
                # check if its and array
                if isinstance(self.results['time_steps'][t][i]['avg'], np.ndarray):
                    self.results['time_steps'][t][i]['avg'] = self.results['time_steps'][t][i]['avg'].tolist()
                if isinstance(self.results['time_steps'][t][i]['std'], np.ndarray):
                    self.results['time_steps'][t][i]['std'] = self.results['time_steps'][t][i]['std'].tolist()

        with open(path, 'w') as f:
            json.dump(self.results, f)


class ColumnNormCalculator:
    def __init__(self):
        '''
        Calculated Column Norm of a matrix incrementally as rows are added
        Assumes 2D matrix
        '''
        self.A = np.zeros((0, 0))
        self.column_norms = torch.tensor([])

    def add_rows(self, rows):
        if len(self.A) == 0:  # If it's the first row
            self.A = rows
            self.column_norms = torch.norm(self.A, dim=0)
        else:
            # self.A = np.vstack((self.A, rows))
            new_row_norms = torch.norm(rows, dim=0)
            self.column_norms = torch.sqrt(self.column_norms**2 + new_row_norms**2)

    def get_column_norms(self):
        return self.column_norms



class TimeLayerColumnNorm:
    '''
    Column Norm calculator for all timesteps and layers
    '''

    def __init__(self, T, n_layers):
        self.T = T
        self.n_layers = n_layers
        self.column_norms = {}
        for t in range(T):
            self.column_norms[t] = {}
            for i in range(n_layers):
                self.column_norms[t][i] = ColumnNormCalculator()

    def update(self, rows, t, n_layer):
        self.column_norms[t][n_layer].add_rows(rows)

    def get_column_norms(self):
        results = {}
        for t in range(self.T):
            results[t] = {}
            for i in range(self.n_layers):
                results[t][i] = self.column_norms[t][i].get_column_norms()
        return results
    
    def save(self, path):
        results = self.get_column_norms()
        torch.save(results, path)