import os
import torch
import pickle
import numpy as np
from diffusers.models.activations import GEGLU
from transformers.models.clip.modeling_clip import CLIPMLP
from neuron_receivers.base_receiver import BaseNeuronReceiver


class NeuronRemover(BaseNeuronReceiver):
    def __init__(self, seed, path_expert_indx, T, n_layers, replace_fn = GEGLU, keep_nsfw=False, hook_module='unet'):
        super(NeuronRemover, self).__init__(seed, replace_fn, keep_nsfw, hook_module)
        self.neuron_indices = {}
        self.T = T
        self.n_layers = n_layers
        for i in range(0, T):
            self.neuron_indices[i] = {}
            for j in range(0, n_layers):
                print("Reading skilled neuron indices from", os.path.join(path_expert_indx, f'timestep_{i}_layer_{j}.pkl'))
                with open(os.path.join(path_expert_indx, f'timestep_{i}_layer_{j}.pkl'), 'rb') as f:
                    # load sparse matrix from pickle file
                    indices = pickle.load(f)
                    # convert to array
                    self.neuron_indices[i][j] = torch.tensor(indices.toarray())
        
        self.timestep = 0
        self.layer = 0
        self.replace_fn = replace_fn
    
    def update_time_layer(self):
        # updates the timestep when self.layer reaches the last layer
        if self.layer == self.n_layers - 1:
            self.layer = 0
            self.timestep += 1
        else:
            self.layer += 1

    def reset_time_layer(self):
        self.timestep = 0
        self.layer = 0

    def unet_hook_fn(self, module, input, output):
        # Linear, second layer of the FFN
        # change weights by applying the binary mask
        old_weights = module.weight.clone()
        # read the expert indices
        binary_mask = self.neuron_indices[self.timestep][self.layer]
        binary_mask = binary_mask.to(old_weights.device)
        new_weights = old_weights * (1 - binary_mask)
        # replace the weights with old weights
        hidden_states = torch.nn.functional.linear(input[0], new_weights, module.bias)
        
        assert hidden_states.shape == output.shape, "Output shape should be same as hidden states"
        
        self.update_time_layer()
        return hidden_states
    
    def text_hook_fn(self, module, input, output):
        old_weights = module.fc2.weight.clone()
        # read the expert indices, only one timestep
        binary_mask = self.neuron_indices[0][self.layer]
        binary_mask = binary_mask.to(old_weights.device)
        new_weights = old_weights * (1 - binary_mask)

        hidden_states = module.fc1(input[0])       
        hidden_states = module.activation_fn(hidden_states)

        hidden_states = torch.nn.functional.linear(hidden_states, new_weights, module.fc2.bias)

        assert hidden_states.shape == output.shape, "Output shape should be same as hidden states"
        
        self.update_time_layer()
        return hidden_states
    
    def observe_activation(self, model, ann):
        # hook the model
        hooks = []
        # Unlike Base receiver that attaches hook to GEGLU block (first layer of FFN)
        # this receiver attaches hook to LoRACompatibleLinear block (second layer of FFN)
        if self.hook_module == 'unet':
            num_modules = 0
            for name, module in model.unet.named_modules():
                if isinstance(module, torch.nn.Linear) and 'ff.net' in name and not 'proj' in name:
                    hook = module.register_forward_hook(self.unet_hook_fn)
                    num_modules += 1
                    hooks.append(hook)
        
        elif self.hook_module == 'unet-ffn-1':
            num_modules = 0
            for name, module in model.unet.named_modules():
                if isinstance(module, torch.nn.Linear) and 'ff.net' in name and 'proj' in name:
                    hook = module.register_forward_hook(self.unet_hook_fn)
                    num_modules += 1
                    hooks.append(hook)
        
        elif self.hook_module == 'attn_key':
            num_modules = 0
            for name, module in model.unet.named_modules():
                if isinstance(module, torch.nn.Linear) and 'attn2' in name and 'to_k' in name:
                    hook = module.register_forward_hook(self.unet_hook_fn)
                    num_modules += 1
                    hooks.append(hook)

        elif self.hook_module == 'attn_val':
            num_modules = 0
            for name, module in model.unet.named_modules():
                if isinstance(module, torch.nn.Linear) and 'attn2' in name and 'to_v' in name:
                    hook = module.register_forward_hook(self.unet_hook_fn)
                    num_modules += 1
                    hooks.append(hook)

        elif self.hook_module == 'text':
            num_modules = 0
            for name, module in model.text_encoder.named_modules():
                if isinstance(module, CLIPMLP) and 'mlp' in name and 'encoder.layers' in name:
                    hook = module.register_forward_hook(self.text_hook_fn)
                    num_modules += 1
                    hooks.append(hook)

        # forward pass
        #  fix seed to get the same output for every run
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        out = model(ann, safety_checker=self.safety_checker).images[0]

        # remove the hook
        self.remove_hooks(hooks)
        return out