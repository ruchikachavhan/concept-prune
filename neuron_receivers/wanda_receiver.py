from utils import TimeLayerColumnNorm, ColumnNormCalculator
import torch
import numpy as np
from diffusers.models.activations import GEGLU, GELU
from neuron_receivers.base_receiver import BaseNeuronReceiver

class Wanda(BaseNeuronReceiver):
    def __init__(self, seed, T, n_layers, replace_fn = GEGLU, keep_nsfw=False, hook_module='unet'):
        super(Wanda, self).__init__(seed, replace_fn, keep_nsfw, hook_module)
        self.T = T
        self.n_layers = n_layers
        if hook_module in ['unet', 'unet-ffn-1', 'attn_key']:
            # create a dictionary to store activation norms for every time step and layer
            self.activation_norm = TimeLayerColumnNorm(T, n_layers)
        elif hook_module == 'text':
            # create a dictionary to store activation norms for every layer
            self.activation_norm = {}
            for l in range(self.n_layers):
                self.activation_norm[l] = ColumnNormCalculator()
        self.timestep = 0
        self.layer = 0
    
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
    
    def hook_fn(self, module, input, output):
        ''' 
            Store the norm of the gate for each layer and timestep of the FFNs in UNet
        '''
        
        args = (1.0,)
        if self.replace_fn == GEGLU:
            # First layer of the FFN
            hidden_states, gate = module.proj(input[0]).chunk(2, dim=-1)
            out = hidden_states * module.gelu(gate)

            # Store the input activation to the second layer
            save_gate = out.clone().view(-1, out.shape[-1]).detach().cpu()
            # normalize across the sequence length to avoid inf values
            save_gate = torch.nn.functional.normalize(save_gate, p=2, dim=1)
            self.activation_norm.update(save_gate, self.timestep, self.layer)

            # update the time step and layer
            self.update_time_layer()

            return out
        
    def unet_ffn_1_hook_fn(self, module, input, output):
        ''' 
            Store the norm of the gate for each layer and timestep of the FFNs in UNet
        '''
        
        args = (1.0,)
        if self.replace_fn == GEGLU:
            # First layer of the FFN
            hidden_states, gate = module.proj(input[0]).chunk(2, dim=-1)
            out = hidden_states * module.gelu(gate)

            # Store the input activation to the second layer
            save_gate = input[0].clone().view(-1, input[0].shape[-1]).detach().cpu()
            # normalize across the sequence length to avoid inf values
            save_gate = torch.nn.functional.normalize(save_gate, p=2, dim=1)
            self.activation_norm.update(save_gate, self.timestep, self.layer)

            # update the time step and layer
            self.update_time_layer()

            return out
        
    def unet_attn_layer(self, module, input, output):
        ''' 
            Store the norm of the gate for each layer and timestep of the FFNs in UNet
        '''
        save_gate = input[0].clone().detach().cpu()
        save_gate = save_gate.view(-1, save_gate.shape[-1])
        # normalize across the sequence length to avoid inf values
        save_gate = torch.nn.functional.normalize(save_gate, p=2, dim=1)
        self.activation_norm.update(save_gate, self.timestep, self.layer)
        self.update_time_layer()
        return output
        
    def text_hook_fn(self, module, input, output):
        ''' 
            Store the norm of the gate for each layer and timestep of the FFNs in text encoder (CLIP)
        '''
                
        # First layer of the FFN
        hidden_states = module.fc1(input[0])
        hidden_states = module.activation_fn(hidden_states)

         # Store the input activation to the second layer
        save_gate = hidden_states.clone().detach().cpu()
        save_gate = save_gate.view(-1, hidden_states.shape[-1])
        # normalize across the sequence length to avoid inf values
        save_gate = torch.nn.functional.normalize(save_gate, p=2, dim=1)
        if self.layer < self.n_layers:
            self.activation_norm[self.layer].add_rows(save_gate)
            
        # Output of the second layer of the FFN
        hidden_states = module.fc2(hidden_states)
        # update the time step and layer
        self.update_time_layer()
        return hidden_states