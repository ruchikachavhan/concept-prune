import torch
import numpy as np
from diffusers.models.activations import GEGLU, GELU
from diffusers.pipelines.stable_diffusion import safety_checker
from transformers.models.clip.modeling_clip import CLIPMLP

def sc(self, clip_input, images):
    return images, [False for i in images]

class BaseNeuronReceiver:
    '''
    This is the base class for storing and changing activations
    '''

    def __init__(self, seed = 0, replace_fn = GEGLU, keep_nsfw = False, hook_module='unet'):
        self.seed = seed
        self.keep_nsfw = keep_nsfw
        if self.keep_nsfw:
            print("Removing safety checker")
            safety_checker.StableDiffusionSafetyChecker.forward = sc
        self.safety_checker = safety_checker.StableDiffusionSafetyChecker
        self.replace_fn = replace_fn
        self.hook_module = hook_module
        
    
    def hook_fn(self, module, input, output):
        # custom hook function
        raise NotImplementedError

    def text_hook_fn(self, module, input, output):
        # custom hook function
        raise NotImplementedError
    
    def remove_hooks(self, hooks):
        for hook in hooks:
            hook.remove()
    
    def observe_activation(self, model, ann):
        hooks = []

        # hook the model
        if self.hook_module == 'unet':
            # hook the unet
            num_modules = 0
            for name, module in model.unet.named_modules():
                if isinstance(module, self.replace_fn) and 'ff.net' in name:
                    hook = module.register_forward_hook(self.hook_fn)
                    num_modules += 1
                    hooks.append(hook)
        
        elif self.hook_module == 'unet-ffn-1':
            # hook the text encoder
            num_modules = 0
            for name, module in model.unet.named_modules():
                if isinstance(module, self.replace_fn) and 'ff.net' in name:
                    hook = module.register_forward_hook(self.unet_ffn_1_hook_fn)
                    num_modules += 1
                    hooks.append(hook)

        elif self.hook_module == 'attn_key':
            # hook the text encoder
            num_modules = 0
            for name, module in model.unet.named_modules():
                if isinstance(module, self.replace_fn) and 'attn2' in name and 'to_k' in name:
                    hook = module.register_forward_hook(self.unet_attn_layer)
                    num_modules += 1
                    hooks.append(hook)


        elif self.hook_module == 'text':
            # hook the text encoder
            num_modules = 0
            for name, module in model.text_encoder.named_modules():
                if isinstance(module, self.replace_fn) and 'mlp' in name and 'encoder.layers' in name:
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
    
    def test(self, model, ann = 'A brown dog in the snow'):
        # Method to write a test case
        raise NotImplementedError
    
