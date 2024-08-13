# ConceptPrune
Code for the paper - ConceptPrune: Concept Editing in Diffusion Models via Skilled Neuron Pruning ([arxiv preprint](https://arxiv.org/abs/2405.19237v1))

## Introduction
While large-scale text-to-image diffusion models have demonstrated impressive image-generation capabilities, there are significant concerns about their potential misuse for generating unsafe content, violating copyright, and perpetuating societal biases. Recently, the text-to-image generation community has begun addressing these concerns by editing or unlearning undesired concepts from pre-trained models. However, these methods often involve data-intensive and inefficient fine-tuning or utilize various forms of token remapping, rendering them susceptible to adversarial jailbreaks. In this paper, we present a simple and effective training-free approach, ConceptPrune, wherein we first identify critical regions within pre-trained models responsible for generating undesirable concepts, thereby facilitating straightforward concept unlearning via weight pruning. Experiments across a range of concepts including artistic styles, nudity, object erasure, and gender debiasing demonstrate that target concepts can be efficiently erased by pruning a tiny fraction, approximately 0.12% of total weights, enabling multi-concept erasure and robustness against various white-box and black-box adversarial attacks.


## Experiments

### Environment Setup
Create the environment from the ```environment.yml``` file.

```conda env create -f environment.yml```


```conda activate concept-prune```

We recommend using ```diffusers v0.29.2``` as the results may change for different versions.

### Code Structure


The file structure is as follows

&nbsp; ```configs``` - Contains ```.yaml`` file for basic arguments. These arguments can be changed within the scripts using argument parsers.

&nbsp; ```datasets``` - Contains txt or csv file with prompts for different concepts

&nbsp; ```neuron_receivers``` - Contains classes to hook Feed Forward network (FFN) modules within the Unet to record neuron activations

&nbsp; ```wanda``` - Contains scripts to calculate WANDA pruing metric introduced in [Sun et. al](https://arxiv.org/abs/2306.11695) for FFN weights

&nbsp; ```utils``` - Basic helper functions

&nbsp; ```benchmarking``` - Scripts to run all the benchmarks in the paper for different concepts


### Pruning the model using WANDA


To obtain a pruned model for a concept ```<target>```, run the following - 

1. Discover skilled neurons for a concept

    ```
    python wanda/wanda.py --target <target> --skill_ratio 0.01
    ```

    ```<target>``` is the concept that we want to erase. Replace  ```<target``` with any of - 


    &nbsp; 1. Artist Styles - ```Van Gogh, Monet, Pablo Picasso, Da Vinci, Salvador Dali```. Example - base prompt = ```a cat``` and target prompt = ```a cat in the style of Van Gogh```

    &nbsp; 2. Nudity - ```naked```. Example - base prompt = ```a photo of a man``` and target prompt = ```a photo of a naked man```

    &nbsp; 3. Objects (Imagnette classes) - ```parachute, golf ball, garbage truck, cassette player, church, tench, english springer, french horn, chain saw, gas pump```. 

    &nbsp; &nbsp; Example - base prompt = ```a room``` and target prompt = ```a parachute in a room```

    &nbsp; 4. Gender reversal - ```male, female```. Example - base prompt = ```a son``` and target prompt = ```a daughter``` for female to male reversal.

    &nbsp; 5. Memorization - ```memorize_$i$```. For this concept, prompts are loaded from corresponding to datasets/memorize_0.txt. Please pass ```--target_file memorize_$i$``` for this concept.


    The argument ```skill_ratio``` denotes the sparsity level which defines the top-k% neurons considered for WANDA pruning. This command saves skilled neurons discovered for every timestep and layer in a different .pkl file as a sparse matrix. 


2. Check if removing skilled neurons from all timesteps and layers removes the concept.

    We first check whether hyper-parameters like ```skill_ratio``` used in the previous step are optimal for concept removal. We attach hook functions to FFN layers for every timestep and apply the pruning mask. The following command will save images after skilled neurons are removed.

    ```
    python wanda/remove_neurons.py --target <target> --skill_ratio <skill_ratio>
    ```

3. Next, we take a union over skilled neurons for the first few timesteps.

    Run the follwing command to obtain the pruned model.
    ```
    python wanda/save_union_over_time.py --target <target> --timesteps <tau> --skill_ratio <skill_ratio>
    ```

    We provide the values of these hyper-parameters in Table 7 in the Appendix for every concept.


### Benchmarks

#### Baselines

Train concept erasure baselines - [UCE](https://github.com/rohitgandikota/unified-concept-editing), [FMN](https://github.com/SHI-Labs/Forget-Me-Not), [ESD](https://github.com/rohitgandikota/erasing), [Concept-Ablation](https://github.com/nupurkmr9/concept-ablation) using their respective repositories. n our code base, we provide code to evalaute these baselines on concept-erasure benchmarks for different concepts. In the following experiments, ausedd ```uce, esd, fmn, concept-ablation``` for ```<baseline>``` respectively to run the above baselines. 

### Download Checkpoints

We will provide checkpoints on Hugging Face soon!

#### Evaluate ConceptPrune

1. Artist Styles

    To evaluate artist style erasure for ```Van Gogh, Monet, Pablo Picasso, Da Vinci, Salvador Dali`` for ConceptPrune, run 
    ```
    python benchmarking/artist_erasure.py --target <target> --baseline concept-prune --ckpt_name <path to checkpoint>
    ```
    We created a dataset of 50 prompts using ChatGPT for different artists such that each prompt contains the painting name along with the name of the artist. These propmts are available in ```datasets/```. The script saves images  and a json files with CLIP metric reported in the paper in the ```results/``` folder.

2. Nudity 

    To evaluate nudity erasure with ConceptPrune on the I2P dataset, run
    ```
    python benchmarking/nudity_eval.py --eval_dataset i2p --baseline 'concept-prune' --gpu 0 --ckpt_name <path to checkpoint>
    ```

    To run ConceptPrune on black-box adversarial prompt datasets, [MMA](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_MMA-Diffusion_MultiModal_Attack_on_Diffusion_Models_CVPR_2024_paper.pdf) and [Ring-A-Bell](https://arxiv.org/abs/2310.10012), replace ```i2p``` with ```mma``` and ```ring-a-bell``` respectively.

    We evaluate nudity in images using the [NudeNet detector](https://pypi.org/project/nudenet/). The script saves images and a json files with NudeNet scores reported in the paper in the ```results/``` folder.


3. Object Erasing

    To evaluate object erasure with ConceptPrune, run
    ```
    python benchmarking/object_erase.py --target <object> --baseline concept-prune --removal_mode erase --ckpt_name <path to checkpoint>
    ```
    
    To check interference of concept removal with unrelated classes, run
    ```
    python benchmarking/object_erase.py --target <object> --baseline concept-prune --removal_mode keep --ckpt_name <path to checkpoint>
    ```

    where ```<object>``` is the name of a class in ImageNette classes.  he script saves images and a json files with ResNet50 accuracies reported in the paper in the ```results/``` folder.


4. Gender reversal

    To evaluate gender reversal from Female to Male, run

    ```
    python benchmarking/gender_reversal.py --target male --ckpt_name <path to checkpoint>
    ```

    Replace ```male``` with ```female``` to reverse gender from Male to Female. We calculate the success of gender reversal using CLIP to classify between males females. The script saves images in the ```results/``` folder for 250 seeds.



5. COCO evaluation

    To evaluate ConceptPrune on COCO dataset, run

    ```
    python benchmarking/eval_coco.py --target <target> --baseline concept-prune --ckpt_name <path to checkpoint>
    ``` 

5. Memorization

    To evaluate ConceptPrune on COCO dataset, run

    ```
    python benchmarking/inference_mem.py --target memorize_$i$ --baseline concept-prune --ckpt_name <path to checkpoint>
    ``` 

    This will save images and calculate SSCD and CLIP score and store the results in a json file. We run this script for 10 different seeds for every model and report average performance.


### Cite us!

If you find our paper useful, please consider citing our work. 
```
@article{chavhan2024conceptpruneconcepteditingdiffusion,
      title={ConceptPrune: Concept Editing in Diffusion Models via Skilled Neuron Pruning}, 
      author={Ruchika Chavhan and Da Li and Timothy Hospedales},
      year={2024},
      journal={ArXiv}
}
```

```
@article{chavhan2024conceptpruneconcepteditingdiffusion,
      title={Memorized Images in Diffusion Models share a Subspace that can be Located and Deleted}, 
      author={Ruchika Chavhan and  Ondrej Bohdal and Yongshuo Zong and Da Li and Timothy Hospedales},
      year={2024},
      journal={ArXiv}
}
```

### Contact

Please contact ruchika.chavhan@ed.ac.uk for any questions!