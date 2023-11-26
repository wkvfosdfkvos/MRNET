
<h4 align="right">
    <p>
        | <b>English</b> |
    </p>
</h4>

<h1 align="center">
    <b>MRNet</b>
</h1>

<h2 align="center">
    <b>Multi-Route Neural Network to extract musical representation at various time scales and processing levels
</h2>

<h3 align="left">
	<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=Python&logoColor=white"></a>
	<a href="https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_22-08.html#rel-22-08"><img src="https://img.shields.io/badge/22.08-2496ED?style=for-the-badge&logo=Docker&logoColor=white"></a>
	<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white"></a>
	<a href="https://huggingface.co/"><img src="https://github.com/Jungwoo4021/OS-KDFT/blob/main/readme/icon_hugging_face.png?raw=true"></a>
	</p>
</h3>

# Introduction
Pytorch code for following paper:
* **Title** : Multi-Route Neural Network to extract musical representation at various time scales and processing levels
* **Autor** :  Jungwoo Heo, Hyun-seo Shin, Ju-ho Kim, Chan-yeong Lim, Kyo-won Koo, and Ha-Jin Yu

This repository presents MRNet experiments that performs genre classification in the FMA dataset partition. 

## Paper abstract
Music information retrieval (MIR) aims to enable users to extract the musical information contained within the music, such as genre, singer, and tempo. Music information contains short-term features like pitch and long-term features like melody; the extraction of distinct types of musical information necessitates varying levels of processing. Considering these characteristics, this paper proposes a multi-route neural network (MRNet) that processes information at various time scales and diverse processing levels. Specifically, MRNet extracts the features of different temporal scales by stacking convolution modules with different dilations on the time axis and extracts features at varying depths by handling outputs from multiple levels of processing. In the music classification (a signature MIR task), MRNet achieved accuracies of 94.5%, 56.6%, 63.2%, and 71.3% for the respective target clusters within the GTZAN, FMA small, FMA large, and Melon datasets. Experiments are available on GitHub. 

# Prerequisites

## Environment Setting
* We used 'nvcr.io/nvidia/pytorch:22.08-py3' image of Nvidia GPU Cloud for conducting our experiments. 

* Python 3.8

* Pytorch 1.13.0+cu117

* Torchaudio 0.13.0+cu117

# Datasets
* FMA: A Dataset For Music Analysis

# Run experiment

### STEP1. Set system arguments
First, you need to set system arguments. You can set arguments in `arguments.py`. Here is list of system arguments to set.

```python
1. 'path_log': path of saving experiment logs.
    CAUTION!! 
        If a directory already exists in the path, it remove the existing directory.

2. 'path_FMA': path where FMA dataset is stored.

```

### STEP2. Set system arguments
### Additional logger
We have a basic logger that stores information in local. However, if you would like to use an additional online logger (wandb or neptune):

1. In `arguments.py`
```python
# Wandb: Add 'wandb_user' and 'wandb_token'
# Neptune: Add 'neptune_user' and 'neptune_token'
# input this arguments in "system_args" dictionary:
# for example
'wandb_user'   : 'user-name',
'wandb_token'  : 'WANDB_TOKEN',

'neptune_user'  : 'user-name',
'neptune_token' : 'NEPTUNE_TOKEN'
```

2. In `main.py`

```python
# Just remove "#" in logger

logger = LogModuleController.Builder(args['name'], args['project'],
        ).tags(args['tags']
        ).description(args['description']
        ).save_source_files(args['path_scripts']
        ).use_local(args['path_log']
        #).use_wandb(args['wandb_user'], args['wandb_token'] <- here
        #).use_neptune(args['neptune_user'], args['neptune_token'] <- here
        ).build()
```

### STEP3. RUN
Run main.py in scripts.

```python
>>> python main.py
```
