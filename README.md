# MMInA: Benchmarking Multihop Multimodal Internet Agents

<!-- <p align="center">
    <img src="media/logo.png" alt="Logo" width="80px">
    <br>
</p> -->


<p align="center">
<a href="https://mmina.cliangyu.com/">Project Page</a> |
<a href="https://arxiv.org/abs/2404.09992">Paper</a> |
<a href="https://drive.google.com/file/d/1Vf9-g9S5NS5xk53Ye6LFop_IFZm-m27Q/view?usp=drive_link">Dataset</a>
</p>

![Overview](media/teaser.png)


## News
* [04/15/2024] Release the paper and the codebase of MMInA.

## Release Plan

- [ ] More data subsets for multihop tasks
- [ ] Enhanced arguments design for one-stop usage of MMInA
- [x] Paper, codebase, and dataset release

## Installation
### Prerequisites
MMInA is built-off the <a href="https://github.com/web-arena-x/webarena">WebArena environment</a>. The prerequisites are the same as WebArena.

To install the environment, you need to have the following dependencies installed: 

```bash
# one-step installation  from the environment script
git clone https://github.com/shulin16/MMInA
conda env create -f environment.yml

# or install step by step with Python 3.10+
conda create -n mmina python=3.10; conda activate mmina
pip install -r requirements.txt
playwright install
pip install -e .

# optional, dev only
pip install -e ".[dev]"
mypy --install-types --non-interactive browser_env agents evaluation_harness
pip install pre-commit
pre-commit install
```

## mmina Dataset Structure
The mmina dataset is a collection of tasks that require long-chain reasoning over multimodal information. The dataset is divided into several subsets, each of which contains tasks with different numbers of hops. The dataset is stored in the following structure:

``` bash
Data Root
└── normal/ # All of them are 2-hop tasks.
    └── x.json
    ...
└── multi567/ # All 5-hop, 6-hop, 7-hop tasks are here.
    └── x.json
    ...
└── compare/ # All tasks in this folder need to answer a comparable question first.
    └── x.json
    ...
└── multipro/ # All 8-hop, 9-hop, 10-hop tasks are here.
    └── x.json
    ...
└── shopping/ # All tasks here are about items in OneStopMarket
    └── x.json
    ...
└── wikipedia/ # All tasks here are limited in wikipedia.
    └── x.json
    ...
```

To use our dataset, which is designed as multimodal web agent tasks, you can download from this [Google Drive link](https://drive.google.com/file/d/1wlJQTR4I2DPuZ6S5pIxTIRnWZJVb5QNU/view?usp=drive_link). Please refer to [this section](##Usage) for detailed instructions for download.

If you want to test different subsets of the dataset, you can specify the subset name in the `domain` argument when running the code. For example, if you want to test the `shopping` subset, you can set the `domain` argument as `shopping`.


## Usage
### Quick Start
#### 1. Prepare the environment
You can modify the prepare.sh file to set the environment variable such as your working directory, API keys (from OpenAI, Google etc.). Then run the following command to prepare the environment.
```bash
bash prepare.sh
```

#### 2. Download the data
```bash
cd $WORK_DIR
mkdir mmina
curl -o mmina.zip https://drive.google.com/file/d/1wlJQTR4I2DPuZ6S5pIxTIRnWZJVb5QNU/view?usp=drive_link
unzip mmina.zip && rm mmina.zip
```

#### 3. Test the developed agents
If you want to try agents without history memories:
```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
--test_start_idx 1 --test_end_idx 10 \
--provider custom --model MODEL_NAME \
--domain DOMAIN_NAME \
--result_dir RESULT_DIR 
```

If you want to try agents with history memories, you have to set the `hist` tag as `True`, and specify the history number and the history folder where the history data is stored. Usually

```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
--test_start_idx 1 --test_end_idx 10 \
--provider custom --model MODEL_NAME \
--domain DOMAIN_NAME \
--result_dir RESULT_DIR \
--hist True --hist_num NUM --hist_fold HIST_FOLDER
```

#### 4. Test your own agents
You can also implement customized LLMs or VLMs as agents to test out the long-chain reasoning ability of the models. After downloading the model weights, the agents should be implemented in [`agent.py`](agent/agent.py) under `agent/` folder. 

Remeber to initializa a new agent instance and modify the respective configs in [`run.py`](run.py)  to test your own agents.

``` python
# Code snippets to initialize and customize the agent configs
llm_config.gen_config["temperature"] = args.temperature
llm_config.gen_config["top_p"] = args.top_p
llm_config.gen_config["context_length"] = args.context_length
llm_config.gen_config["max_tokens"] = args.max_tokens
llm_config.gen_config["stop_token"] = args.stop_token
llm_config.gen_config["max_obs_length"] = args.max_obs_length   
```

## Citation
If you use our environment or data, please cite our paper:
```
@misc{zhang2024mmina,
      title={MMInA: Benchmarking Multihop Multimodal Internet Agents}, 
      author={Ziniu Zhang and Shulin Tian and Liangyu Chen and Ziwei Liu},
      year={2024},
      eprint={2404.09992},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement
Our benchmark is built upon the <a href="https://webarena.dev/">WebArena</a> environment, which is a standalone, self-hostable web environment for building autonomous agents with textual inputs. We thank the authors for their great work and the open-source codebase.
