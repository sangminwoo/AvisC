# Don't Miss the Forest for the Trees: Attentional Vision Calibration for Large Vision Language Models

<!-- Arxiv Link, Project Link -->
<div style='display:flex; gap: 0.25rem; '>
<a href="https://arxiv.org/abs/"><img src="https://img.shields.io/badge/arXiv-2312.15980-b31b1b.svg"></a>
<a href="https://github.io/AvisC"><img src="https://img.shields.io/badge/Project%20Page-online-brightgreen"></a>
<a href='LICENSE'><img src='https://img.shields.io/badge/License-Apache 2.0-g.svg'></a>
</div>

TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO



## Setup

```bash
conda create AvisC python=3.10
conda activate AvisC
git clone https://github.com/sangminwoo/AvisC.git
cd AvisC
pip install -r requirements.txt
```


## Models (TODO)

* **LLaVA-1.5**
* **InstructBLIP**
* **Qwen-VL**

*About model checkpoints preparation*
TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO



## Evaluation (TODO)

* **POPE (llava, qwen-vl)**: `bash eval_bench/scripts/pope_eval.sh` 
  - Need to specify "model", "model_path"
* **POPE (instructblip)**: `bash experiments/cd_scripts/pope_eval.sh` 
  - Need to specify "model", "model_path"
* **LLaVA-Bench (llava)**: `bash eval_bench/scripts/llava_bench_eval.sh`
  - Need to specify "model", "model_path"
* **MME**: `bash experiments/cd_scripts/mme_eval.sh`
  - Need to specify "model", "model_path"

*About datasets preparation*
TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO



## Acknowledgments
This codebase borrows from most notably [VCD](https://github.com/DAMO-NLP-SG/VCD), [OPERA](https://github.com/shikiw/OPERA), and [LLaVA](https://github.com/haotian-liu/LLaVA).
Many thanks to the authors for generously sharing their codes!



## Citation
If you find this repository helpful for your project, please consider citing our work :)

```
@article{placeholder2024,
  title={placeholder}, 
  author={placeholder},
  journal={placeholder},
  year={2024},
}
```