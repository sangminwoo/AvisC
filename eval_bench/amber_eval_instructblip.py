import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/experiments')
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM

from utils import dist_util
from utils.logger import create_logger
from glob import glob

from PIL import Image
import math

from amber_loader import AMBERDataSet
from lavis.models import load_model_and_preprocess
# import kornia
from transformers import set_seed
from avisc_utils.vcd_add_noise import add_diffusion_noise
from avisc_utils.avisc_sample import evolve_avisc_sampling
evolve_avisc_sampling()

torch.multiprocessing.set_sharing_strategy('file_system')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="AMBER-Adv evaluation on LVLMs.")
    parser.add_argument("--model-path", type=str, default="path/checkpoints/instruct_blip")
    parser.add_argument("--model-base", type=str, default=None)
    
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    
    parser.add_argument("--json_path", type=str, default="path/to/to/experiments/AMBER/data/query/query_all.json")
    parser.add_argument("--data_path", type=str, default="path/dataset/AMBER/image")
    parser.add_argument("--log_path", type=str, default="path/logs/amber")

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", type=str2bool, default=False)
    parser.add_argument("--cd_alpha", type=float, default=1.0)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=7, help="specify the gpu to load the model.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)

    parser.add_argument("--use_avisc", type=str2bool, default=True)
    parser.add_argument("--layer_gamma", type=float, default=0.5)
    parser.add_argument("--masking_scheme", type=str, default="zeros")
    parser.add_argument("--lamb", type=float, default=0.99)
    parser.add_argument("--exp_description", type=str, default="..")
    parser.add_argument("--max_token", type=int, default=64)
    parser.add_argument("--use_m3id", type=str2bool, default=True)
    args = parser.parse_args()
    return args


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def recorder(out):
    NEG_WORDS = ["No", "not", "no", "NO"]

    out = out.replace('.', '')
    out = out.replace(',', '')
    words = out.split(' ')
    if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
        return "No"
    else:
        return "Yes"


def main():
    args = parse_args()

    # Setup DDP:
    dist_util.setup_dist(args)
    device = dist_util.device()

    # Setup an experiment folder:
    if dist.get_rank() == 0:
        os.makedirs(
            args.log_path, exist_ok=True
        )  # Make results folder (holds all experiment subfolders)
        model_string_name = args.model_path.split("/")[-1]
        experiment_dir = f"{args.log_path}"  # Create an experiment folder
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"exp_description: {args.exp_description}")
    else:
        logger = create_logger(None)

    # ========================================
    #             Model Initialization
    # ========================================
    print('Initializing Model')
    logger.info(f"use_cd: {args.use_cd}, method: {args.use_avisc}, layer_gamma: {args.layer_gamma}, masking_scheme: {args.masking_scheme}, lamb: {args.lamb}")

    
    disable_torch_init()
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)
    
    
    
    AMBER_dataset = AMBERDataSet(
        json_path=args.json_path, 
        data_path=args.data_path,
        trans=vis_processors,
        model='instructblip'
    )
    AMBER_loader = torch.utils.data.DataLoader(
        AMBER_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        drop_last=False
    )

    print ("load data finished")

    print("Start eval...")
    result_json_path = os.path.join(experiment_dir, "Amber_result.json")
    
    result = []
    
    
    for batch_id, data in tqdm(enumerate(AMBER_loader), total=len(AMBER_loader)):
        image = data["image"]
        qs = data["query"]
        ids = data["id"]
        image_path = data["image_path"]

        # ==============================================
        #             Text prompt setting
        # ==============================================
        
        if args.use_cd:
            image_tensor_cd = add_diffusion_noise(image, args.noise_step)
        else:
            image_tensor_cd = None    
        
        input_ids = []
        

        # ==============================================
        #             Image tensor setting
        # ==============================================
        

        with torch.inference_mode():
            outputs = model.generate(
                {"image": image.to(device), "prompt": qs[0]}, 
                use_nucleus_sampling=True,
                num_beams=1,
                top_p=args.top_p,
                repetition_penalty=1,
                images_cd=image_tensor_cd.half().to(device) if image_tensor_cd is not None else None,
                cd_beta = args.cd_beta,
                use_avisc=args.use_avisc,
                layer_gamma=args.layer_gamma,
                masking_scheme=args.masking_scheme,
                lamb=args.lamb,
                max_length=args.max_token,
                cd_alpha=args.cd_alpha,
                use_m3id=args.use_m3id,
                )
            outputs = outputs            

            for ip, q, a in zip(image_path, qs, outputs):
                    logger.info(f"[{ip}]")
                    logger.info(f"Q: {q}")
                    logger.info(f"A: {a}")
            
            for batch_id in range(len(ids)):
                if ids[batch_id] > 1004: 
                    outputs[batch_id] = recorder(outputs[batch_id])
                    
            for id, a in zip(ids, outputs):
                item = {
                    "id": int(id),
                    "response": a
                }
                result.append(item)
                
                    
    with open(result_json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
