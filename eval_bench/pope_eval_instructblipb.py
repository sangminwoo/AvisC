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
from transformers import set_seed,AutoTokenizer, AutoModelForCausalLM


from utils import dist_util
from utils.logger import create_logger
from glob import glob

from PIL import Image
import math

from pope_loader import POPEDataSet
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
    parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
    parser.add_argument("--model-path", type=str, default="path/checkpoints/instruct_blip")
    parser.add_argument("--model-base", type=str, default=None)
    
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    
    parser.add_argument("--pope_path", type=str, default="path/to/experiments/data/POPE/coco/coco_pope_random.json")
    parser.add_argument("--data_path", type=str, default=" /path/data/coco/val2014")
    parser.add_argument("--log_path", type=str, default="path/logs")

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", type=str2bool, default=False)
    parser.add_argument("--cd_alpha", type=float, default=1.0)
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)

    parser.add_argument("--use_avisc", type=str2bool, default=False)
    parser.add_argument("--layer_gamma", type=float, default=0.5)
    parser.add_argument("--masking_scheme", type=str, default="zeros")
    parser.add_argument("--lamb", type=float, default=0.95)
    parser.add_argument("--exp_description", type=str, default="")  
    parser.add_argument("--max_token", type=int, default=64)
    parser.add_argument("--cd_beta", type=float, default=0.1)

    
    args = parser.parse_args()
    return args


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def print_acc(pred_list, label_list):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)
    # unknown_ratio = pred_list.count(2) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    return acc, precision, recall, f1, yes_ratio

def recorder(out, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]
    for line in out:

        line = line.replace('.', '')
        line = line.replace(',', '')
        words = line.split(' ')
        if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
            pred_list.append(0)
        else:
            pred_list.append(1)
    
    return pred_list

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
        experiment_dir = f"{args.log_path}/{model_string_name}/avisc_{args.use_avisc}_avisc_{args.use_cd}_cd_alpha_{args.cd_alpha}"  # Create an experiment folder
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"exp_description: {args.exp_description}")
    else:
        logger = create_logger(None)

    # ========================================
    #             Model Initialization
    # ========================================
    logger.info(f"use_cd: {args.use_cd}, method: {args.use_avisc}")
    logger.info(f"pope_path: {args.pope_path}")
    logger.info(f"alpha = {args.cd_alpha}, beta = {args.cd_beta}, max_token = {args.max_token}")

    #### for avisc
    disable_torch_init()
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)
    

    pope_dataset = POPEDataSet(
        pope_path=args.pope_path, 
        data_path=args.data_path,
        trans=vis_processors,
        model='instructblip'
    )
    pope_loader = torch.utils.data.DataLoader(
        pope_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        drop_last=False
    )

    print ("load data finished")

    print("Start eval...")
    pred_list, label_list = [], []
    for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):

        image_path = data["image_path"][0]
        image = data["image"]
        qs = data["query"][0]
        label = data["label"]
        label_list = label_list + list(label)

        # ==============================================
        #             avisc method 
        # ==============================================
        if args.use_cd:
            image_tensor_cd = add_diffusion_noise(image, args.noise_step)
        else:
            image_tensor_cd = None    

        with torch.inference_mode():
            outputs = model.generate(
                {"image": image.to(device), "prompt": qs }, 
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
                )
                
            outputs = outputs
            pred_list = recorder(outputs, pred_list)

        
            logger.info(f"[{image_path}]")
            logger.info(f"Q: {qs}")
            logger.info(f"A: {outputs[0]}")
            if label[0] == 1: logger.info(f"GT: Yes")
            elif label[0] == 0: logger.info(f"GT: No")

    print("[{}]===============================================".format(args.seed)) 
    if len(pred_list) != 0:
        acc, precision, recall, f1, yes_ratio = print_acc(pred_list, label_list)
        logger.info(
            f"acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}, yes_ratio: {yes_ratio}"
        )

if __name__ == "__main__":
    main()
