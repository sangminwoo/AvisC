import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random

TOTAL_LEN = 15220
GEN_LEN = 1004

class AMBERDataSet(Dataset):
    def __init__(self, json_path, data_path, trans, model, num_gen=0, num_dis=5000):
        self.json_path = json_path
        self.data_path = data_path
        self.trans = trans
        self.model = model
        self.num_gen = num_gen
        self.num_dis = num_dis  
        
        gen_idx = random.sample(range(1, GEN_LEN), self.num_gen)
        dis_idx = random.sample(range(GEN_LEN + 1, TOTAL_LEN), self.num_dis)
        

        image_list, query_list, id_list = [], [], []
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)


        
        
        for line in data:
            if line['id'] in gen_idx or line['id'] in dis_idx:
                image_list.append(line['image'])
                query_list.append(line['query'])
                id_list.append(line['id'])
            

        assert len(image_list) == len(query_list)
        assert len(image_list) == len(id_list)

        self.image_list = image_list
        self.query_list = query_list
        self.id_list = id_list

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.image_list[index])
        
        if self.model == 'llava':
            raw_image = Image.open(image_path)
            image = self.trans.preprocess(raw_image, return_tensor='pt')['pixel_values'][0]
            query = self.query_list[index]
            id = self.id_list[index]
            return {"image": image, "query": query, "id": id, "image_path": image_path}
            
        elif self.model == 'qwen-vl':
            raw_image = Image.open(image_path).convert("RGB")
            image = self.trans(raw_image)
            query = self.query_list[index]
            id = self.id_list[index]
            return {"image": image, "query": query, "id": id, "image_path": image_path}
        
        elif self.model == 'instructblip':
            raw_image = Image.open(image_path).convert("RGB")
            image_tensor = self.trans['eval'](raw_image)
            query = self.query_list[index]
            id = self.id_list[index]
            return {"image": image_tensor, "query": query, "id": id, "image_path": image_path}

