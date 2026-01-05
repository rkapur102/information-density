'''
MOST OF THIS CODE IS EITHER COPIED EXACTLY FROM OR ADAPTED FROM EITHER 1) : 
Code for CLIPScore (https://arxiv.org/abs/2104.08718)
@inproceedings{hessel2021clipscore,
  title={{CLIPScore:} A Reference-free Evaluation Metric for Image Captioning},
  author={Hessel, Jack and Holtzman, Ari and Forbes, Maxwell and Bras, Ronan Le and Choi, Yejin},
  booktitle={EMNLP},
  year={2021}
}
AND/OR 2) : https://github.com/elisakreiss/contextual-description-evaluation/blob/main/metrics/clipscore/clipscore.py
'''

import argparse
import clip
import torch
from PIL import Image
from sklearn.preprocessing import normalize
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import tqdm
import numpy as np
import sklearn.preprocessing
import collections
import os
import pathlib
import json
import random
import generation_eval_utils
import pprint
import warnings
from packaging import version
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     'candidates_json',
    #     type=str,
    #     help='Candidates json mapping from image_id --> candidate.')

    # parser.add_argument(
    #     'image_dir',
    #     type=str,
    #     help='Directory of images, with the filenames as image ids.')

    parser.add_argument(
        '--shuffle',
        type = bool,
        default = False,
        help = 'Optional: shuffled image--text pairs')

    parser.add_argument(
        '--references_json',
        default = None,
        help = 'Optional references json mapping from image_id --> [list of references]')

    parser.add_argument(
        '--compute_other_ref_metrics',
        default = 1,
        type = int,
        help = 'If references is specified, should we compute standard reference-based metrics?')

    parser.add_argument(
        '--save_per_instance',
        default = None,
        help = 'if set, we will save per instance clipscores to this file')

    args = parser.parse_args()

    if isinstance(args.save_per_instance, str) and not args.save_per_instance.endswith('.json'):
        print('if you\'re saving per-instance, please make sure the filepath ends in json.')
        quit()
    return args


class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, prefix = 'A photo depicts'):
        self.data = data
        self.prefix = prefix
        if self.prefix[-1] != ' ':
            self.prefix += ' '

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate = True).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation = Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image': image}

    def __len__(self):
        return len(self.data)


def extract_all_captions(captions, model, device, batch_size = 256, num_workers = 16):
    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions),
        batch_size = batch_size, num_workers = num_workers, shuffle = False)
    all_text_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['caption'].to(device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def extract_all_images(images, model, device, batch_size = 64, num_workers = 16):
    print("extract_all_images len(images): ", len(images))
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images),
        batch_size = batch_size, num_workers = num_workers, shuffle = False)
    all_image_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['image'].to(device)
            if device == 'cuda':
                b = b.to(torch.float16)
            all_image_features.append(model.encode_image(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features


def get_clip_score(model, images, candidates, device, w = 2.5):
    '''
    get standard image-text clipscore.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    '''
    if isinstance(images, list):
        # need to extract image features
        images = extract_all_images(images, model, device)

    candidates = extract_all_captions(candidates, model, device)

    # as of numpy 1.21, normalize doesn't work properly for float16
    if version.parse(np.__version__) < version.parse('1.21'):
        images = sklearn.preprocessing.normalize(images, axis = 1)
        candidates = sklearn.preprocessing.normalize(candidates, axis = 1)
    else:
        warnings.warn(
            'due to a numerical instability, new numpy normalization is slightly different than paper results. '
            'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
        images = images / np.sqrt(np.sum(images**2, axis = 1, keepdims = True))
        candidates = candidates / np.sqrt(np.sum(candidates**2, axis = 1, keepdims = True))
    per = w * np.clip(np.sum(images * candidates, axis = 1), 0, None)
    print(per)
    result_list = per.tolist()
    return result_list


def get_refonlyclipscore(model, references, candidates, device):
    '''
    The text only side for refclipscore
    '''
    if isinstance(candidates, list):
        candidates = extract_all_captions(candidates, model, device)

    flattened_refs = []
    flattened_refs_idxs = []
    for idx, refs in enumerate(references):
        flattened_refs.extend(refs)
        flattened_refs_idxs.extend([idx for _ in refs])

    flattened_refs = extract_all_captions(flattened_refs, model, device)

    if version.parse(np.__version__) < version.parse('1.21'):
        candidates = sklearn.preprocessing.normalize(candidates, axis = 1)
        flattened_refs = sklearn.preprocessing.normalize(flattened_refs, axis = 1)
    else:
        warnings.warn(
            'due to a numerical instability, new numpy normalization is slightly different than paper results. '
            'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')

        candidates = candidates / np.sqrt(np.sum(candidates**2, axis = 1, keepdims = True))
        flattened_refs = flattened_refs / np.sqrt(np.sum(flattened_refs**2, axis = 1, keepdims = True))

    cand_idx2refs = collections.defaultdict(list)
    for ref_feats, cand_idx in zip(flattened_refs, flattened_refs_idxs):
        cand_idx2refs[cand_idx].append(ref_feats)

    assert len(cand_idx2refs) == len(candidates)

    cand_idx2refs = {k: np.vstack(v) for k, v in cand_idx2refs.items()}

    per = []
    for c_idx, cand in tqdm.tqdm(enumerate(candidates)):
        cur_refs = cand_idx2refs[c_idx]
        all_sims = cand.dot(cur_refs.transpose())
        per.append(np.max(all_sims))

    return np.mean(per), per


def main():
    with open('clipscores.json', 'r') as f:
        data = json.load(f)

    # intermediary files from the prompt generation process that we have not included, all data was combined into processed_data_cache.pkl
    output_file = './result_clipscores.txt'
    paths_file = './paths.txt' # image paths for the 5000 images sampled from COCO
    captions_file = './captions.txt'

    image_paths = []
    with open(paths_file, 'r') as file:
        for line in file:
            image_paths.append(line.strip())

    filtered_entries = [entry for entry in data if entry.get('image_path') in image_paths]
    data = json.dumps(filtered_entries, indent = 4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cpu':
        warnings.warn(
            'CLIP runs in full float32 on CPU. Results in paper were computed on GPU, which uses float16. '
            'If you\'re reporting results on CPU, please note this when you report.')
    model, transform = clip.load("ViT-B/32", device = device, jit = False)
    model.eval()

    image_feats = extract_all_images(image_paths, model, device)

    captions = [
        caption if caption is not None else ""
        for img_data in data.values()
        for caption in [
            img_data['generated_caption_2'],
            img_data['generated_caption_3'],
            img_data['generated_caption_4'],
            img_data['chosen_coco_caption']
        ]
    ]

    with open(captions_file, 'w') as file:
        counter = 1
        for row in captions:
            file.write(str(counter) + '\n')
            file.write(row + '\n')
            counter += 1

    batch_size = 100
    total_pairs = len(captions)

    with open(output_file, 'a') as f:
        for start_idx in range(0, total_pairs, batch_size):
            end_idx = min(start_idx + batch_size, total_pairs)
            current_captions = np.repeat(captions[start_idx:end_idx], 5000, axis = 0)
            current_image_feats = np.tile(image_feats, (batch_size, 1))

            batch_scores = get_clip_score(
                model,
                current_image_feats,
                current_captions,
                device
            )

            batch_scores = list(map(float, batch_scores))
            for score in batch_scores:
                f.write(f"{score:.6f}\n")

            print(f"Processed pairs {start_idx} to {end_idx} out of {total_pairs}")


if __name__ == '__main__':
    main()
