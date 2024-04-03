#!/usr/bin/env python3

__author__ = "Omar S. M. El Nahhas"
__copyright__ = "Copyright 2024, Kather Lab"
__license__ = "MIT"
__maintainer__ = ["Omar S. M. El Nahhas"]
__email__ = "omar.el_nahhas@tu-dresden.de"

# %%
from collections import namedtuple
from functools import partial

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from fastai.vision.all import load_learner
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import os
import glob
import argparse
from fastai.basics import load_learner
import glob
# %%
def get_toptile_coords(scores, coords):
    coords_list=np.array([str(tuple(foo)) for foo in coords])
    return pd.DataFrame({'gradcam': scores,
     'coords': coords_list
    })

def toptile_coords(list_of_df):
    full_df=pd.concat(list_of_df)
    mean_scores_coords=full_df.groupby('coords').mean('gradcam').reset_index()
    top_n_tile_coords = (mean_scores_coords.sort_values(by='gradcam',ascending=False))
    return top_n_tile_coords


def vals_to_im(scores, coords, stride):
    size = coords.max(0)[::-1] 
    if scores.ndimension() == 1:
        im = np.zeros(size)
    elif scores.ndimension() == 2:
        im = np.zeros((*size, scores.size(-1)))
    else:
        raise ValueError(f"{scores.ndimension()=}")
    for score, c in zip(scores, coords[:]):
        x, y = c[0], c[1]
        im[y:(y+stride), x:(x+stride)] = score.cpu().detach().numpy()
    return im

def save_qkv(_module, _args, output):
    global q, k
    qkv = output.chunk(3, dim=-1)
    n_heads = 8
    q, k, _ = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=n_heads), qkv)

def main(learner_path, feature_name_pattern, output_folder):
    # Load the learner
    learn = load_learner(learner_path)

    # Find all matching feature files
    feature_files = glob.glob(feature_name_pattern)
    learn.model.cuda().eval()

    for file in feature_files:
        f_hname=os.path.basename(file)
        print(f'processing {f_hname}...')
        if os.path.exists(output_folder+f"/{f_hname}_toptiles_layer_0.csv"):
            print("Exists. Skipping...")
            continue

        f = h5py.File(file)
        coords = f["coords"][:]

        xs = np.sort(np.unique(coords[:, 0]))
        stride = np.min(xs[1:] - xs[:-1])
        q, k = None, None

        for transformer_layer in range(1):
            img_coll=[]
            coords_coll=[]
            for attention_head_i in range(8): #8heads in model
                feats = torch.tensor(f["feats"][:]).cuda().float()
                feats.requires_grad = True
                embedded = learn.fc(feats.unsqueeze(0).float())
                with_class_token = torch.cat([learn.cls_token, embedded], dim=1)

                with learn.model.transformer.layers[transformer_layer][
                    0
                ].fn.to_qkv.register_forward_hook(save_qkv):
                    transformed = learn.transformer(with_class_token)[:, 0]
                a = F.softmax(q @ k.transpose(-2, -1) * 0.125, dim=-1)

                # calculate attention gradcam
                a[0, attention_head_i, 0, 1:].sum().backward()
                gradcam = (feats.grad * feats).abs().sum(-1)
                X = vals_to_im(gradcam, coords, stride)
                if transformer_layer==0:
                    coords_coll.append(get_toptile_coords(gradcam.cpu().detach().numpy(),coords))

                if (X.max() - X.min()) > 0:
                    X_std = (X - X.min()) / (X.max() - X.min())
                else:
                    continue
                X_scaled = X_std * (255)
                img_coll.append(X_scaled) #for the aggregated heatmap later, like Firas' paper
            img=np.mean(img_coll,axis=0)
            print(f_hname)
            if transformer_layer==0:
                df = toptile_coords(coords_coll)
                df.to_csv(output_folder+f"/{f_hname}_toptiles_layer_{transformer_layer}.csv")
            plt.imshow(img, alpha=np.float32(img != 0))
            plt.axis("off")
            plt.savefig(output_folder+f"/{f_hname}_attention_map_layer_{transformer_layer}.png",dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Oncotype project data.")
    
    parser.add_argument("--learner_path", type=str, required=True,
                        help="Path to the exported learner (.pkl) file.")
    parser.add_argument("--feature_name_pattern", type=str, required=True,
                        help="Pattern to match feature files (e.g., '/path/to/files/*.h5').")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to the output folder where results will be saved.")
    
    args = parser.parse_args()

    main(args.learner_path, args.feature_name_pattern, args.output_folder)