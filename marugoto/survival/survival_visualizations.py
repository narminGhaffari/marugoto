from pathlib import Path
import h5py
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import torch
from torch import Tensor
from fastai.learner import load_learner
from fastai.vision.learner import Learner, load_learner

from marugoto.survival import mil
from marugoto.survival.mil.model import MILModel

import openslide
from openslide.deepzoom import DeepZoomGenerator
from math import floor
import os
from argparse import ArgumentParser




def parse_arg():
    parser = ArgumentParser()

    parser.add_argument(
        '-f', '--feature_dir',
        type = Path,
        required = True,
        help = 'features directory (h5 files)'
    )
    parser.add_argument(
        '-sd', '--slide_directory',
        type = Path,
        required = True,
        help = 'slides directory (svs files)'
    )
    parser.add_argument(
        '-m', '--model_path',
        type = Path,
        required = True,
        help = 'model path (best_model.pth)'
    )
    parser.add_argument(
        '-o','--output_path',
        type = Path,
        required = True,
        help = 'Path to the directory where results will be saved'
    )
    parser.add_argument(
        '-sc', '--score_csv',
        type = Path,
        required = True,
        help = 'the Marugoto Survival Deploy output file'
    )

    parser.add_argument(
        '-nt', '--n_tiles',
        type = int,
        default = 10,
        help = 'number of tiles per patient to store'
    )

    parser.add_argument(
        '-np', '--n_patients',
        type = int,
        default = 10,
        help = 'number of patient to plot'
    )

    parser.add_argument(
        '-tl', '--top_list',
        type = bool,
        default = True,
        help = 'a boolean representing whether top patients must be plotted, otherwise bottom patients'
    )

    parser.add_argument(
        '-nf', '--n_features',
        type = int,
        default = 1024,
        help = 'number of features extracted (e.g 1024 UNI, 768 CTransPath)'
    )

    parser.add_argument(
        '-g', '--geojson',
        type = bool,
        default = False,
        help = 'boolean representing whether the geojson of the top tiles must be saved (geojson could be imported into QuPath to observe the context of the top tiles)'
    )
    return parser.parse_args()


def convert_coord(img, slide, df_slide):
    """
    convert_coord function convert coord to the svs dimensions (slide)

    :param img: image thumbnails (numpy array)
    :param slide: opened whole-slide image from openslide
    :param df_slide: data frame with tile coordinates
    :return: Data frame with tile coordinates in WSI dimensions
    """ 
    power = int(slide.properties["openslide.objective-power"])

    h, w = img.shape[:2]
    w_slide, h_slide = slide.dimensions
    h_factor, w_factor = h / h_slide, w / w_slide
    resolution = power / 20 
    df_slide["x_img"] = df_slide.x * w_factor * resolution * 224
    df_slide["y_img"] = df_slide.y * h_factor * resolution * 224
    
    return df_slide


def plot_attention_heatmap(
    slide,
    attention_df: pd.DataFrame,
    out_dir: Path,
    name

) -> None:
    """
    plot_attention_heatmap Plot attention heatmap

    :param slide: opened whole-slide image from openslide
    :param attention_df: data frame with coordinates of tiles and attention
    :param out_dir: Path to directory where results are stored
    :param name: the name of the output results
    """

    img_pil = slide.get_thumbnail((1000, 1000))
    img = np.array(img_pil)

    pred_df = attention_df.copy()
    pred_df = convert_coord(img, slide, pred_df)

    fig, axes = plt.subplots(1, 3, figsize = (15, 5))
    axes = axes.flatten()
    ax1, ax2, ax3 = axes

    ax1.imshow(img)
    ax1.set_title("Original image")
    
    ax2.scatter(pred_df.x_img, pred_df.y_img, s = 5, c = pred_df.attention, cmap = "coolwarm")
    ax2.invert_yaxis()
    ax2.set_title("Attention")
    
    ax3.scatter(pred_df.x_img, pred_df.y_img, s = 5, c = pred_df.attention, cmap = "coolwarm")
    ax3.imshow(img)
    ax3.set_title("Image with Attention")

    # Both subplot same size
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{name}_attention_heatmaps.png")
    extent = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f"{out_dir}/{name}_superpose_attention_heatmaps.png", bbox_inches = extent)
    extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f"{out_dir}/{name}_image.png", bbox_inches = extent)
    plt.close()


def get_n_toptiles(
    name,
    slide,
    out_dir: Path,
    coords: Tensor,
    scores: Tensor,
    n: int = 15,
    geojson: bool=False
) -> None:

    """
    get_n_toptiles Generate and save top tiles
    :param name: the name of the output results
    :param slide: opened whole-slide image from openslide
    :param outdir: Path to the directory where the results will be saved
    :param coord: a dataframe with coordinates of tiles
    :param scores: a dataframe with the scores of the tiles
    :param n: number of tiles to save
    :param geojson: boolean representing whether the coordinates should be saved in geojson format

    """ 
    print(f"\tWriting tiles images of {name}")
    coord_df = coords.numpy()
    scores_df = scores.numpy()
    
    dz = DeepZoomGenerator(slide, tile_size = 224, overlap = 0)
    
    pos_idx = coord_df
    pos_idx = pos_idx.astype(int)
    power = int(slide.properties["openslide.objective-power"])
    tile_size = floor(11.2 * power)

    pos_list_img = [dz.get_tile(pts[0], pts[1:3]) for pts in pos_idx]
    
    geoTopTiles = '{"type":"FeatureCollection", "features":['
    i = 0
    for img,sc,pts in zip(pos_list_img, scores_df, pos_idx):
        img.save(f"{out_dir}/Scores_{sc}_{name}.png")
        i = i + 1

        if geojson :
            xleft = int(pts[1]) * tile_size
            xright = xleft + tile_size
            yleft = int(pts[2]) * tile_size
            yright = yleft + tile_size
            tileGeojson = '{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[%i, %i],[%i, %i],[%i, %i],[%i, %i],[%i, %i]]]},"properties":{"object_type":"annotation","classification":{"name":"TopTiles", "colorRGB":-65536},"isLocked":false}},' % (xleft, yleft, xleft, yright, xright, yright, xright, yleft, xleft, yleft)
            geoTopTiles = "".join([geoTopTiles, tileGeojson])
    
    if geojson : 
        geoTopTiles = geoTopTiles[:-1]
        geoTopTiles = "".join([geoTopTiles, ']}'])
        output_geojson = f"{out_dir}/{name}.geojson"
        file_geojson = open(output_geojson, "w")
        file_geojson.write(geoTopTiles)
        file_geojson.close()
                


def get_cohort_df(
     score_csv: Path, feature_dir: str
) -> pd.DataFrame:

    """
    get_cohort_df Generate cohort df
    :param score_csv: Path of slide dataframe (column must be at least FILENAME PATIENT)
    :param features_dir:  features directory path

    :return: Dataframe score_csv column and features_path
    """
    
    h5s = set(feature_dir.glob('*.h5'))
    suffix = Path(score_csv).suffix
    if suffix == '.csv':
        df = pd.read_csv(score_csv)
    else:
        df = pd.read_excel(score_csv)

    try:
        df = df.drop('slide_path', axis = 1)
    except:
        pass
    h5_df = pd.DataFrame(h5s, columns = ["slide_path"])
    h5_df["FILENAME"] = h5_df.slide_path.map(lambda p: p.stem)
    df = df.merge(h5_df, on = "FILENAME")


    # reduce to one row per patient with list of slides in `df['slide_path']`
    patient_df = df.groupby("PATIENT").first().drop(columns = "slide_path")

    patient_slides = df.groupby("PATIENT").slide_path.apply(list)
    df = patient_df.merge(
        patient_slides, left_on = "PATIENT", right_index = True
    ).reset_index()

    return df


def plot_att_tiles_heatmap(
    feature_dir: Path,
    model_path: Path,
    slide_dir: Path,
    score_csv: Path,
    n_patients=15,
    n_tiles: int = 15,
    out_dir: Path = None,
    top_list: bool = True,
    n_feats : int = 768,
    geojson=False
) -> None:

    """
    _plot_tiles_heatmap Generate/save top tiles heatmap and dataframe results
    :param features_dir: features directory path
    :param model_path:  model (.pth) path
    :param slide_dir: slide directory path (svs)
    :param score_csv: the  Marugoto Survival Deploy output file( at least column SCORE FILENAME PATIENT)
    :param n_patients: number of top patients to plot
    :param n_tiles: number of tiles to store
    :param out_dir: Path to the directory where results will be saved
    :param top_list: a boolean representing whether top patients must be plotted, otherwise bottom patients
    :param n_feats: number of extracted features (768 for Ctranspath 1024 for UNI)
    :param geojson: boolean representing whether the geojson of the top tiles must be saved
    
    """ 


    feature_dir = Path(feature_dir)
    if not out_dir:
        out_dir = Path(score_csv).parent
    else:
        out_dir = Path(out_dir)

    df = get_cohort_df(
        score_csv=score_csv,
        feature_dir=feature_dir,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learn = MILModel(n_feats, 1) #2048 the len of feature
    learn.load_state_dict(torch.load(model_path,map_location = torch.device('cpu')))
    learn.to(device)

    encoder = learn.encoder.eval()

    attention = learn.attention.eval()

    head = learn.head.eval()
    

    top_ts = []
    iter_df = df.copy()
    i_patients = np.arange(n_patients)

    if top_list: # Sort patients by descending scores
        sort_df = iter_df.sort_values(by = ['SCORE'], ascending = False)
        top_patients = iter_df.nlargest(iter_df.shape[0], "SCORE")["slide_path"].values
    else : # Sort patients by ascending scores
        sort_df = iter_df.sort_values(by = ['SCORE'], ascending = True)
        top_patients = iter_df.nsmallest(iter_df.shape[0], "SCORE")["slide_path"].values


    
    for i_patient, slide_paths in zip(
        i_patients, top_patients
    ):  

        name = f"{sort_df.iloc[i_patient].loc['FILENAME']}"

        patient_dir = out_dir / f"{i_patient}_{name}"
        patient_dir.mkdir(exist_ok = True, parents = True)

        feats, coords, sizes = [], [], []
        for slide_path in slide_paths: 
            with h5py.File(Path(slide_path), "r") as f:
                feats.append(torch.from_numpy(f["feats"][:]).float())
                sizes.append(len(f["feats"]))
                coords.append(torch.from_numpy(f["coords"][:]))
        feats, coords = torch.cat(feats), torch.cat(coords)

        # get the attention, score for each tile
        encs = encoder(feats).squeeze()
        patient_atts = torch.softmax(attention(encs).squeeze(), 0).detach()
        patient_atts *= len(patient_atts)
        patient_scores = head(encs).detach().squeeze()
        patient_weighted = patient_atts * patient_scores
        n = len(patient_scores)

        df = pd.DataFrame(patient_weighted.numpy())

        coord_df = pd.DataFrame(coords.numpy())

        df.to_csv(f"{patient_dir}/weighted_attention.csv")
        
        coord_df.to_csv(f"{patient_dir}/tiles_coordinates.csv")
        coord_df = coord_df.rename(columns = {0: "z", 1:"x", 2:"y"})
        df = df.rename(columns={0: "attention"})

        result = pd.concat([coord_df, df], axis=1)
        result.to_csv(f"{patient_dir}/result.csv")
        
        # Sort tiles by scores
        top_idxs = patient_weighted.topk(n_tiles,largest=top_list).indices
        slide = openslide.OpenSlide(str(f"{slide_dir}/{sort_df.iloc[i_patient].loc['FILENAME']}.svs")) 
        
        plot_attention_heatmap(
            slide = slide,
            attention_df = result,
            out_dir = patient_dir,
            name = name
        )
        
        get_n_toptiles(
            name = name,
            slide = slide,
            out_dir = patient_dir,
            scores = patient_weighted[top_idxs],
            coords = coords[top_idxs],
            n = n_tiles,
            geojson = geojson
        )
    









if __name__ == "__main__":
    args = parse_arg()
    out_dir = args.output_path

    if not out_dir:
        out_dir = Path(patient_preds_csv).parent
    else:
        out_dir = Path(out_dir)

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    top_res= "Top" if args.top_list else "Bottom"
    print(f"Writing {top_res} results to: {out_dir}\n")


    plot_att_tiles_heatmap(
        feature_dir = args.feature_dir,
        model_path = args.model_path,
        slide_dir = args.slide_directory,
        score_csv = args.score_csv,
        n_patients = args.n_patients,
        n_tiles = args.n_tiles,
        out_dir = out_dir,
        top_list = args.top_list,
        n_feats = args.n_features,
        geojson = args.geojson
    )

    print("Done!")





    