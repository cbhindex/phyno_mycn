"""
Cell-level feature maps and correlation analysis for Pheno-MYCN components.

For a selected GMM phenotype component, relates the model's projected tile
features to the cell-level morphometric/textural/topological descriptors
extracted from the corresponding tiles (HoverNet-derived ``cell_info`` tables),
and renders the deep-feature correlation, between-class correlation difference,
and top-feature summary figures used in the cell-level analysis.

Adapted from the CLAM heatmap pipeline (Mahmood Lab, GPL-3.0):
https://github.com/mahmoodlab/CLAM. The default paths below are placeholders —
point the ``--classic_feat_csv`` / ``--patch_dir`` / ``--vis_folder`` arguments
at your own cell-feature tables and saved per-tile outputs.

Part of Pheno-MYCN: interpretable histological phenotype discovery associated
with MYCN amplification in paediatric neuroblastoma.

Author:                     Dr Olga Fourkioti   (https://github.com/olgarithmics)
Code review & refactoring:  Dr Binghao Chai     (https://bhchai.com/, https://github.com/cbhindex)

License: GPL-3.0 (see the LICENSE file at the repository root).
"""

from __future__ import print_function
import argparse
import os
import pdb
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import matplotlib.colors as mcolors
from vis_utils.heatmap_utils import initialize_wsi, drawCatHeatmap, drawHeatmap
from wsi_core.batch_process_utils import initialize_df
import torch
import torch.nn as nn
import umap
import matplotlib.pyplot as plt
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler


def hex_to_rgb_mpl_255(hex_color):
    rgb = mcolors.to_rgb(hex_color)
    return tuple([int(x*255) for x in rgb])

def get_default_cmap(n=32):
    colors = [
        '#696969', '#556b2f', '#a0522d', '#483d8b',
        '#008000', '#008b8b', '#000080', '#7f007f',
        '#8fbc8f', '#b03060', '#ff0000', '#ffa500',
        '#00ff00', '#8a2be2', '#00ff7f', '#FFFF54',
        '#00ffff', '#00bfff', '#f4a460', '#adff2f',
        '#da70d6', '#b0c4de', '#ff00ff', '#1e90ff',
        '#f0e68c', '#0000ff', '#dc143c', '#90ee90',
        '#ff1493', '#7b68ee', '#ffefd5', '#ffb6c1'
    ]

    colors = colors[:n]
    label2color_dict = dict(zip(range(n), [hex_to_rgb_mpl_255(x) for x in colors]))
    return label2color_dict


def load_params(df_entry, params):
    for key in params.keys():
        if key in df_entry.index:
            dtype = type(params[key])
            val = df_entry[key]
            val = dtype(val)
            if isinstance(val, str):
                if len(val) > 0:
                    params[key] = val
            elif not np.isnan(val):
                params[key] = val
            else:
                pdb.set_trace()

    return params
def parse_config_dict(args, config_dict):
    if args.save_exp_code is not None:
        config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
    if args.overlap is not None:
        config_dict['patching_arguments']['overlap'] = args.overlap
    return config_dict

def load_params(df_entry, params):
    for key in params.keys():
        if key in df_entry.index:
            dtype = type(params[key])
            val = df_entry[key]
            val = dtype(val)
            if isinstance(val, str):
                if len(val) > 0:
                    params[key] = val
            elif not np.isnan(val):
                params[key] = val
            else:
                pdb.set_trace()

    return params
def parse_config_dict(args, config_dict):
    if args.save_exp_code is not None:
        config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
    if args.overlap is not None:
        config_dict['patching_arguments']['overlap'] = args.overlap
    return config_dict
def hex_to_rgb_mpl_255(hex_color):
    rgb = mcolors.to_rgb(hex_color)
    return tuple([int(x*255) for x in rgb])

def get_default_cmap(n=32):
    colors = [
        '#696969', '#556b2f', '#a0522d', '#483d8b',
        '#008000', '#008b8b', '#000080', '#7f007f',
        '#8fbc8f', '#b03060', '#ff0000', '#ffa500',
        '#00ff00', '#8a2be2', '#00ff7f', '#FFFF54',
        '#00ffff', '#00bfff', '#f4a460', '#adff2f',
        '#da70d6', '#b0c4de', '#ff00ff', '#1e90ff',
        '#f0e68c', '#0000ff', '#dc143c', '#90ee90',
        '#ff1493', '#7b68ee', '#ffefd5', '#ffb6c1'
    ]

    colors = colors[:n]
    label2color_dict = dict(zip(range(n), [hex_to_rgb_mpl_255(x) for x in colors]))
    return label2color_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='HistoTree')
    parser.add_argument('--save_exp_code', type=str, default='CLAM_SB')
    parser.add_argument('--overlap', type=float, default=None)
    parser.add_argument('--path_file', type=str,
                        default='/path/to/results/neuro_myc/fold9.csv', help='')
    parser.add_argument('--path_WSI', type=str,
                        default='/path/to/neuroblastoma_slides/', help='')
    parser.add_argument('--path_graph', type=str,
                        default= '/path/to/data/h5_files/', help='')
    parser.add_argument('--vis_folder', type=str,
                        default= '/path/to/results/prototype_4/', help='')
    parser.add_argument('--config_file', type=str,
                        default='heatmap_config_camelyon.yaml', help='')
    parser.add_argument('--classic_feat_csv', type=str,
                        default='/path/to/results/prototype_4/features/cell_info_updated.csv', help='')
    parser.add_argument('--patch_dir', type=str,
                       default ='/path/to/results/prototype_4/patches',
                       help ='')


    args = parser.parse_args()

    path_graph = args.path_graph
    vis_folder = args.vis_folder
    patch_dir  = args.patch_dir
    classic_feat_csv = args.classic_feat_csv


    filenames = pd.read_csv(args.path_file, sep=',')

    config_path = os.path.join(args.config_file)
    config_dict = yaml.safe_load(open(config_path, 'r'))
    config_dict = parse_config_dict(args, config_dict)

    args = config_dict
    patch_args = argparse.Namespace(**args['patching_arguments'])
    data_args = argparse.Namespace(**args['data_arguments'])
    exp_args = argparse.Namespace(**args['exp_arguments'])

    slides = sorted(os.listdir(data_args.data_dir))
    slides = [slide for slide in slides if data_args.slide_ext in slide]

    os.makedirs(exp_args.raw_save_dir, exist_ok=True)

    test_names = filenames['train'].dropna().tolist()

    classic_feats = pd.read_csv(classic_feat_csv)

    print (classic_feats)


    myc_deep_feats = []
    n_myc_deep_feats = []

    myc_classic_feats = []
    n_myc_classic_feats = []

    for ind, name in enumerate(test_names):
        # if ind == 20:
        #     break
        slide_name = filenames['train'][ind]


        slide_label = filenames['train_label'][ind]

        if data_args.slide_ext not in slide_name:
            slide_name += data_args.slide_ext
        print('\nprocessing: ', slide_name)

        slide_id = slide_name.replace(data_args.slide_ext, '')

        patch_folder = os.path.join(patch_dir, slide_id)

        if not os.path.isdir(patch_folder):
            continue

        r_slide_save_dir = os.path.join(vis_folder, slide_id)
        os.makedirs(r_slide_save_dir, exist_ok=True)

        file_path = os.path.join(path_graph, slide_id + '.h5')

        with h5py.File(file_path, 'r') as h5_file:

                coords_dataset = h5_file['coords']
                coords = np.array(coords_dataset)
                feats_dataset = h5_file['features']
                feats = np.array(feats_dataset)

        feats = torch.load(os.path.join(vis_folder, '{}_gmm_feats.pt'.format(slide_id))).squeeze(0).cpu().detach().numpy()

        for patch in os.listdir(patch_folder):
            if patch.endswith('png'):
                patch_name = patch.split('.')[0]

                patch_key = f"{slide_id}|{patch_name}"


                parts = patch_name.split('_')
                x_coord = int(parts[2])
                y_coord = int(parts[4])

                index = np.where(((coords == [x_coord, y_coord])).all(axis=1))[0]
                deep_feats = feats[index,:]
                classic_df = classic_feats[classic_feats.iloc[:, 0]  == patch_key]



                # classic_df = pd.DataFrame(classic_df)
                # classic_df = classic_df.drop(classic_df.columns[0], axis=1)
                # deep_df = pd.DataFrame(deep_feats, columns=[f"deep_{i}" for i in range(deep_feats.shape[1])])
                #
                # classic_df = (classic_df - classic_df.mean()) / classic_df.std()
                #
                # deep_feats = (deep_feats - deep_feats.mean()) / deep_feats.std()

                if deep_feats.shape[0]>0 and classic_df.shape[0]>0:
                        if slide_label == 0:
                             n_myc_classic_feats.append(classic_df)
                             n_myc_deep_feats.append(deep_feats)

                        else:
                            myc_classic_feats.append(classic_df)
                            myc_deep_feats.append(deep_feats)


    def preprocess_feats(classic_feats_list, deep_feats_list, classic_columns):
        # Stack features
        classic_feats = np.vstack(classic_feats_list)
        deep_feats = np.vstack(deep_feats_list)

        # Build DataFrames
        classic_df = pd.DataFrame(classic_feats, columns=classic_columns)
        deep_df = pd.DataFrame(deep_feats, columns=[f"deep_{i}" for i in range(deep_feats.shape[1])])

        # Drop first column if it's an index or non-feature
        if 'Unnamed: 0' in classic_df.columns or classic_df.columns[0].lower().startswith('unnamed'):
            classic_df = classic_df.drop(columns=classic_df.columns[0])

        # Clean classical features
        #classic_df = classic_df.loc[:, classic_df.isna().mean() < 0.6]
        classic_df = classic_df.fillna(classic_df.mean())
        classic_df = classic_df.loc[:, classic_df.std() > 0]

        # Clean deep features
        #deep_df = deep_df.loc[:, deep_df.isna().mean() < 0.6]
        deep_df = deep_df.fillna(deep_df.mean())
        deep_df = deep_df.loc[:, deep_df.std() > 0]

        # Standardize
        classic_scaled = pd.DataFrame(StandardScaler().fit_transform(classic_df), columns=classic_df.columns)
        deep_scaled = pd.DataFrame(StandardScaler().fit_transform(deep_df), columns=deep_df.columns)

        return classic_scaled, deep_scaled


    myc_plus_classic_scaled, myc_plus_deep_scaled = preprocess_feats(myc_classic_feats, myc_deep_feats,
                                                                     classic_feats.columns.tolist())

    # Process MYC−
    nonmyc_classic_scaled, nonmyc_deep_scaled = preprocess_feats(n_myc_classic_feats, n_myc_deep_feats,
                                                                 classic_feats.columns.tolist())

    # Correlation matrices
    corr_myc_plus = np.dot(myc_plus_deep_scaled.T, myc_plus_classic_scaled) / myc_plus_deep_scaled.shape[0]
    corr_myc_minus = np.dot(nonmyc_deep_scaled.T, nonmyc_classic_scaled) / nonmyc_deep_scaled.shape[0]

    # Wrap into DataFrames
    corr_df_plus = pd.DataFrame(corr_myc_plus, index=myc_plus_deep_scaled.columns,
                                columns=myc_plus_classic_scaled.columns)
    corr_df_minus = pd.DataFrame(corr_myc_minus, index=nonmyc_deep_scaled.columns,
                                 columns=nonmyc_classic_scaled.columns)

    threshold = 0.2

    # MYC+ strong rows
    filtered_plus = corr_df_plus[(corr_df_plus.abs() > threshold).any(axis=1)]
    filtered_plus = filtered_plus.loc[:, (filtered_plus.abs() > threshold).any(axis=0)]

    # MYC− strong rows
    filtered_minus = corr_df_minus[(corr_df_minus.abs() > threshold).any(axis=1)]
    filtered_minus = filtered_minus.loc[:, (filtered_minus.abs() > threshold).any(axis=0)]

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

    # Heatmap for MYC+
    sns.heatmap(filtered_plus, ax=axes[0], cmap='coolwarm', center=0, cbar=True,
                cbar_kws={'ticks': []})  # Remove numbers from colorbar
    axes[0].set_title('MYC+ Correlation (Deep ↔ Classic)')
    axes[0].set_xlabel('Classical Features')
    axes[0].set_ylabel('Deep Features')
    axes[0].tick_params(axis='x', rotation=90)

    # Heatmap for MYC−
    sns.heatmap(filtered_minus, ax=axes[1], cmap='coolwarm', center=0, cbar=True,
                cbar_kws={'ticks': []})  # Remove numbers from colorbar
    axes[1].set_title('MYC− Correlation (Deep ↔ Classic)')
    axes[1].set_xlabel('Classical Features')
    axes[1].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.savefig('/path/to/results/images/deep_feats_corr.png', dpi=300, bbox_inches='tight')
    plt.show()

    # --- Visualize difference (MYC+ minus MYC−) ---
    corr_diff = corr_df_plus - corr_df_minus

    # Filter: top N deep features based on correlation difference
    top_n = 30
    max_diff = corr_diff.abs().max(axis=1)
    top_deep_features = max_diff.sort_values(ascending=False).head(top_n).index

    filtered_diff = corr_diff.loc[top_deep_features]
    threshold = 0.3  # you can change this
    filtered_diff = filtered_diff.loc[:, (filtered_diff.abs() > threshold).any(axis=0)]

    plt.figure(figsize=(14, 8))
    ax = sns.heatmap(filtered_diff, cmap='bwr', center=0)
    plt.title('Deep vs Classic Feature Correlation Difference (MYC+ − MYC−)')
    plt.xlabel('Classical Features')
    plt.ylabel('Deep Features')
    plt.xticks(rotation=45)
    cbar = ax.collections[0].colorbar
    cbar.set_label('Correlation Difference')
    plt.tight_layout()
    plt.savefig('/path/to/results/images/corr_diff.png', dpi=300, bbox_inches='tight')
    plt.show()


    from scipy.stats import ttest_ind
    import pandas as pd

    results = []
    for feat in myc_plus_classic_scaled.columns:
        if feat in nonmyc_classic_scaled.columns:
            stat, pval = ttest_ind(
                myc_plus_classic_scaled[feat],
                nonmyc_classic_scaled[feat],
                equal_var=False,
                nan_policy='omit'
            )
            results.append({'Feature': feat, 'T-stat': stat, 'P-value': pval})

    # Top 10 by absolute T-stat
    top_classic_diff = pd.DataFrame(results).sort_values(by='T-stat', key=abs, ascending=False).head(20)

    # Plot
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 8))
    ax = sns.heatmap(
        filtered_diff,
        cmap='bwr',
        center=0,
        linewidths=0.5,  # adds clear separation between cells
        linecolor='gray'  # subtle color for gridlines
    )

    # Enhancing readability and aesthetics
    plt.title('Deep vs. Classic Feature Correlation Difference (MYC+ − MYC−)', fontsize=16, pad=15)
    plt.xlabel('Classical Features', fontsize=13)
    plt.ylabel('Deep Features', fontsize=13)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)

    # Adjust colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label('Correlation Difference', fontsize=12)

    plt.tight_layout()

    plt.savefig('/path/to/results/images/corr_diff.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.bar(top_classic_diff['Feature'], top_classic_diff['T-stat'], color='indianred')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('T-statistic')
    plt.title('Top Differentiating Classical Features (MYC+ vs Non-MYC)')
    plt.grid(True)
    plt.tight_layout()

    # Saving the plot
    plt.savefig('/path/to/results/images/top_features_plot.png', dpi=300, bbox_inches='tight')

    plt.show()