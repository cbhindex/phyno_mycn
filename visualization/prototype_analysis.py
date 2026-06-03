"""
Prototype (GMM-component) representative-tile analysis for Pheno-MYCN.

Embeds the representative tiles of a selected GMM phenotype component
(prototype) with t-SNE/UMAP over their cell-level features and assembles the
representative-tile montage / scatter figures used to characterise the
component.

Adapted from the CLAM visualisation utilities (Mahmood Lab, GPL-3.0):
https://github.com/mahmoodlab/CLAM. The default paths below are placeholders —
point ``--classic_feat_csv`` / ``--save_name`` and ``image_root`` at your own
cell-feature table and the corresponding tile-image folder.

Part of Pheno-MYCN: interpretable histological phenotype discovery associated
with MYCN amplification in paediatric neuroblastoma.

Code review & refactoring:  Dr Binghao Chai     (https://bhchai.com/, https://github.com/cbhindex)
Author:                     Dr Olga Fourkioti   (https://github.com/olgarithmics)

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
from sklearn.impute import SimpleImputer

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

    parser.add_argument('--overlap', type=float, default=None)
    parser.add_argument('--classic_feat_csv', type=str,
                        default='/path/to/results/fold_9/images/prototype_4/features/cell_info_updated.csv', help='')
    parser.add_argument('--save_name', type=str,
                        default='/path/to/results/top_df_proto_2.png',
                        help='')


    args = parser.parse_args()


    classic_feat_csv = args.classic_feat_csv
    save_name =  args.save_name


    classic_feats = pd.read_csv(classic_feat_csv, index_col=0)


    image_root = '/path/to/results/fold_9/images/prototype_4/patches/'

    # Convention: class_0 = non-amplified, class_1 = MYCN-amplified (label 1 = amp, label 0 = non-amp).
    # myc_classic_feats holds MYCN-amp tile features; n_myc_classic_feats holds non-amp tile features.
    myc_classic_feats = classic_feats[classic_feats.index.str.startswith('class_1|')].values.tolist()
    n_myc_classic_feats = classic_feats[classic_feats.index.str.startswith('class_0|')].values.tolist()



    feat_df = classic_feats.copy()
    feat_df = feat_df.reset_index().rename(columns={'Unnamed: 0': 'filename'})


    # Replace '|' with '/', add .png extension, and prepend the image path
    feat_df['image_path'] = feat_df['filename'].apply(
        lambda x: os.path.join(image_root, x.replace('|', '/') + '.png')
    )


    def preprocess_feats(classic_feats_list, classic_columns):



        classic_df = pd.DataFrame(classic_feats_list, columns=classic_columns)

        classic_df = classic_df.dropna(axis=1, how='all')
        #
        # # 2️⃣ (Optional) Drop features with too many missing values
        # classic_df = classic_df.loc[:, classic_df.isnull().mean() < 0.5]
        #
        # # 3️⃣ Fill remaining NaNs
        # classic_df = classic_df.fillna(classic_df.median(numeric_only=True))


        #classic_df = classic_df.fillna(classic_df.median())
        #classic_df = classic_df.loc[:, classic_df.std() > 0]


        # Standardize
        #classic_scaled = pd.DataFrame(StandardScaler().fit_transform(classic_df), columns=classic_df.columns)


        return classic_df


    myc_plus_classic_scaled = preprocess_feats(myc_classic_feats,
                                                                     classic_feats.columns.tolist())

    # Process MYC−
    nonmyc_classic_scaled = preprocess_feats(n_myc_classic_feats,
                                                                 classic_feats.columns.tolist())

    columns_to_drop = ['hist_gray', 'lbp_hist']

    # Find existing columns to drop
    cols_to_drop = [col for col in columns_to_drop if col in nonmyc_classic_scaled.columns]

    # Drop them (if they exist) from both DataFrames
    if cols_to_drop:
        nonmyc_classic_scaled = nonmyc_classic_scaled.drop(columns=cols_to_drop)
        myc_plus_classic_scaled = myc_plus_classic_scaled.drop(columns=cols_to_drop)
        print(f"✅ Dropped columns: {cols_to_drop}")
    else:
        print("ℹ️ No 'hist_gray' or 'lbp_hist' columns to drop.")

    from scipy.stats import ttest_ind
    import pandas as pd

    results = []
    for feat in myc_plus_classic_scaled.columns:
        if feat in nonmyc_classic_scaled.columns:
            # Check if feature has variance (not all zeros or constant) in both groups
            if myc_plus_classic_scaled[feat].nunique() > 1 and nonmyc_classic_scaled[feat].nunique() > 1:
                stat, pval = ttest_ind(
                    myc_plus_classic_scaled[feat],
                    nonmyc_classic_scaled[feat],
                    equal_var=False,
                    nan_policy='omit'
                )
                print(feat, stat, pval)
                results.append({'Feature': feat, 'T-stat': stat, 'P-value': pval})
            else:
                print(f"⚠️ Skipping {feat}: constant or zero in one/both groups.")


    top_classic_diff = pd.DataFrame(results)
    top_classic_diff = top_classic_diff.sort_values(by='P-value', ascending=True).head(30)

    plt.figure(figsize=(10, 5))
    plt.bar(top_classic_diff['Feature'], top_classic_diff['T-stat'], color='indianred')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('T-statistic')
    #plt.title('Top Differentiating Classical Features (MYC+ vs Non-MYC)')
    #plt.savefig("top_feat_proto_2.png", dpi=300, bbox_inches='tight')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print (top_classic_diff)
    columns_to_keep = (top_classic_diff['Feature'].to_list())
    myc_top_feats = myc_plus_classic_scaled[columns_to_keep]
    non_myc_top_feats = nonmyc_classic_scaled[columns_to_keep]

    combined_top_feats = pd.concat([myc_top_feats, non_myc_top_feats], axis=0)

    labels = [1] * len(myc_top_feats) + [0] * len(non_myc_top_feats)  # 1=MYC+, 0=MYC−
    combined_top_feats = combined_top_feats.fillna(combined_top_feats.median())

    #combined_top_feats = combined_top_feats.dropna(axis=1)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Combine and scale data

    y = np.array([1] * len(myc_plus_classic_scaled) + [0] * len(nonmyc_classic_scaled))

    clf = LogisticRegression(solver='liblinear')
    clf.fit(combined_top_feats, y)

    import shap

    # Initialize SHAP explainer (KernelExplainer is general-purpose)
    explainer = shap.Explainer(clf, combined_top_feats)
    shap_values = explainer(combined_top_feats)

    shap_values_array = shap_values.values if hasattr(shap_values, 'values') else shap_values
    mean_abs_shap = np.abs(shap_values_array).mean(axis=0)

    # Create a DataFrame for easy viewing
    shap_df = pd.DataFrame({
        'Feature': combined_top_feats.columns,
        'Mean |SHAP value|': mean_abs_shap
    })

    # Sort by SHAP value and print top N
    top_n = 20
    shap_df_sorted = shap_df.sort_values(by='Mean |SHAP value|', ascending=False).reset_index(drop=True)



    # shap.plots.heatmap(shap_values, max_display=20, show=False)
    # plt.savefig("shap_hetamap_proto_2.png", dpi=300, bbox_inches='tight')
    # plt.close()
    # shap.plots.violin(shap_values, feature_names=combined_top_feats.columns, plot_type="layered_violin", show=False)
    # plt.savefig("shap_violin_proto_2.png", dpi=300, bbox_inches='tight')
    # plt.close()





