"""
Whole-slide GMM-component / attention / energy heatmaps for Pheno-MYCN.

Renders per-slide visualisations over the original H&E WSI from the per-tile
outputs saved at test time:
  * ``<slide>_gmm.pt``       — per-tile GMM responsibilities ([n_tiles, K]);
  * ``<slide>_gmm_feats.pt`` — projected GMM features;
  * ``<slide>_att.pt``       — per-tile MIL attention;
  * ``<slide>_energy.pt``    — per-tile GMM free-energy.
Tile coordinates are read from the per-slide ``.h5`` patch graph and the
heatmap-rendering parameters come from a CLAM-style config
(``heatmap_config_camelyon.yaml``).

Adapted from the CLAM heatmap pipeline (Mahmood Lab, GPL-3.0):
https://github.com/mahmoodlab/CLAM. The default paths below are placeholders —
point the ``--path_file`` / ``--path_WSI`` / ``--path_graph`` / ``--vis_folder``
arguments at your own data and saved per-tile outputs.

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
from scipy.spatial.distance import cdist
import seaborn as sns


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
    parser.add_argument('--path_file', type=str, default='/path/to/results/neuro_myc/fold9.csv', help='')
    parser.add_argument('--path_WSI', type=str, default='/path/to/neuroblastoma_slides/', help='')
    parser.add_argument('--path_graph', type=str,
                        default= '/path/to/data/h5_files/', help='')
    parser.add_argument('--vis_folder', type=str,
                        help= '', default ='results/fold_9/pt_outputs/')
    parser.add_argument('--config_file', type=str,
                        default='heatmap_config_camelyon.yaml', help='')



    args = parser.parse_args()

    path_graph = args.path_graph
    vis_folder = args.vis_folder

    filenames = pd.read_csv(args.path_file, sep=',')

    config_path = os.path.join(args.config_file)
    config_dict = yaml.safe_load(open(config_path, 'r'))
    config_dict = parse_config_dict(args, config_dict)

    args = config_dict
    patch_args = argparse.Namespace(**args['patching_arguments'])
    data_args = argparse.Namespace(**args['data_arguments'])
    exp_args = argparse.Namespace(**args['exp_arguments'])
    heatmap_args = argparse.Namespace(**args['heatmap_arguments'])
    sample_args = argparse.Namespace(**args['sample_arguments'])

    patch_size = tuple([patch_args.patch_size for i in range(2)])
    step_size = tuple((np.array(patch_size) * (1 - patch_args.overlap)).astype(int))
    print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1],
                                                                                  patch_args.overlap,
                                                                                  step_size[0], step_size[1]))
    preset = data_args.preset
    def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False,
                      'keep_ids': 'none', 'exclude_ids': 'none'}
    def_filter_params = {'a_t': 50.0, 'a_h': 8.0, 'max_n_holes': 10}
    def_vis_params = {'vis_level': -1, 'line_thickness': 250}
    def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}


    if preset is not None:
        preset_df = pd.read_csv(preset)
        for key in def_seg_params.keys():
            def_seg_params[key] = preset_df.loc[0, key]

        for key in def_filter_params.keys():
            def_filter_params[key] = preset_df.loc[0, key]

        for key in def_vis_params.keys():
            def_vis_params[key] = preset_df.loc[0, key]

        for key in def_patch_params.keys():
            def_patch_params[key] = preset_df.loc[0, key]

    slides = sorted(os.listdir(data_args.data_dir))
    slides = [slide for slide in slides if data_args.slide_ext in slide]
    df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params,
                       use_heatmap_args=True)

    mask = df['process'] == 1
    process_stack = df[mask].reset_index(drop=True)
    total = len(process_stack)
    print('\nlist of slides to process: ')
    print(process_stack.head(len(process_stack)))

    os.makedirs(exp_args.raw_save_dir, exist_ok=True)


    blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': patch_size,
                         'custom_downsample': patch_args.custom_downsample, 'level': patch_args.patch_level,
                         'use_center_shift': heatmap_args.use_center_shift}
    attentions_scores = []
    energies = []
    slide_labels =[]
    nmyc_feats= []
    myc_feats = []
    myc_labels = []
    nmyc_labels = []
    nmyc_probs = []
    myc_probs = []
    test_names = filenames['train'].dropna().tolist()

    for ind, name in enumerate(test_names):

        slide_name = filenames['train'][ind]

        slide_label = filenames['train_label'][ind]


        if data_args.slide_ext not in slide_name:
            slide_name += data_args.slide_ext
        print('\nprocessing: ', slide_name)

        slide_id = slide_name.replace(data_args.slide_ext, '')

        r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code)
        #r_slide_save_dir = os.path.join(vis_folder, slide_id)
        os.makedirs(r_slide_save_dir, exist_ok=True)

        if os.path.exists(os.path.join(r_slide_save_dir, '{}_att.tiff'.format(slide_id))):
                        continue

        if isinstance(data_args.data_dir, str):
            slide_path = os.path.join(data_args.data_dir, slide_name)

        mask_file = os.path.join(r_slide_save_dir, slide_id + '_mask.pkl')

        seg_params = def_seg_params.copy()

        filter_params = def_filter_params.copy()
        vis_params = def_vis_params.copy()

        keep_ids = str(seg_params['keep_ids'])
        if len(keep_ids) > 0 and keep_ids != 'none':
            seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
        else:
            seg_params['keep_ids'] = []

        exclude_ids = str(seg_params['exclude_ids'])
        if len(exclude_ids) > 0 and exclude_ids != 'none':
            seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
        else:
            seg_params['exclude_ids'] = []

        for key, val in seg_params.items():
            print('{}: {}'.format(key, val))

        for key, val in filter_params.items():
            print('{}: {}'.format(key, val))

        for key, val in vis_params.items():
            print('{}: {}'.format(key, val))

        print('Initializing WSI object')
        # wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params,
        #                             filter_params=filter_params)

        # wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]
        #
        # vis_patch_size = tuple(
        #     (np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))
        #
        # if vis_params['vis_level'] < 0:
        #     best_level = wsi_object.wsi.get_best_level_for_downsample(64)
        #     vis_params['vis_level'] = best_level
        # vis_params['line_thickness'] = 250
        #
        file_path = os.path.join(path_graph, slide_id + '.h5')

        with h5py.File(file_path, 'r') as h5_file:

                coords_dataset = h5_file['coords']
                coords = np.array(coords_dataset)
                feats_dataset = h5_file['features']
                feats = np.array(feats_dataset)


        feats = torch.load(os.path.join(vis_folder, '{}_gmm_feats.pt'.format(slide_id))).squeeze(0).cpu().detach().numpy()

        att_matrix = torch.load(os.path.join(vis_folder, '{}_att.pt'.format(slide_id)))

        scores = att_matrix.cpu()

        scores = scores.detach().numpy()

        # wsi_kwargs = {'patch_size': patch_size, 'step_size': step_size,
        #               'custom_downsample': patch_args.custom_downsample, 'level': patch_args.patch_level,
        #               'use_center_shift': heatmap_args.use_center_shift}
        #
        # heatmap_save_name = '{}_blockmap.tiff'.format(slide_id)
        #
        # heatmap_1 = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap,
        #                         alpha=heatmap_args.alpha,
        #                         use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
        #                         thresh=-1, patch_size=vis_patch_size, convert_to_percentiles=True)
        #
        # heatmap_1.save(os.path.join(r_slide_save_dir, '{}_att.tiff'.format(slide_id)), format='TIFF')

        probs = torch.load(os.path.join(vis_folder, '{}_gmm.pt'.format(slide_id))).squeeze(0).numpy()

        labels = np.argmax(probs, axis=1)

        energy = torch.load(os.path.join(vis_folder, '{}_energy.pt'.format(slide_id))).squeeze(0).numpy()
        energies.append(energy)
        slide_labels.append(slide_label)

        unique_labels, counts = np.unique(labels, return_counts=True)

        for label, count in zip(unique_labels.tolist(), counts.tolist()):
            print(f"Label {label}: {count} instances")


        label2color_dict = get_default_cmap(6)

        # heatmap_1 = drawCatHeatmap(labels, coords,
        #                         label2color_dict,
        #                         slide_path,
        #                         wsi_object = wsi_object,
        #                         alpha=heatmap_args.alpha,
        #                         use_holes=True,
        #                         vis_level=-1,
        #                         blur = False,
        #                         blank_canvas= True,
        #                         patch_size = vis_patch_size)
        #
        # heatmap_1.save(os.path.join(r_slide_save_dir, '{}_cat_map.tiff'.format(slide_id)), format='TIFF')

        #
        # top_k = 10
        # for proto_id in range(probs.shape[1]):
        #
        #     # if proto_id != 4:
        #     #     continue
        #
        #     proto_indices = np.where(labels == proto_id)[0]
        #
        #     proto_scores = scores[:, proto_indices]
        #     proto_scores = proto_scores.squeeze(0)
        #
        #     num_to_select = min(top_k, len(proto_indices))
        #     #
        #     # if num_to_select <  20:
        #     #     continue
        #
        #     top_indices = proto_indices[np.argsort(proto_scores)[::-1][:num_to_select]]
        #
        #     for idx, patch_idx in enumerate(top_indices):
        #         output_data = []
        #         s_coord = coords[patch_idx]
        #         s_prob = probs[patch_idx, proto_id]
        #
        #         patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level, (patch_args.patch_size,
        #                                                                                     patch_args.patch_size)).convert(
        #             'RGB')
        #
        #         patch_dir = os.path.join(r_slide_save_dir, 'class_{}'.format(str(slide_label)))
        #         #patch_dir = os.path.join(r_slide_save_dir, 'patches')
        #         os.makedirs(patch_dir, exist_ok=True)
        #         proto_slide_save_dir = os.path.join(patch_dir, f'prototype_{proto_id}')
        #         os.makedirs(proto_slide_save_dir, exist_ok=True)
        #         patch.save(os.path.join(proto_slide_save_dir, '{}_x_{}_y_{}.png'.format(idx, s_coord[0], s_coord[1])))

        def get_features(X):

            proto_labels=[]
            feats = []
            f_probs = []
            top_k = 100
            for proto_id in range(probs.shape[1]):
                proto_indices = np.where(labels == proto_id)[0]

                if len(proto_indices) == 0:
                    continue  # Skip empty prototypes

                proto_scores = scores[:, proto_indices]
                proto_scores = proto_scores.squeeze(0)

                num_to_select = min(top_k, len(proto_indices))

                top_indices = proto_indices[np.argsort(proto_scores)[::-1][:num_to_select]]

                feats.extend(X[top_indices])
                f_probs.extend(probs[top_indices])
                proto_labels.extend([proto_id] * len(top_indices))


            if len(feats) > 0:
                feats = np.vstack(feats)
                f_probs = np.vstack(f_probs)
                proto_labels = np.vstack(proto_labels)
            else:
                feats = np.array([])
                f_probs = np.array([])
                proto_labels = np.array([])

            return np.array(feats), np.array(proto_labels), f_probs



        X = feats
        normalized_colors = {k: tuple([v_i / 255.0 for v_i in v] + [0.4]) for k, v in label2color_dict.items()}
        X_f, f_labels, f_probs = get_features(X)

        if slide_label == 0:
            nmyc_labels.append(f_labels)
            nmyc_feats.append(X_f)
            nmyc_probs.append(f_probs)
        else:
            myc_labels.append(f_labels)
            myc_feats.append(X_f)
            myc_probs.append(f_probs)


    import matplotlib.pyplot as plt
    import seaborn as sns

    myc_probs = np.vstack(myc_probs)
    nmyc_probs = np.vstack(nmyc_probs)

    myc_feats = np.vstack(myc_feats)
    nmyc_feats = np.vstack(nmyc_feats)
    #
    num_components = myc_probs.shape[1]
    fig, axes = plt.subplots(1, num_components, figsize=(15, 5), sharey=True)

    for i in range(num_components):
        sns.histplot(nmyc_probs[:, i], kde=True, bins=30, color='red', label="No MYC", ax=axes[i], log_scale=True,  stat="density")
        sns.histplot(myc_probs[:, i], kde=True, bins=30, color='blue', label="MYC", ax=axes[i], log_scale=True,  stat="density")
        axes[i].set_title(f'GMM Component {i+1}')
        axes[i].legend()

    plt.savefig('gmm_plot_train.png', bbox_inches='tight')
    #plt.savefig(os.path.join(exp_args.raw_save_dir, 'gmm_plot_val.png') ,bbox_inches='tight')
    plt.show()


    myc_df = pd.DataFrame(myc_probs, columns=[f'Comp_{i+1}' for i in range(num_components)])
    nmyc_df = pd.DataFrame(nmyc_probs, columns=[f'Comp_{i+1}' for i in range(num_components)])

    myc_df['Class'] = "MYC+"
    nmyc_df['Class'] = "MYC-"

    # Combine both DataFrames
    responsibility_df = pd.concat([myc_df, nmyc_df])

    # Melt DataFrame for Seaborn
    df_melted = responsibility_df.melt(id_vars=['Class'], var_name='Component', value_name='Responsibility')
    #
    # Boxplot
    plt.figure(figsize=(12, 6))
    sns.violinplot(
        x="Component",
        y="Responsibility",
        hue="Class",
        data=df_melted,
        split=True,
        inner="point",  # show means/medians as points
        scale="width"
    )
    #plt.title("Violin Plot of GMM Responsibilities for MYC+ and MYC- Classes")
    plt.xlabel("Component")
    plt.ylabel("Responsibility")
    plt.legend(title="Class",loc='upper right' )
    plt.savefig('violin_plot_train.png', bbox_inches='tight', dpi=300)
    #plt.savefig(os.path.join(exp_args.raw_save_dir, 'violin_plot_val.png'), bbox_inches='tight', dpi=300)
    plt.show()
    #plt.suptitle("Distribution of GMM Responsibilities for Class 0 and Class 1")



