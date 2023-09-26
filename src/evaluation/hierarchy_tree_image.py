import os

from timeit import default_timer as timer
from datetime import timedelta

from torchvision.datasets.folder import find_classes
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import chi2


import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from PIL import Image

import graphviz

def mahalanobis(x):
    mu = x.mean(0)
    cov = np.cov(x.T)
    #mah_dist = stats.spatial.distance.mahalanobis(x, mu, np.linalg.inv(cov))
    #return mah_dist

    x_mu = x - mu
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left, x_mu.T)
    return mahal.diagonal()

    """
    diff = x - mu
    top = np.dot(diff, np.linalg.inv(cov))
    top = -0.5 * np.einsum('ij,ji->i', top, diff.T)
    top = np.exp(top)
    bottom = np.sqrt(np.power(2*np.pi, x.shape[1]) * np.linalg.det(cov))
    print(top[0])
    print(top.mean())
    return top / bottom
    """

def create_hierarchical_tree_vis(features, hierarchy_labels, reduction_method="tsne", hierarchy_label_map=None, 
    top_k=6, output="", rerun_reduction=False, verbose=False, remove_outliers=False):
    assert reduction_method in ["tsne", "pca"], f"{reduction_method} not supported."
    os.makedirs(output, exist_ok=True)

    def log(x):
        if verbose:
            print(x)

    lvls = len(hierarchy_labels[0])
    base_colors = [ x for x in mcolors.BASE_COLORS.values() ]

    def run_reduction(in_feats):
        log(f"Running {reduction_method} on features: {in_feats.shape}. This may take some time.")
        start = timer()
        out_feats = reduce.fit_transform(in_feats)
        end = timer()
        log(f"{reduction_method} completed: {timedelta(seconds=end-start)}")

        return out_feats
    
    if reduction_method == "tsne":
        reduce = TSNE(2)
    elif reduction_method == "pca":
        reduce = PCA(2)

    if not rerun_reduction:
        features = run_reduction(features)

    data_queue = [(features, hierarchy_labels, 0)]
    largest_name = None
    graph_data = []
    while len(data_queue) > 0:
        past_largest_name = largest_name
        cur_plot_lvl = past_largest_name if past_largest_name is not None else "Root"

        feats, lbls, lvl = data_queue.pop(0)
        if rerun_reduction:
            reduced_feats = run_reduction(feats)
        else:
            reduced_feats = feats
        if hierarchy_label_map is not None:
            lvl_lbl_map = hierarchy_label_map[lvl]
        
        log(f"Plotting Level {cur_plot_lvl}")

        lbl_lengths = []
        sorted_lbls = sorted(list(set(lbls[:, lvl])))
        for lbl in sorted_lbls:
            idx = lbls[:, lvl] == lbl
            lbl_lengths.append([lbl, len(reduced_feats[idx])])
        lbl_lengths = sorted(lbl_lengths, key=lambda x: x[1], reverse=True)

        fig = plt.figure(figsize=(14, 10))
        ax = plt.gca()
        plt.setp(ax.spines.values(), lw=5, color='black') # Set border width
        title_parts = cur_plot_lvl.split('_')
        fig_title = '-'.join(title_parts[-2 if len(title_parts) > 1 else 0:])
        fig.suptitle(f"({lvl+1}) {fig_title}", fontsize=50, y=0.98)
        
        #plt.title(f"Hierarchy Level: {cur_plot_lvl}")
        #plt.axis('off')
        most_feats = None
        most_lbls = None
        highest_num = 0
        c = 0
        #print(sorted_lbls)
        for lbl in sorted_lbls:
            if top_k > 0:
                if lbl not in np.array(lbl_lengths)[:top_k, 0]: continue
            idx = lbls[:, lvl] == lbl
            feat = reduced_feats[idx]
            if hierarchy_label_map is not None:
                name = lvl_lbl_map[str(lbl)]
            else:
                name = f"{lbl.split('_')[-1]}"

            if len(feat) > highest_num:
                highest_num = len(feat)
                if rerun_reduction:
                    most_feats = feats[idx]
                else:
                    most_feats = feat
                most_lbls = lbls[idx]
                largest_name = lbl
                if hierarchy_label_map is not None:
                    largest_name = lvl_lbl_map[lbl]

            plot_feat = feat
            if remove_outliers:
                #mu = np.mean(feat, axis=0)
                #cov = np.cov(feat, rowvar=False)
                #var = np.var(feat, axis=0)
                #z_score = (feat - mu) / var
                #filtered_idx = np.logical_and(z_score >= -3, z_score <= 3)
                #filtered_idx = np.sum(filtered_idx, axis=1) == filtered_idx.shape[1]
                #plot_feat = feat[filtered_idx]
                #gauss_feat = calc_gauss(feat, mu, cov)
                mah_dist = mahalanobis(feat)
                # https://www.statology.org/mahalanobis-distance-python/
                # Typically a p-value that is < .001 is considered an outlier
                p = 1 - chi2.cdf(mah_dist, feat.shape[1]-1)
                filtered_idx = p >= 0.001
                plot_feat = feat[filtered_idx]
               
            plt.scatter(plot_feat[:, 0], plot_feat[:, 1], label=name, color=base_colors[c], alpha=0.5) # TODO
            

            c += 1

        # Add to queue
        if (lvl+1) < lvls:
            data_queue.append((most_feats, most_lbls, lvl+1))

        if lvl > 0:
            save_path = os.path.join(output, f"depth_{lvl}_{past_largest_name}.png")
        else:
            save_path = os.path.join(output, f"depth_{lvl}.png")

        plt.legend(loc='upper right', ncols=2, markerscale=3, fontsize=24)
        # Turn off axis ticks
        plt.xticks([], [])
        plt.yticks([], [])

        fig.tight_layout()
        plt.savefig(save_path)
        graph_data.append((save_path, lvl))
        plt.close()


    two_by_three = True
    imgs = [np.array(Image.open(data[0])) for data in graph_data]
    if two_by_three:
        rows = [np.concatenate((imgs[i], imgs[i+1], imgs[i+2]), axis=1) for i in range(0, len(imgs)-1, 3)]
        final_image = np.concatenate(rows, axis=0)
    else:
        final_image = np.concatenate(imgs, axis=1)

    # Assume 6 images
    #for path, lbl in graph_data[1:]:
    #    img = np.array(Image.open(path))
    #    full_img = np.concatenate((full_img, img), axis=0)
    Image.fromarray(final_image).save(os.path.join(output, "full_image.png"))
    Image.fromarray(final_image).save(os.path.join(output, "full_image.pdf"))

    """
    g = graphviz.Digraph('Hierarchy', filename=os.path.join(output, 'heirarchy_image.gv'))
    edges = []

    for path, lvl in graph_data:
        if lvl > 0:
            edges.append([str(lvl-1), str(lvl)])
        #g.node(dp['name'], image=dp['path'], shape='rectangle', scale='false', fontsize='0', imagescale='true', fixedsize='true', height='1.5', width='3')
        g.node(str(lvl), image=path, shape='rectangle', fontsize='0')
    
    for edge in edges:
        g.edge(edge[0], edge[1])
    
    g.render(os.path.join(output, 'heirarchy_image'), format="png", view=False)
    """

def _get_hierarchy_lbl_map(root):
    classes, class_to_idx = find_classes(root)
    idx_to_class = {val: key for (key, val) in class_to_idx.items()}
    
    return idx_to_class
    
def _get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--inat_root", type=str, default="/local/scratch/cv_datasets/inat21/raw/val")
    parser.add_argument("--out_root", type=str, default="/local/scratch/carlyn.1/clip_paper_bio/features")
    parser.add_argument("--exp_type", type=str, default="8_25_2023_83_epochs", choices=["openai_pretrain", "8_25_2023_83_epochs"])
    parser.add_argument("--reduction", type=str, default="tsne", choices=["tsne", "pca"])
    parser.add_argument("--rerun", action="store_true", default=False)
    parser.add_argument("--remove_outliers", action="store_true", default=False)
    parser.add_argument("--subset", type=int, default=-1)
    parser.add_argument("--top_k", type=int, default=6)

    return parser.parse_args()

if __name__ == "__main__":
    import numpy as np
    args = _get_args()

    hierarchy_label_map = _get_hierarchy_lbl_map(args.inat_root)

    features = np.load(os.path.join(args.out_root, args.exp_type + "_features.npy"))
    labels = np.load(os.path.join(args.out_root, args.exp_type + "_labels.npy"))

    hierarchy_labels = []
    for lbl in labels:
        lvls = hierarchy_label_map[lbl].split("_")[1:]
        hierarchy_labels.append(["_".join(lvls[:lvl+1]) for lvl in range((len(lvls)))])

    hierarchy_labels = np.array(hierarchy_labels)

    folder_name = f"{args.exp_type}_{args.reduction}_{'rerun' if args.rerun else 'no_rerun'}"
    folder_name += f"_top_{args.top_k}_{'remove_outliers' if args.remove_outliers else 'outliers_remain'}"
    output_name = os.path.join("tmp", folder_name)

    if args.subset > 0:
        features = features[:args.subset]
        hierarchy_labels = hierarchy_labels[:args.subset]
    
    create_hierarchical_tree_vis(features, hierarchy_labels, 
        reduction_method=args.reduction, output=output_name,
        rerun_reduction=args.rerun, verbose=True, top_k=args.top_k,
        remove_outliers=args.remove_outliers)