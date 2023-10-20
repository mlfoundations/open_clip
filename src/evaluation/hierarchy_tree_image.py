import os

from timeit import default_timer as timer
from datetime import timedelta

from torchvision.datasets.folder import find_classes
from sklearn.manifold import TSNE
#from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA
from scipy.stats import chi2


import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from evaluation.extract_features import get_taxon_names

TAXONOMICAL_RANKS = [
    "Kingdoms", "Phyla", "Classes", "Orders", "Families", "Genera", "Species"
]

COLORS = [(127, 60, 141), (17, 165, 121), (57, 105, 172), (242, 183, 1), (231, 63, 116), (128, 186, 90), (230, 131, 16), (0, 134, 149), (207, 28, 144), (249, 123, 114), (165, 170, 153)]

def get_colors():
    colors = []
    for c in COLORS:
        colors.append((c[0]/255, c[1]/255, c[2]/255))
    return colors

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
    top_k=6, output="", rerun_reduction=False, verbose=False, remove_outliers=False, precomputed_reductions=None,
    dpi=100):
    assert reduction_method in ["tsne", "pca"], f"{reduction_method} not supported."
    os.makedirs(output, exist_ok=True)

    def log(x):
        if verbose:
            print(x)

    lvls = len(hierarchy_labels[0])
    #base_colors = [ x for x in mcolors.BASE_COLORS.values() ]
    base_colors = get_colors()

    def run_reduction(in_feats):
        log(f"Running {reduction_method} on features: {in_feats.shape}. This may take some time.")
        start = timer()
        out_feats = reduce.fit_transform(in_feats)
        end = timer()
        log(f"{reduction_method} completed: {timedelta(seconds=end-start)}")

        return out_feats
    
    if reduction_method == "tsne":
        reduce = TSNE(n_jobs=16, n_components=2, verbose=2)
    elif reduction_method == "pca":
        reduce = PCA(2)

    if not rerun_reduction:
        if precomputed_reductions is None:
            features = run_reduction(features)
        else:
            features = precomputed_reductions
        saved_features = features
        saved_labels = hierarchy_labels
    else:
        saved_features = []
        saved_labels = []

    data_queue = [(features, hierarchy_labels, 0)]
    largest_name = None
    graph_data = []

    nrows=3
    ncols=3
    height = 7
    width = 5
    final_fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(height*nrows, width*ncols), dpi=dpi)
    #plt.subplots_adjust(wspace=0, hspace=0.25)
    while len(data_queue) > 0:
        past_largest_name = largest_name
        cur_plot_lvl = past_largest_name if past_largest_name is not None else "Root"

        feats, lbls, lvl = data_queue.pop(0)
        if rerun_reduction:
            if precomputed_reductions is None:
                reduced_feats = run_reduction(feats)
            else:
                reduced_feats = precomputed_reductions[lvl]
            saved_features.append(reduced_feats)
            saved_labels.append(lbls)
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

        row = lvl//ncols
        col = lvl%ncols
        if lvl == 6:
            col +=1
        ax = axs[row, col]
        plt.setp(ax.spines.values(), lw=3, color='black') # Set border width
        title_parts = cur_plot_lvl.split('_')
        #fig_title = '-'.join(title_parts[-2 if len(title_parts) > 1 else 0:])
        fig_title = TAXONOMICAL_RANKS[lvl]
        if lvl > 0:
            fig_title += f" of {title_parts[-1]}"
        #fig.suptitle(f"({lvl+1}) {fig_title}", fontsize=50, y=0.98)
        ax.set_title(f"({lvl+1}) {fig_title}", fontsize=25, y=1.02)

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
               
            markersize = 2
            if lvl == 5:
                markersize *= 2
            elif lvl == 6:
                markersize *= 4
            ax.scatter(plot_feat[:, 0], plot_feat[:, 1], label=name, 
                color=base_colors[c], alpha=0.50, s=markersize, rasterized=True) # have to rasterize for .pdf to load quicker
            

            c += 1

        # Add to queue
        if (lvl+1) < lvls:
            data_queue.append((most_feats, most_lbls, lvl+1))

        if lvl > 0:
            save_path = os.path.join(output, f"depth_{lvl}_{past_largest_name}.png")
        else:
            save_path = os.path.join(output, f"depth_{lvl}.png")

        markerscale = 8
        if lvl == 5:
            markerscale = 6
        elif lvl == 6:
            markerscale = 4
        ax.legend(loc='upper right', ncols=2, markerscale=markerscale, fontsize=12)
        # Turn off axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

        #ax.tight_layout()
        ax_extent = ax.get_window_extent().transformed(final_fig.dpi_scale_trans.inverted())
        final_fig.savefig(save_path, bbox_inches=ax_extent)
        graph_data.append((save_path, lvl))
    
    for r in range(nrows):
        for c in range(ncols):
            axs[r][c].set_xticks([])
            axs[r][c].set_yticks([])
    plt.setp(axs[-1][-1].spines.values(), lw=0)
    plt.setp(axs[-1][-3].spines.values(), lw=0)

    final_fig.tight_layout(h_pad=1, w_pad=1)
    final_fig.savefig(os.path.join(output, "full_image.png"))
    final_fig.savefig(os.path.join(output, "full_image.pdf"))

    """
    two_by_three = True
    imgs = [np.array(Image.open(data[0])) for data in graph_data]
    if two_by_three:
        rows = [np.concatenate((imgs[i], imgs[i+1], imgs[i+2]), axis=1) for i in range(0, len(imgs)-1, 3)]
        final_image = np.concatenate(rows, axis=0)
    else:
        final_image = np.concatenate(imgs, axis=1)
    """
    # Assume 6 images
    #for path, lbl in graph_data[1:]:
    #    img = np.array(Image.open(path))
    #    full_img = np.concatenate((full_img, img), axis=0)
    #Image.fromarray(final_image).save(os.path.join(output, "full_image.png"))
    #Image.fromarray(final_image).save(os.path.join(output, "full_image.pdf"))

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

    return saved_features, saved_labels

def _get_hierarchy_lbl_map(root):
    classes, class_to_idx = find_classes(root)
    idx_to_class = {val: key for (key, val) in class_to_idx.items()}
    
    return idx_to_class
    
def _get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--inat_root", type=str, default="/local/scratch/cv_datasets/inat21/raw/val")
    parser.add_argument("--out_root", type=str, default="/local/scratch/carlyn.1/clip_paper_bio/features")
    parser.add_argument("--exp_type", type=str, default="8_25_2023_83_epochs")
    parser.add_argument("--taxon_file", type=str, default="/local/scratch/carlyn.1/clip_paper_bio/taxons/taxon-merged.json")
    parser.add_argument("--reduction", type=str, default="tsne", choices=["tsne", "pca"])
    parser.add_argument("--rerun", action="store_true", default=False)
    parser.add_argument("--remove_outliers", action="store_true", default=False)
    parser.add_argument("--use_logits", action="store_true", default=False)
    parser.add_argument("--use_text", action="store_true", default=False)
    parser.add_argument("--subset", type=int, default=-1)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--dpi", type=float, default=500)

    return parser.parse_args()

if __name__ == "__main__":
    import numpy as np
    args = _get_args()

    hierarchy_label_map = _get_hierarchy_lbl_map(args.inat_root)

    logit_suffix = "_logits" if args.use_logits else ""
    if args.use_text:
        logit_suffix = "_text" if args.use_logits else ""

    if args.use_text:
        taxon_names = get_taxon_names(args.taxon_file)
        features_end = np.load(os.path.join(args.out_root, args.exp_type + "_features" + logit_suffix + ".npy"))
        print(features_end.shape)
        part_files = []
        for root, dirs, files in os.walk(os.path.join(args.out_root, args.exp_type + "_text_tmps")):
            for f in files:
                if f.split(".")[-1] != "npy": continue
                part_files.append(os.path.join(root, f))
        part_files = sorted(part_files, key=lambda x: int(x.split(os.path.sep)[-1].split(".")[0]))
        loaded = [np.load(p) for p in part_files]
        features = np.concatenate(loaded, axis=0)
        features = np.concatenate((features, features_end), axis=0)
        labels = taxon_names

        hierarchy_labels = []
        for lbl in labels:
            lvls = lbl.split("_")
            hierarchy_labels.append(["_".join(lvls[:lvl+1]) for lvl in range((len(lvls)))])
    else:
        features = np.load(os.path.join(args.out_root, args.exp_type + "_features" + logit_suffix + ".npy"))
        labels = np.load(os.path.join(args.out_root, args.exp_type + "_labels" + logit_suffix + ".npy"))

        hierarchy_labels = []
        for lbl in labels:
            lvls = hierarchy_label_map[lbl].split("_")[1:]
            hierarchy_labels.append(["_".join(lvls[:lvl+1]) for lvl in range((len(lvls)))])

    hierarchy_labels = np.array(hierarchy_labels)

    folder_name = f"{args.exp_type}_{args.reduction}_{'rerun' if args.rerun else 'no_rerun'}"
    folder_name += f"_top_{args.top_k}_{'remove_outliers' if args.remove_outliers else 'outliers_remain'}"
    if args.use_logits:
        folder_name += "_use_logits"
    elif args.use_text:
        folder_name += "_use_text"
    output_name = os.path.join("tmp", folder_name)

    if args.subset > 0:
        features = features[:args.subset]
        hierarchy_labels = hierarchy_labels[:args.subset]

    precomputed_features = None
    precompute_name = "precomputed_0.npy" if args.rerun else "precomputed.npy"
    precomputed_lbl_name = "precomputed_0_labels.npy" if args.rerun else "precomputed_labels.npy"
    precomputed_path = os.path.join(output_name, precompute_name)
    precomputed_label_path = os.path.join(output_name, precomputed_lbl_name)
    if os.path.exists(precomputed_path):
        if not args.rerun:
            precomputed_features = np.load(precomputed_path)
        else:
            precomputed_features = []
            for i in range(7):
                precomputed_features.append(np.load(os.path.join(output_name, f"precomputed_{i}.npy")))
    
    sf, sl = create_hierarchical_tree_vis(features, hierarchy_labels, 
        reduction_method=args.reduction, output=output_name,
        rerun_reduction=args.rerun, verbose=True, top_k=args.top_k,
        remove_outliers=args.remove_outliers, precomputed_reductions=precomputed_features,
        dpi=args.dpi)

    if not args.rerun:
        np.save(precomputed_path, sf)
        np.save(precomputed_label_path, sl)
    else:
        for i, f in enumerate(sf):
            np.save(os.path.join(output_name, f"precomputed_{i}.npy"), f)
        for i, l in enumerate(sl):
            np.save(os.path.join(output_name, f"precomputed_{i}_labels.npy"), l)

    print("Reduction features saved")

    

    