from argparse import ArgumentParser
import os

from PIL import Image

import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="tmp")
    args = parser.parse_args()

    base = "tsne_rerun_top_6_remove_outliers"

    all_dirs = [os.path.join(args.root, "openai_pretrain_" + base)]
    for model in ["1m", "10m", "inat"]:
        for text in ["sci", "sci_com", "taxon", "taxon_com", "com", "random"]:
            dir_path = os.path.join(args.root, model + "_" + text + "_" + base)
            if os.path.exists(dir_path):
                all_dirs.append(dir_path)

    image_paths = []
    for dir_path in all_dirs:
        image_paths.append({})
        for root, dirs, files in os.walk(dir_path):
            for f in files:
                basename, ext = f.split(".")
                if ext != "png": continue
                parts = basename.split("_")
                if parts[0] != "depth": continue
                depth = int(parts[1])
                image_paths[-1][depth] = os.path.join(root, f)
    
    final_image = None
    for path_list in image_paths:
        row = [np.array(Image.open(path_list[i])) for i in range(7)]
        row = np.concatenate(row, axis=1)
        if final_image is None:
            final_image = row
        else:
            final_image = np.concatenate((final_image, row), axis=0)

    Image.fromarray(final_image).save(os.path.join(args.root, "compare_image.png"))




