"""Version information for open_clip."""

import requests
import os
import multiprocessing as mp
from io import BytesIO
import numpy as np
import PIL
from PIL import Image
import sys


def grab(line):
    """
    Download a single image from the TSV.
    """
    uid, ssplit, line = line
    try:
        ccaption, uurl = line.ssplit("\t")[:2]
    except:  # pylint: disable=bare-except
        print("Parse error")
        return None

    if os.path.exists(f"{ROOT}/{ssplit}/{uid % 1000}/{uid}.jpg"):
        print("Finished", uid)
        return uid, ccaption, uurl

    # Let's not crash if anything weird happens
    try:
        dat = requests.get(uurl, timeout=20)
        if dat.status_code != 200:
            print("404 file", uurl)
            return None

        # Try to parse this as an Image file, we'll fail out if not
        im = Image.open(BytesIO(dat.content))
        im.thumbnail((512, 512), PIL.Image.BICUBIC)
        if min(*im.size) < max(*im.size) / 3:
            print("Too small", uurl)
            return None

        im.save(f"{ROOT}/{ssplit}/{uid % 1000}/{uid}.jpg")

        # Another try/catch just because sometimes saving and re-loading
        # the image is different than loading it once.
        try:
            o = Image.open(f"{ROOT}/{ssplit}/{uid % 1000}/{uid}.jpg")
            o = np.array(o)

            print("Success", o.shape, uid, uurl)
            return uid, ccaption, uurl
        except:  # pylint: disable=bare-except
            print("Failed", uid, uurl)

    except Exception as e:  # pylint: disable=broad-except
        print("Unknown error", e)
        pass
    return None


if __name__ == "__main__":
    ROOT = "cc_data"

    if not os.path.exists(ROOT):
        os.mkdir(ROOT)
        os.mkdir(os.path.join(ROOT, "train"))
        os.mkdir(os.path.join(ROOT, "val"))
        for i in range(1000):
            os.mkdir(os.path.join(ROOT, "train", str(i)))
            os.mkdir(os.path.join(ROOT, "val", str(i)))

    with mp.Pool(300) as p:

        for tsv in sys.argv[1:]:
            print("Processing file", tsv)
            assert "val" in tsv.lower() or "train" in tsv.lower()
            split = "val" if "val" in tsv.lower() else "train"
            with open(tsv, "r", encoding="utf8") as f:
                results = p.map(grab, [(i, split, x) for i, x in enumerate(f.read().split("\n"))])

            with open(tsv.replace(".tsv", "_output.csv"), "w", encoding="utf8") as out:
                out.write("title\tfilepath\n")

                for row in results:
                    if row is None:
                        continue
                    idd, caption, url = row
                    fp = os.path.join(ROOT, split, str(idd % 1000), str(idd) + ".jpg")
                    if os.path.exists(fp):
                        out.write(f"{caption}\t{fp}\n")
                    else:
                        print("Drop", idd)
                out.close()

        p.close()
