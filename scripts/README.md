# Imageomics-Specific Scripts

`cub2011_to_splits.py` converts 12K images from the raw CUB-200 format into train/test splits and resizes the images. [cub2011-eval.md](https://github.com/Imageomics/open_clip/blob/main/docs/imageomics/cub2011-eval.md) has more documentation on exactly how to use it, but it is <100 lines. It is very straightforward.

`inat21_to_wds.py` converts raw iNat21 images to a webdataset format and holds 1K classes out for an unseen evaluation. To use it, you will have to edit some variables in the script (`inat_root`, `output_root`). **Note: this script is outdated because we use `make_evobio10m.py`, which includes BIOSCAN-1M, iNat21 and EOL images.**

`inat_common_names.py` gets a mapping of scientific names to common names for all species in the iNat21 dataset. In the future, this will be expanded to all species on iNaturalist, not just those in the iNat21 splits. You don't need to run this script because the mapping is already committed to version control.

`jiggins_zenodo.py` reads some csv and json data from `data/butterflies` and creates a json file describing the class to image relationship for the butterflies dataset. I (Sam) forgot exactly how it works, but I am going to refresh myself and update this page later.
