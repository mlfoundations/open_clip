# Training Data on OSC

We want to create a webdataset made up of the 10M images from iNat21, EOL and BIOSCAN.

Here are the steps:

1. Get a list of all images.
2. Choose a validation split.
3. Write the splits in the webdataset format.

## 1. Get a list of all images

Edit the `--tag` from v2 to v3.

```sh
sbatch slurm/make-dataset-mapping.sh
```

## 2. Choose a validation split

You have to pass a different `mapping.sqlite` file if you used a different tag in the previous step.

```sh
python scripts/evobio10m/make_splits.py \
  --db /fs/ess/PAS2136/open_clip/data/evobio10m-v2/mapping.sqlite \
  --seed 17 \
  --val-split 5
```

## 3. Write the splits in webdataset formats

Edit the `--tag` from v2 to v3 if you did in step 1.

```sh
sbatch slurm/make-dataset-wds.sh
```

## 4. Verification

You must point the `--shardlist` argument to the train and the val splits.

```sh
python scripts/evobio10m/check_wds.py --shardlist '/fs/ess/PAS2136/open_clip/data/evobio10m-v2.1/224x224/val/shard-{000003..000031}.tar' --workers 8
```

This will print out some bad files, like:

```
/fs/ess/PAS2136/open_clip/data/evobio10m-v2.1/224x224/val/shard-000003.tar
/fs/ess/PAS2136/open_clip/data/evobio10m-v2.1/224x224/val/shard-000004.tar
/fs/ess/PAS2136/open_clip/data/evobio10m-v2.1/224x224/val/shard-000013.tar
```

Then you should delete these files and re-run the jobs.

## Configuration

There are various configuration options hidden all over the codebase.
Here are some different ones to look out for:

* `src/imageomics/eol.py`: `VernacularNameLookup` has a default argument for `data/eol/vernacularnames.csv`. If your `vernacularnames.csv` is not in `data/eol`, you will have to edit this.
* `src/imageomics/eol.py`: `EolNameLookup` has many default arguments for different files.
* `src/iamgeomics/evobio10m.py`: `eol_root_dir`, `inat21_root_dir` and `bioscan_root_dir` all point to specific image folders on OSC.
* `src/iamgeomics/evobio10m.py`: `get_output_dir` returns the default location for evobio10m datasets.
* `scripts/evobio10m/make_wds.py`: `seen_in_training_json` and `unseen_in_training_json` are the seen and unseen species used in the rare species benchmarks. There are version controlled so they should be in the correct location in your repo.

