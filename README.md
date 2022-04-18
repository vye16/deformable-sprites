# Deformable Sprites Codebase

Vickie Ye, Zhengqi Li, Richard Tucker, Angjoo Kanazawa, Noah Snavely

<img src=".teaser.gif" height="320px">

Website: [link](https://deformable-sprites.github.io)
arXiv: [link](https://arxiv.org/abs/2204.07151)

## Installation
---
This code has been tested with PyTorch 1.10 and Python 3.9. Please make sure you have updated NVIDIA drivers supporting at least CUDA 10.2.

To start, create a conda environment with
```sh
conda env create -f environment.yml
conda activate deformable-sprites
```

Alternatively, use `pip install -r requirements.txt`.

## Organizing and Extracting Frames for Custom Data

We've provided scripts in the `scripts` subdirectory for organizing and preprocessing data for custom videos.
We recommend using the `dataset_extract.py` script to keep track of the necessary outputs (RGB frames, optical flow).

This will allow us to create a custom video dataset with the structure
- /path/to/custom\_videos
    - videos
    - PNGImages
    - raw\_flows\_gap1
    - ...

Start by creating a `custom_specs.json` file for all custom videos you want to process:

        {
            "video_name_wo_ext": {
                "start": 0,
                "end": 10,
                "fps": 10
            }
        }

Then run 
```sh
python dataset_extract.py --root <path/to/custom_videos> --specs <path/to/specs.json>
```
to extract desired frames.
Once the frames are extracted, we can process them for optical flow in the same way as the other datasets (DAVIS, FBMS, etc.)

## Computing Optical Flow on Datasets

We tested our model on the DAVIS, FBMS, SegTrackV2, and custom datasets, and handle those data organization schemes.
We used [RAFT](https://github.com/princeton-vl/RAFT) to compute optical flow, so please clone the repo separately and update the paths in `scripts/run_raft.py`.

Once all paths are updated, run
```sh
python dataset_raft.py --root <path/to/data> --dtype <dataset_name> --gap <gap_between_frames> --gpus <list of gpus>
```

## Running optimization
We use [hydra](https://hydra.cc/docs/intro) to manage configurations. To run optimization, first update `confs/data/<dataset>.yaml` with the paths to your data.
Then run
```sh
python run_opt.py data=<dataset> data.seq=<sequence_name>
```

## BibTeX

If you use our code in your research, please cite the following paper:

```
@inproceedings{ye2022sprites,
    title = {Deformable Sprites for Unsupervised Video Decomposition},
    author = {Ye, Vickie and Li, Zhengqi and Tucker, Richard and Kanazawa, Angjoo and Snavely, Noah},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2022}
}
```
