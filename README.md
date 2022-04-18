# Deformable Sprites Codebase

Vickie Ye, Zhengqi Li, Richard Tucker, Angjoo Kanazawa, Noah Snavely

![Teaser](https://raw.github.com/vye16/deformable-sprites/master/teaser_edited.gif)

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
python dataset_extract.py --root {path/to/custom_videos} --specs {path/to/specs.json}
```
to extract desired frames.
Once the frames are extracted, we can process them for optical flow in the same way as the other datasets (DAVIS, FBMS, etc.)

## Computing Optical Flow on Datasets

We tested our model on the DAVIS, FBMS, SegTrackV2, and custom datasets, and handle those data organization schemes.
We used [RAFT](https://github.com/princeton-vl/RAFT) to compute optical flow, so please clone the repo separately and update the paths in `scripts/run_raft.py`.

Once all paths are updated, run
```sh
python dataset_raft.py --root {path/to/data} --dtype {dataset_name} --gap {gap_between_frames} --gpus {list of gpus}
```

## Running optimization
We use [hydra](https://hydra.cc/docs/intro) to manage configurations. To run optimization, first update `confs/data/{dataset}.yaml` with the paths to your data.
Then run
```sh
python run_opt.py data={dataset} data.seq={sequence_name}
```
This will launch the optimization using the configurating in `confs/config.yaml`.
These are the parameters used for the datasets in the paper and project website.
However, you might find that for a custom video, you might want to adjust the parameters for better results.

### Optimization Time
The number of epochs/iterations of optimization is currently set on the lower end.
If you would like to let the optimization run longer, please increase the number of iterations in `iters_per_phase.deform` in `confs/train.yaml`.

### Model Parameters
The BSpline parameters are in `confs/model/transform/bspline.yaml`.
The number of knots parameterizing the 2D warp fields will control how flexible the 2D warp fields.
To increase the number of 2D knots, decrease `xy_step`. To increase how much an individual knot can deform, increase `max_step`.

The texture generation parameters are in `confs/model/base_tex.yaml`.
These parameters are UNet parameters and affect the quality of the output textures.
Increasing the number of levels and channels of the texture generator will improve the output quality but increase optimization time.

The mask prediction parameters are in `confs/model/no_tex.yaml`.
These paramaters are also UNet parameters.
We don't recommend increasing these too much because they may cause overfitting to each frame (rather than learning appearance cues for the entire video).
However, if you find that groups are not separating out well, play with the `init_std` parameter.
This is the standard deviation that the last layer of the weights are sampled from, and determine the variance of the foreground predicted probabilities.
Increasing this parameter separate out multiple foreground objects.

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
