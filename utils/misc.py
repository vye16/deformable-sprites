import os
import glob
import imageio
import json
import matplotlib.pyplot as plt
import numpy as np
import torch

from argparse import Namespace

from .flow_viz import flow_to_image


def move_to(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device)
    if isinstance(item, dict):
        return dict([(k, move_to(v, device)) for k, v in item.items()])
    if isinstance(item, (tuple, list)):
        return [move_to(x, device) for x in item]
    print(type(item))
    raise NotImplementedError


def get_unique_log_path(log_dir, resume):
    past_logs = sorted(glob.glob("{}/*".format(log_dir)))
    cur_version = len(past_logs)
    if resume:  # assume resuming the most recent run
        cur_version = max(0, cur_version - 1)
    return "{}/v{:03d}".format(log_dir, cur_version)


def load_checkpoint(path, **kwargs):
    if not os.path.isfile(path):
        print("{} DOES NOT EXIST!".format(path))
        return 0
    print("RESUMING FROM", path)
    ckpt = torch.load(path)
    start_iter = ckpt["i"]
    for name, module in kwargs.items():
        if name not in ckpt:
            print("{} not saved in checkpoint, skipping".format(name))
            continue
        module.load_state_dict(ckpt[name])
    return start_iter


def save_checkpoint(path, i, **kwargs):
    print("ITER {:6d} SAVING CHECKPOINT TO {}".format(i, path))
    save_dict = {name: module.state_dict() for name, module in kwargs.items()}
    save_dict["i"] = i
    torch.save(save_dict, path)


def save_args(args, path):
    with open(path, "w") as f:
        json.dump(vars(args), f, indent=1)


def load_args(path):
    with open(path, "r") as f:
        arg_dict = json.load(f)
    return Namespace(**arg_dict)


def cat_tensor_dicts(dict_list, dim=0):
    if len(dict_list) < 1:
        return {}
    keys = dict_list[0].keys()
    return {k: torch.cat([d[k] for d in dict_list], dim=dim) for k in keys}


def save_vis_dict(out_dir, vis_dict, save_keys=[], skip_keys=[], overwrite=False):
    """
    :param out_dir
    :param vis_dict dict of 4+D tensors
    :param skip_keys (optional) list of keys to skip
    :return the paths each tensor is saved to
    """
    if os.path.isdir(out_dir) and not overwrite:
        print("{} exists already, skipping".format(out_dir))
        return

    if len(vis_dict) < 1:
        return []

    os.makedirs(out_dir, exist_ok=True)
    vis_dict = {k: v.detach().cpu() for k, v in vis_dict.items()}

    if len(save_keys) < 1:
        save_keys = vis_dict.keys()
    save_keys = set(save_keys) - set(skip_keys)

    out_paths = {}
    for name, vis_batch in vis_dict.items():
        if name not in save_keys:
            continue
        if vis_batch is None:
            continue
        out_paths[name] = save_vis_batch(out_dir, name, vis_batch)
    return out_paths


def save_vis_batch(out_dir, name, vis_batch, rescale=False, save_dir=False):
    """
    :param out_dir
    :param name
    :param vis_batch (B, *, C, H, W) first dimension is time dimension
    """
    if len(vis_batch.shape) < 4:
        return None

    C = vis_batch.shape[-3]
    if C > 3:
        return

    if C == 2:  # is a flow map
        vis_batch = flow_to_image(vis_batch)

    if rescale:
        vmax = vis_batch.amax(dim=(-1, -2), keepdim=True)
        vmax = torch.clamp_min(vmax, 1)
        vis_batch = vis_batch / vmax

    return save_batch_imgs(os.path.join(out_dir, name), vis_batch, save_dir)


def save_batch_imgs(name, vis_batch, save_dir):
    """
    Saves a 4+D tensor of (B, *, 3, H, W) in separate image dirs of B files.
    :param out_dir_pre prefix of output image directories
    :param vis_batch (B, *, 3, H, W)
    """
    vis_batch = vis_batch.detach().cpu()
    B, *dims, C, H, W = vis_batch.shape
    vis_batch = vis_batch.view(B, -1, C, H, W)
    vis_batch = (255 * vis_batch.permute(0, 1, 3, 4, 2)).byte()
    M = vis_batch.shape[1]

    paths = []
    for m in range(M):
        if B == 1:  # save single image
            path = f"{name}_{m}.png"
            imageio.imwrite(path, vis_batch[0, m])
        elif save_dir:  # save directory of images
            path = f"{name}_{m}"
            save_img_dir(path, vis_batch[:, m])
        else:  # save gif
            path = f"{name}_{m}.gif"
            imageio.mimwrite(path, vis_batch[:, m])

        paths.append(path)
    return paths


def save_img_dir(out, vis_batch):
    os.makedirs(out, exist_ok=True)
    for i in range(len(vis_batch)):
        path = f"{out}/{j:05d}.png"
        imageio.imwrite(path, vis_batch[i])


def save_vid(path, vis_batch):
    """
    :param vis_batch (B, 3, H, W)
    """
    vis_batch = vis_batch.detach().cpu()
    save = (255 * vis_batch.permute(0, 2, 3, 1)).byte()
    imageio.mimwrite(path, save)
