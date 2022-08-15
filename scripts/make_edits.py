import os
import glob
import imageio

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import sys

sys.path.append(os.path.abspath("__file__/../.."))
import utils


def propagate_edits(log_dir, img_dir, out_name, edit_idcs, ext="mp4", pad=8):
    coords, edits, masks, imgs = load_components(log_dir, img_dir, edit_idcs)

    out_dir = os.path.join(log_dir, out_name)
    os.makedirs(out_dir, exist_ok=True)

    N, M, h, w, _ = coords.shape
    N, _, H, W = imgs.shape
    for i in range(N):
        mask = TF.resize(masks[i], (H, W), antialias=True)  # (M, 1, H, W)
        apprs = imgs[i, None].repeat(M, 1, 1, 1)  # (M, 3, H, W)

        # warp the edits into each frame
        edit_w = F.grid_sample(edits, coords[i, edit_idcs])  # (M, C, h, w)
        edit_w = TF.resize(edit_w, (H, W), antialias=True)
        # composite the edits with the original frame if applicable
        if edit_w.shape[1] == 4:
            edit_w = edit_w[:, 3:] * edit_w[:, :3] + (1 - edit_w[:, 3:]) * imgs[i, None]
        apprs[edit_idcs] = edit_w
        comp = (mask * apprs).sum(dim=0)  # (3, H, W)
        comp = (255 * comp).byte().permute(1, 2, 0)
        imageio.imwrite(f"{out_dir}/{i:05d}.png", comp)


def load_components(log_dir, img_dir, edit_idcs):
    coords = load_coords_precomputed(log_dir)
    N, M, h, w, _ = coords.shape

    imgs = load_imgs(img_dir, N)  # (N, 3, H, W)
    masks = torch.stack(
        [load_imgs(f"{log_dir}/masks_{m}", N) for m in range(M)], dim=1
    )  # (N, M, 1, h, w)
    edits = load_edits(log_dir, edit_idcs)  # (n_edits, C, th, tw)
    return coords, edits, masks, imgs


def load_coords_precomputed(log_dir):
    """
    loads the masks, textures, coords from src dir
    """
    coord_path = os.path.join(log_dir, "coords.pth")
    if not os.path.isfile(coord_path):
        ## TODO load from saved model checkpoint
        raise NotImplementedError

    coords = torch.load(coord_path).float()
    return coords


def isimage(path):
    ext = os.path.splitext(path)[-1].lower()
    return ext == ".png" or ext == ".jpg" or ext == ".jpeg"


def load_imgs(img_dir, n_expected=-1):
    img_paths = sorted(list(filter(isimage, glob.glob(f"{img_dir}/*"))))
    if n_expected > 0 and len(img_paths) != n_expected:
        print("found {} matching imgs, need {}".format(len(img_paths), n_expected))
        raise ValueError
    imgs = torch.from_numpy(np.stack([imageio.imread(p) for p in img_paths], axis=0))
    imgs = imgs.reshape(*imgs.shape[:3], -1).float()  # (N, H, W, -1)
    imgs = imgs.permute(0, 3, 1, 2)[:, :3] / 255  # (N, 3, H, W)
    return imgs


def load_edits(log_dir, idcs):
    edit_paths = [f"{log_dir}/edit_{i}.png" for i in idcs]
    print(edit_paths)
    assert all(os.path.isfile(p) for p in edit_paths)

    # check size of texture
    test_tex = imageio.imread(f"{log_dir}/texs_0.png")  # (H, W, 3)
    H, W, _ = test_tex.shape

    # pad edits to be the same size
    edits = torch.stack(
        [torch.from_numpy(imageio.imread(path) / 255) for path in edit_paths],
        dim=0,
    ).float()
    edits = pad_diff(edits.permute(0, 3, 1, 2), H, W)  # (N, C, H, W)
    return edits


def pad_diff(src, H, W):
    h, w = src.shape[-2:]
    pl = (W - w) // 2
    pt = (H - h) // 2
    pr = W - w - pl
    pb = H - h - pt
    return TF.pad(src, (pl, pt, pr, pb), fill=1)


def resize_batch(tensor, scale):
    if scale != 1:
        *dims, c, h, w = tensor.shape
        H, W = scale * h, scale * w
        tensor_rs = tensor.view(-1, c, h, w)
        tensor_rs = TF.resize(tensor_rs, (H, W), antialias=True)
        return tensor_rs.view(*dims, c, H, W)
    return tensor


def get_index(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return int(name.split("_")[-1])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", help="log directory with `coords.pth` checkpoint")
    parser.add_argument("img_dir", help="the image input directory")
    parser.add_argument("--out_name", default="edited")
    parser.add_argument("--edit_idcs", default=[0], nargs="*", type=int)
    args = parser.parse_args()

    propagate_edits(args.log_dir, args.img_dir, args.out_name, args.edit_idcs)
