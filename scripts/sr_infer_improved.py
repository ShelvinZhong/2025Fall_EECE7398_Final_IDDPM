import argparse
import os
import glob
import yaml

from tqdm.auto import tqdm
from PIL import Image

import torch as th
from torchvision import transforms

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    sr_improved_model_and_diffusion_defaults,
    sr_improved_create_model_and_diffusion,
)


def load_model_and_diffusion(cfg_path, model_path, device):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_diffusion_defaults = sr_improved_model_and_diffusion_defaults()
    model_diffusion_args = {}
    for k, v_default in model_diffusion_defaults.items():
        if k in cfg:
            model_diffusion_args[k] = cfg[k]
        else:
            model_diffusion_args[k] = v_default

    model, diffusion = sr_improved_create_model_and_diffusion(**model_diffusion_args)
    state_dict = th.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, diffusion, model_diffusion_args


def build_preprocess(small_size):
    return transforms.Compose(
        [
            transforms.Resize((small_size, small_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def sr_infer_on_folder(
    model,
    diffusion,
    model_args,
    lr_dir,
    out_dir,
    num_images=0,
    device=th.device("cpu"),
):
    os.makedirs(out_dir, exist_ok=True)

    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    img_paths = []
    for ext in exts:
        img_paths.extend(glob.glob(os.path.join(lr_dir, ext)))
    img_paths = sorted(img_paths)

    if len(img_paths) == 0:
        raise RuntimeError(f"No image found in {lr_dir}")

    if num_images > 0:
        img_paths = img_paths[:num_images]

    large_size = model_args["large_size"]
    small_size = model_args["small_size"]

    preprocess = build_preprocess(small_size)
    to_pil = transforms.ToPILImage()

    logger.log(f"Found {len(img_paths)} LR images in {lr_dir}")
    logger.log(f"small_size={small_size}, large_size={large_size}")
    logger.log(f"Saving SR images to {out_dir}")

    model.eval()
    th.set_grad_enabled(False)

    for img_path in tqdm(img_paths, desc="SR inference", dynamic_ncols=True):
        lr_img = Image.open(img_path).convert("RGB")
        lr_tensor = preprocess(lr_img).unsqueeze(0).to(device)

        model_kwargs = {"low_res": lr_tensor}

        sr_sample = diffusion.p_sample_loop(
            model,
            (1, 3, large_size, large_size),
            clip_denoised=True,
            model_kwargs=model_kwargs,
            progress=False,
        )

        sr_sample = sr_sample.clamp(-1, 1)
        sr_sample = (sr_sample + 1) / 2.0
        sr_img = to_pil(sr_sample.squeeze(0).cpu())

        base = os.path.basename(img_path)
        name, ext = os.path.splitext(base)
        save_path = os.path.join(out_dir, f"{name}_SR{ext}")
        sr_img.save(save_path)


def main():
    parser = argparse.ArgumentParser("Super-resolution inference script (Improved)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lr_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./sr_results_improved")
    parser.add_argument("--num_images", type=int, default=0)
    args = parser.parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.out_dir)

    device = dist_util.dev()
    logger.log(f"Using device: {device}")

    model, diffusion, model_args = load_model_and_diffusion(
        args.config, args.model_path, device
    )

    sr_infer_on_folder(
        model=model,
        diffusion=diffusion,
        model_args=model_args,
        lr_dir=args.lr_dir,
        out_dir=args.out_dir,
        num_images=args.num_images,
        device=device,
    )

    logger.log("Done! All SR images are saved.")


if __name__ == "__main__":
    main()