import argparse
import os
import yaml
from tqdm.auto import tqdm
import torch as th
from torch.utils.data import DataLoader
import pyiqa
from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import PairedSRDataset
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
)


def load_superres_data(
    hr_dir,
    lr_dir,
    batch_size,
    large_size,
    small_size,
    deterministic=False,
):
    dataset = PairedSRDataset(hr_dir, lr_dir, large_size, small_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )

    device = dist_util.dev()
    while True:
        for hr_batch, model_kwargs in loader:
            hr_batch = hr_batch.to(device, non_blocking=True)
            for k, v in model_kwargs.items():
                if isinstance(v, th.Tensor):
                    model_kwargs[k] = v.to(device, non_blocking=True)
            yield hr_batch, model_kwargs


def compute_batch_psnr(pred, target):
    pred = pred.clamp(-1, 1)
    target = target.clamp(-1, 1)
    pred_01 = (pred + 1) / 2
    target_01 = (target + 1) / 2
    mse = ((pred_01 - target_01) ** 2).flatten(1).mean(1)
    psnr = 10 * th.log10(1.0 / (mse + 1e-8))
    return psnr.mean().item()


@th.no_grad()
def evaluate_metrics(
    model,
    diffusion,
    val_hr_dir,
    val_lr_dir,
    large_size,
    small_size,
    batch_size,
    max_val_batches,
    device,
    iqa_metrics,
):
    model.eval()

    dataset = PairedSRDataset(val_hr_dir, val_lr_dir, large_size, small_size)
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )

    total_psnr = 0.0
    total_images = 0
    metric_logs = {k: [] for k in ["ssim", "lpips", "niqe", "musiq", "clipiqa", "pi"]}

    for batch_idx, (hr_batch, model_kwargs) in enumerate(val_loader):
        if batch_idx >= max_val_batches:
            break

        hr_batch = hr_batch.to(device, non_blocking=True)
        for k, v in model_kwargs.items():
            if isinstance(v, th.Tensor):
                model_kwargs[k] = v.to(device, non_blocking=True)

        sample = diffusion.p_sample_loop(
            model,
            hr_batch.shape,
            clip_denoised=True,
            model_kwargs=model_kwargs,
            progress=False,
        )

        batch_psnr = compute_batch_psnr(sample, hr_batch)
        total_psnr += batch_psnr * hr_batch.shape[0]
        total_images += hr_batch.shape[0]

        pred_01 = (sample.clamp(-1, 1) + 1) / 2.0
        gt_01 = (hr_batch.clamp(-1, 1) + 1) / 2.0

        for name, metric_fn in iqa_metrics.items():
            if name in ["ssim", "lpips"]:
                val = metric_fn(pred_01, gt_01).mean().item()
            else:
                val = metric_fn(pred_01).mean().item()
            metric_logs[name].append(val)

    if total_images == 0:
        model.train()
        return {k: 0.0 for k in ["psnr"] + list(metric_logs.keys())}

    results = {"psnr": total_psnr / total_images}
    for k, v in metric_logs.items():
        results[k] = sum(v) / len(v) if len(v) > 0 else 0.0

    model.train()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train_hr_dir = cfg.get("data_dir", "./data/DIV2K/HR/train")
    train_lr_dir = cfg.get("lr_data_dir", "./data/DIV2K/LR/train")
    val_hr_dir = cfg.get("val_data_dir", "./data/DIV2K/HR/valid")
    val_lr_dir = cfg.get("val_lr_data_dir", "./data/DIV2K/LR/valid")

    batch_size = int(cfg.get("batch_size", 4))
    lr = float(cfg.get("lr", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    ema_rate = float(cfg.get("ema_rate", 0.9999))
    schedule_sampler_name = cfg.get("schedule_sampler", "uniform")

    total_steps = int(cfg.get("total_steps", 20000))
    log_interval = int(cfg.get("log_interval", 10))
    save_interval = int(cfg.get("save_interval", 1000))
    eval_interval = int(cfg.get("eval_interval", 1000))
    eval_num_batches = int(cfg.get("eval_num_batches", 5))
    out_dir = cfg.get("out_dir", "./sr_checkpoints")

    warmup_steps = int(cfg.get("warmup_steps", 500))
    lr_decay_steps = int(cfg.get("lr_decay_steps", 5000))
    lr_gamma = float(cfg.get("lr_gamma", 0.5))

    os.makedirs(out_dir, exist_ok=True)
    dist_util.setup_dist()
    logger.configure(dir=out_dir)

    logger.log(f"Loading config from {args.config}")
    logger.log("Experiment Mode: Original Baseline (Paired HR/LR)")
    logger.log(
        f"LR Schedule: Base {lr:.1e}, Warmup {warmup_steps}, Decay every {lr_decay_steps} by {lr_gamma}"
    )

    model_diffusion_defaults = sr_model_and_diffusion_defaults()
    model_diffusion_args = {}
    for k, v_default in model_diffusion_defaults.items():
        if k in cfg:
            model_diffusion_args[k] = cfg[k]
        else:
            model_diffusion_args[k] = v_default

    model, diffusion = sr_create_model_and_diffusion(**model_diffusion_args)
    device = dist_util.dev()
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Model parameters: {num_params / 1e6:.2f}M")

    ema_params = {k: v.detach().clone() for k, v in model.state_dict().items()}

    schedule_sampler = create_named_schedule_sampler(schedule_sampler_name, diffusion)
    large_size = model_diffusion_args["large_size"]
    small_size = model_diffusion_args["small_size"]

    logger.log("Initializing IQA metrics...")
    iqa_metrics = {
        "ssim": pyiqa.create_metric("ssim", device=device),
        "lpips": pyiqa.create_metric("lpips", device=device),
        "niqe": pyiqa.create_metric("niqe", device=device),
        "musiq": pyiqa.create_metric("musiq", device=device),
        "clipiqa": pyiqa.create_metric("clipiqa", device=device),
        "pi": pyiqa.create_metric("pi", device=device),
    }

    data = load_superres_data(
        train_hr_dir,
        train_lr_dir,
        batch_size=batch_size,
        large_size=large_size,
        small_size=small_size,
        deterministic=False,
    )

    opt = th.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    base_lr = lr

    logger.log("Starting training...")
    pbar = tqdm(range(total_steps), dynamic_ncols=True)

    for step in pbar:
        model.train()
        opt.zero_grad()

        if warmup_steps > 0 and step < warmup_steps:
            warmup_factor = float(step + 1) / float(warmup_steps)
        else:
            warmup_factor = 1.0

        decay_factor = lr_gamma ** (step // lr_decay_steps)
        cur_lr = base_lr * warmup_factor * decay_factor

        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        large_batch, model_kwargs = next(data)

        t, weights = schedule_sampler.sample(large_batch.shape[0], device)
        losses = diffusion.training_losses(
            model, large_batch, t, model_kwargs=model_kwargs
        )
        loss = (losses["loss"] * weights).mean()

        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with th.no_grad():
            for k, v in model.state_dict().items():
                ema_params[k].mul_(ema_rate).add_(v, alpha=1.0 - ema_rate)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{cur_lr:.1e}"})

        if (step + 1) % log_interval == 0:
            logger.log(
                f"step {step+1}, loss {loss.item():.6f}, lr {cur_lr:.6e}"
            )

        if (step + 1) % eval_interval == 0:
            logger.log(f"Starting evaluation at step {step+1}...")
            ema_model, _ = sr_create_model_and_diffusion(**model_diffusion_args)
            ema_model.load_state_dict(ema_params)
            ema_model.to(device)

            metrics = evaluate_metrics(
                ema_model,
                diffusion,
                val_hr_dir=val_hr_dir,
                val_lr_dir=val_lr_dir,
                large_size=large_size,
                small_size=small_size,
                batch_size=batch_size,
                max_val_batches=eval_num_batches,
                device=device,
                iqa_metrics=iqa_metrics,
            )

            log_msg = (
                f"Evaluation Step {step+1} | "
                f"PSNR: {metrics['psnr']:.2f} | "
                f"SSIM: {metrics['ssim']:.4f} | "
                f"LPIPS: {metrics['lpips']:.4f} | "
                f"NIQE: {metrics['niqe']:.4f} | "
                f"PI: {metrics['pi']:.4f} | "
                f"CLIPIQA: {metrics['clipiqa']:.4f} | "
                f"MUSIQ: {metrics['musiq']:.4f}"
            )
            logger.log(log_msg)

        if (step + 1) % save_interval == 0:
            ckpt_path = os.path.join(out_dir, f"model_step_{step+1}.pt")
            ema_ckpt_path = os.path.join(out_dir, f"ema_model_step_{step+1}.pt")
            th.save(model.state_dict(), ckpt_path)
            th.save(ema_params, ema_ckpt_path)
            logger.log(f"Saved model to {ckpt_path}")


if __name__ == "__main__":
    main()
