import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

log1_path = Path("/root/Shelvin/improved-diffusion/sr_runs/div2k_x4_original_setting_2/log.txt")
log2_path = Path("/root/Shelvin/improved-diffusion/sr_runs/div2k_x4_improved_setting_2/log.txt")
log3_path = Path("/root/Shelvin/improved-diffusion/sr_runs/div2k_x4_improved_setting_3/log.txt")

pattern = re.compile(
    r"Evaluation Step (\d+) \| PSNR: ([0-9.]+) \| SSIM: ([0-9.]+) \| LPIPS: ([0-9.]+).*?"
    r"NIQE: ([0-9.]+) \| PI: ([0-9.]+) \| CLIPIQA: ([0-9.]+) \| MUSIQ: ([0-9.]+)"
)

def clean_lpips(s: str) -> float:
    if "..." in s:
        s = s.replace("...", "")
    return float(s)

def parse_log(path, model_name):
    rows = []
    if not path.exists():
        print(f"Warning: File not found {path}")
        return pd.DataFrame()
        
    for line in path.read_text(errors="ignore").splitlines():
        m = pattern.search(line)
        if m:
            step, psnr, ssim, lpips, niqe, pi, clipiqa, musiq = m.groups()
            rows.append(
                {
                    "model": model_name,
                    "step": int(step),
                    "psnr": float(psnr),
                    "ssim": float(ssim),
                    "lpips": clean_lpips(lpips),
                    "niqe": float(niqe),
                    "pi": float(pi),
                    "clipiqa": float(clipiqa),
                    "musiq": float(musiq),
                }
            )
    return pd.DataFrame(rows).sort_values("step")

df1 = parse_log(log1_path, "Baseline (Original)")
df2 = parse_log(log2_path, "Ours (SFT Injection)")  
df3 = parse_log(log3_path, "Ours (Simple Add)")     

df = pd.concat([df1, df3, df2], ignore_index=True).sort_values(["model", "step"])

df.to_csv("metrics_ablation_study.csv", index=False)
print("Saved metrics_ablation_study.csv")
print(df.groupby("model").size()) 

metric_list = ["psnr", "ssim", "lpips", "niqe", "pi", "clipiqa", "musiq"]
metric_label = {
    "psnr": "PSNR (dB) ↑",   
    "ssim": "SSIM ↑",
    "lpips": "LPIPS ↓",      
    "niqe": "NIQE ↓",
    "pi": "PI ↓",
    "clipiqa": "CLIPIQA ↑",
    "musiq": "MUSIQ ↑",
}

for metric in metric_list:
    plt.figure(figsize=(8, 6))
    


    data_base = df[df["model"] == "Baseline (Original)"].sort_values("step")
    plt.plot(data_base["step"], data_base[metric], marker="o", linestyle="--", label="Baseline", color="gray", alpha=0.7)

    data_add = df[df["model"] == "Ours (Simple Add)"].sort_values("step")
    plt.plot(data_add["step"], data_add[metric], marker="^", linestyle="-.", label="Ours (Simple Add)", color="blue")

    data_sft = df[df["model"] == "Ours (SFT Injection)"].sort_values("step")
    plt.plot(data_sft["step"], data_sft[metric], marker="*", linestyle="-", linewidth=2, label="Ours (SFT)", color="red")

    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel(metric_label.get(metric, metric), fontsize=12)
    plt.title(f"Ablation Study: {metric.upper()}", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    out_path = f"ablation_{metric}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")