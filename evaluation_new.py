#!/usr/bin/env python3
import argparse, os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models_x import *          # LUTs / Classifier / TrilinearInterpolation
from datasets import *          # ImageDataset_sRGB / ImageDataset_XYZ
from utils.config import load_and_merge_config
from torchvision.transforms.functional import to_pil_image

# ----------------------------
# Args + config (same pattern)
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="YAML/JSON config file with model_dir/epoch/etc.")
parser.add_argument("--gpus", type=int, default=1, help="DataParallel over N GPUs if >1")
# Fallbacks if not in config
parser.add_argument("--epoch", type=int, default=145, help="checkpoint epoch to load")
parser.add_argument("--model_dir", type=str, default="LUTs/paired/fiveK_480p_3LUT_sm_1e-4_mn_10", help="dir of saved models")
parser.add_argument("--input_color_space", type=str, default="sRGB", choices=["sRGB","XYZ"])
parser.add_argument("--dataset_name", type=str, default="fiveK")
parser.add_argument("--batch_size", type=int, default=1, help="keep 1 if using trilinear path")
parser.add_argument("--n_cpu", type=int, default=4)
parser.add_argument("--out_dir", type=str, default="", help="optional output dir; defaults to <model_dir>_eval_<epoch>")
# for CSV-based sRGB test set (matches your train/unpaired loader style)
parser.add_argument("--test_csv", type=str, default="", help="optional CSV for test set; used by ImageDataset_sRGB_unpaired_CSV")
parser.add_argument("--export_ext", type=str, default="png", choices=["png","jpg","jpeg"],
                    help="Output format; both saved at highest quality")
parser.add_argument("--use_global_weight", action="store_true",help="If set, use one global weight (median over loader) instead of image-specific weights.")
parser.add_argument("--global_type", type=str, default="median",help="using mean/median/mode global weight (median over loader) instead of image-specific weights.")
parser.add_argument("--global_sampling_rate", type=float, default="1.0",help="pctg of tiles used to calcule global weight for wsi")
cfg, _, eff = load_and_merge_config(parser)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = (device.type=="cuda")

# ----------------
# Helpers to calculate slide-specific and population-specific (i.e. across ) median weight
# ----------------


@torch.no_grad()
def compute_median_weight(loader, classifier, device):
    ws, rows = [], []
    classifier.eval()
    for batch in loader:
        x = batch["A_input"].to(device, non_blocking=True)
        # Vectorized classifier pass
        p = classifier(x)
        if p.dim() == 1:
            p = p.unsqueeze(0)
        p = p.view(x.size(0), -1)[:, :3]  # [B,3]
        p_cpu = p.detach().float().cpu()
        ws.append(p_cpu)
        # collect CSV rows (tile-level)
        names = batch.get("input_name", [""] * p_cpu.size(0))
        wsis = batch.get("file_name", [""] * p_cpu.size(0))
        for i in range(p_cpu.size(0)):
            rows.append({
                "file_name": str(wsis[i]),
                "input_name": str(names[i]),
                "w0": float(p_cpu[i, 0]), "w1": float(p_cpu[i, 1]), "w2": float(p_cpu[i, 2]),
            })
    W = torch.cat(ws, dim=0)                  # [N,3] on CPU
    med = W.median(dim=0).values.to(device)   # [3] on device
    return med, W, rows

@torch.no_grad()
def compute_mean_weight(loader, classifier, device):
    ws, rows = [], []
    classifier.eval()
    for batch in loader:
        x = batch["A_input"].to(device, non_blocking=True)
        p = classifier(x).view(x.size(0), -1)[:, :3]
        p_cpu = p.detach().float().cpu()
        ws.append(p_cpu)
        names = batch.get("input_name", [""] * p_cpu.size(0))
        wsis = batch.get("file_name", [""] * p_cpu.size(0))
        for i in range(p_cpu.size(0)):
            rows.append({
                "file_name": str(wsis[i]),
                "input_name": str(names[i]),
                "w0": float(p_cpu[i, 0]), "w1": float(p_cpu[i, 1]), "w2": float(p_cpu[i, 2]),
            })
    W = torch.cat(ws, dim=0)
    mean = W.mean(dim=0).to(device)
    return mean, W, rows

@torch.no_grad()
def compute_mode_weight(loader, classifier, device):
    ws, rows = [], []
    classifier.eval()
    for batch in loader:
        x = batch["A_input"].to(device, non_blocking=True)
        p = classifier(x).view(x.size(0), -1)[:, :3]
        p_cpu = p.detach().float().cpu()
        ws.append(p_cpu)
        names = batch.get("input_name", [""] * p_cpu.size(0))
        wsis = batch.get("file_name", [""] * p_cpu.size(0))
        for i in range(p_cpu.size(0)):
            rows.append({
                "file_name": str(wsis[i]),
                "input_name": str(names[i]),
                "w0": float(p_cpu[i, 0]), "w1": float(p_cpu[i, 1]), "w2": float(p_cpu[i, 2]),
            })
    W = torch.cat(ws, dim=0)
    mode = W.mode(dim=0).to(device)
    return mode, W, rows


# ----------------
# Pillow JPEG saver
# ----------------
def save_max_quality(t: torch.Tensor, out_stem: Path, fmt: str):
    """
    t: [C,H,W] in [0,1]; out_stem: path WITHOUT extension
    fmt: 'png' or 'jpg'/'jpeg'
    """
    fmt = fmt.lower()
    if fmt == "jpeg": fmt = "jpg"

    img = torch.nan_to_num(t.detach().cpu(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
    pil = to_pil_image(img).convert("RGB")  # convert [0,1] to [0, 255], and change shape 
                                            # from C, H, W to H, W, C

    out_path = Path(out_stem).with_suffix("." + fmt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "jpg": # [0, 255]
        # Highest practical visual quality (still lossy by design)
        pil.save(out_path, format="JPEG", quality=100, subsampling=0, optimize=True, progressive=True)
    else:  # PNG   [0,1]
        # PNG is lossless; compress_level affects size/time, not quality
        pil.save(out_path, format="PNG", optimize=True, compress_level=0)  # 0 = fastest, still lossless



# ------------
# Model/build
# ------------
criterion_pixelwise = torch.nn.MSELoss().to(device)  # unused here but kept for parity
LUT0 = Generator3DLUT_identity()
LUT1 = Generator3DLUT_zero()
LUT2 = Generator3DLUT_zero()
classifier = Classifier_unpaired()
trilinear_ = TrilinearInterpolation()     # original paired path

if cuda:
    LUT0 = LUT0.cuda(); LUT1 = LUT1.cuda(); LUT2 = LUT2.cuda()
    classifier = classifier.cuda()
    trilinear_.cuda()

# ---- load weights ----  needs to revise here 
mroot = Path(cfg.model_dir)

if (mroot / f"LUTs_best_fid.pth").exists():
    luts_path = mroot / f"LUTs_best_fid.pth"
    cls_path  = mroot / f"classifier_best_fid.pth"

#if (mroot / f"LUTs_best_kid.pth").exists():
#    luts_path = mroot / f"LUTs_best_kid.pth"
#    cls_path  = mroot / f"classifier_best_kid.pth"

elif (mroot / f"LUTs_{cfg.epoch}.pth").exists():
    luts_path = mroot / f"LUTs_{cfg.epoch}.pth"
    cls_path  = mroot / f"classifier_{cfg.epoch}.pth"
else: # homedir
    luts_path = Path("saved_models") / cfg.model_dir / f"LUTs_{cfg.epoch}.pth"
    cls_path  = Path("saved_models") / cfg.model_dir / f"classifier_{cfg.epoch}.pth"
print(f'Best LUT path: {luts_path}')

# ---- only change: use weights_only=True ----
LUTs = torch.load(luts_path, map_location=device, weights_only=True)
LUT0.load_state_dict(LUTs["0"]); LUT1.load_state_dict(LUTs["1"]); LUT2.load_state_dict(LUTs["2"])

cls_sd = torch.load(cls_path, map_location=device, weights_only=True)
# (Optional one-liner in case the checkpoint came from DataParallel)
# cls_sd = {k.replace("module.", "", 1): v for k, v in cls_sd.items()}
classifier.load_state_dict(cls_sd)
LUT0.eval(); LUT1.eval(); LUT2.eval(); classifier.eval()


# ----------------
# Generator wrapper
# ----------------
# using tile specific weights
class EvalGenerator(nn.Module):
    def __init__(self, lut0, lut1, lut2, classifier, trilerp):
        super().__init__()
        self.lut0 = lut0; self.lut1 = lut1; self.lut2 = lut2
        self.classifier = classifier
        self.tri = trilerp

    @torch.no_grad()
    def forward(self, img, fixed_w: torch.Tensor = None):
        """
        img: [B,3,H,W]
        fixed_w: optional [3] or [B,3]; if provided, use this instead of per-image classifier.
        """
        B = img.size(0)

        # If user supplied a dataset-level fixed weight
        if fixed_w is not None:
            if fixed_w.dim() == 1:              # [3] -> [B,3]
                w = fixed_w.view(1, 3).expand(B, 3)
            else:
                w = fixed_w
            #w = _project_simplex_nonneg(w)
            # Fallback trilinear path with fixed weights
            outs = []
            for b in range(B):
                x = img[b:b+1]
                LUT = w[b,0]*self.lut0.LUT + w[b,1]*self.lut1.LUT + w[b,2]*self.lut2.LUT
                _, y = self.tri(LUT, x)
                outs.append(y)
            out = torch.cat(outs, dim=0).clamp(0,1)
            return out, w   
        # default using image specific weight Vectorized classifier pass (big speedup)
        pred = self.classifier(img)                  # [B,3] or [B,3,1,1]
        if pred.dim() > 2:
            pred = pred.view(B, -1)[:, :3]
        else:
            pred = pred.view(B, -1)[:, :3]

        outs = []
        for b in range(B):
            x = img[b:b+1]
            p = pred[b]
            LUT = p[0]*self.lut0.LUT + p[1]*self.lut1.LUT + p[2]*self.lut2.LUT
            _, y = self.tri(LUT, x)
            outs.append(y)
        out = torch.cat(outs, dim=0).clamp(0, 1)
        return out, pred

gen_core = EvalGenerator(LUT0, LUT1, LUT2, classifier, trilinear_).to(device)

if cuda and cfg.gpus > 1 and torch.cuda.device_count() > 1:
    gen = nn.DataParallel(gen_core, device_ids=list(range(min(cfg.gpus, torch.cuda.device_count()))))
    print(f"[Eval] DataParallel on {min(cfg.gpus, torch.cuda.device_count())} GPUs")
else:
    gen = gen_core
    print("[Eval] Single GPU/CPU")


# ------------
# Output dir
# ------------
run_root = Path(cfg.output_dir) / cfg.run_name
ext_tag = str(cfg.export_ext).lower().lstrip('.')  # e.g., "png" or "jpg"
tag = f"best_fid_{ext_tag}" if "best_fid" in str(luts_path) else f"epoch_{cfg.epoch}_{ext_tag}"
out_dir = run_root / "evaluation" / tag
out_dir.mkdir(parents=True, exist_ok=True)
print(f"[Eval] Writing to: {out_dir}")

# study-level summary file once per run
summary_csv = (run_root / "evaluation" / tag / "wsi_global_weights.csv")
summary_csv.parent.mkdir(parents=True, exist_ok=True)
summary_header_needed = not summary_csv.exists()

# read in image list 

if cfg.input_color_space == "sRGB":
    if getattr(cfg, "test_csv", ""):
        file_name = "file_name"
        col_name = 'crude_tile_path'
        usecols = [file_name, col_name, 'tissue_proportion', "scanner_model_new", "split"]
        df_all = pd.read_csv(cfg.test_csv, usecols=usecols, dtype={col_name:str, "scanner_model_new":str, "split":str})
        df = df_all.dropna(subset=usecols).copy()
        # df = df_all[df_all['scanner_model_new']=='PHILIPS']  # only PHILIPS slides

from tqdm.auto import tqdm  # auto picks notebook/terminal backend

unique_files = df['file_name'].unique()
bar = tqdm(unique_files, total=len(unique_files), desc="Processing WSIs")
bar.set_description(f"Processing  WSIs")




for i, wsi in enumerate(bar, start=1):

    # ------------
    # Data loader
    # ------------
    df_sub = df[df['file_name'] == wsi].reset_index(drop=True).copy()
    test_domain = df_sub['scanner_model_new'].iloc[0] # should work for all images

    if cfg.input_color_space == "sRGB":
        # ds = ImageDataset_sRGB_unpaired_CSV_inference(df_sub, mode="test", test_domain="PHILIPS")
        ds = ImageDataset_sRGB_unpaired_CSV_inference_v2(df_sub, mode="test", source_domain=test_domain, target_domain="XR") # should work for all images

    
    # Optional: update the bar label/postfix with live info
    bar.set_description(f"WSI {i}/{len(unique_files)}")
    bar.set_postfix(n_tiles=len(ds), weight=cfg.global_type if cfg.use_global_weight else "image-specific", refresh=False)



    # ------------ add line to check if image-specific weight or global weight exist ------------
    # One pass to compute median weights
    weight_loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.n_cpu, pin_memory=cuda,
    )

    if cfg.use_global_weight:
        if cfg.global_sampling_rate < 1:
            ds_weight = ImageDataset_sRGB_unpaired_CSV_inference_v2(df_sub.sample(frac=cfg.global_sampling_rate, random_state=42), mode="test", source_domain=test_domain, target_domain="XR") # should work for all images
            global_w_loader = DataLoader(ds_weight, batch_size=cfg.batch_size, shuffle=False,
                num_workers=cfg.n_cpu, pin_memory=cuda,
            )
        else:
            global_w_loader = weight_loader
        if "median" in cfg.global_type:
            global_w, W, rows = compute_median_weight(global_w_loader, classifier, device)  # classifer(x) is embedded
            tqdm.write(f"[Eval] Median weight (projected): {global_w.detach().cpu().tolist()}")
        elif "mean" in cfg.global_type:
            global_w, W, rows = compute_mean_weight(global_w_loader, classifier, device)
            tqdm.write(f"[Eval] Mean weight (projected): {global_w.detach().cpu().tolist()}")
        elif "mode" in cfg.global_type:
            global_w, W, rows = compute_mode_weight(global_w_loader, classifier, device)
            tqdm.write(f"[Eval] Mean weight (projected): {global_w.detach().cpu().tolist()}")
        else:
            raise ValueError(...)
        
        

        # --- SAVE per-tile weights for THIS WSI ---
        # after you compute: global_w, W, rows
        wsi_dir = out_dir / wsi 
        wsi_dir.mkdir(parents=True, exist_ok=True)

        
        pd.DataFrame(rows, columns=["file_name","input_name","w0","w1","w2"]) \
        .to_csv(wsi_dir / "tile_weights.csv", index=False)


        # --- APPEND one-row WSI summary (global weight) ---
        pd.DataFrame([{
            "file_name": wsi,
            "reduce": "median" if "median" in cfg.global_type else "mean",
            "n_tiles": int(W.size(0)),
            "w0": float(global_w[0].detach().cpu()),
            "w1": float(global_w[1].detach().cpu()),
            "w2": float(global_w[2].detach().cpu()),
        }]).to_csv(summary_csv, mode="a", header=summary_header_needed, index=False)
        summary_header_needed = False


 

    # Fresh loader for image writing
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.n_cpu, pin_memory=cuda,
    )



    # ---------- Run + save using fixed median weight ----------

    if cfg.use_global_weight:
        print('==================using global weight =================')

        @torch.no_grad()
        def run_and_save():
            for batch in tqdm(loader, desc=f"Tiles in {wsi}", leave=False):
                real_A = batch["A_input"].to(device, non_blocking=True)
                B = real_A.size(0)
                names = batch.get("input_name", None)

                # expand median to [B,3] on the right device
                fixed_batch_w = global_w.view(1, 3).expand(B, 3).to(real_A.device)
                #print(f' fixed_batch_w ', fixed_batch_w)

                fake_B, _ = gen(real_A, fixed_w=fixed_batch_w)   # OK with DataParallel now
                fake_B = fake_B.detach().cpu().clamp(0,1)
                B = fake_B.size(0)
                for k in range(B):
                    base = os.path.basename(str(names[k])) if names is not None else f"{len(os.listdir(wsi_dir)) + k:06d}"
                    root, _ = os.path.splitext(base)
                    save_max_quality(fake_B[k], wsi_dir / root, cfg.export_ext)
        


    else: # needs to revisit
    # ------------
    # Run + save (JPEG via Pillow)
    # ------------
        print('==================using image-specific weight =================')
        @torch.no_grad()
        def run_and_save():
           for batch in tqdm(loader, desc=f"Tiles in {wsi}", leave=False):
                real_A = batch["A_input"].to(device, non_blocking=True)
                names = batch.get("input_name", None)
                fake_B, _ = gen(real_A)
                fake_B = fake_B.detach().cpu().clamp(0, 1)
                B = fake_B.size(0)
                for k in range(B):
                    base = os.path.basename(str(names[k])) if names is not None else f"{len(os.listdir(wsi_dir)) + k:06d}"
                    root, _ = os.path.splitext(base)
                    save_max_quality(fake_B[k], wsi_dir / root, cfg.export_ext)
    
       


    if __name__ == "__main__":
        run_and_save()
        tqdm.write("[Eval] Done.")