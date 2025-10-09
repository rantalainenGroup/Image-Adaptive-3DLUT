import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
from pathlib import Path

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd

from models_x import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.transforms.functional import to_pil_image
import shutil, os



# === NEW/CHANGED ===
from utils.config import load_and_merge_config, save_configs

# ========= NEW: metrics import (safe) =========
try:
    from torch_fidelity import calculate_metrics as _tf_calculate_metrics
    _HAS_TF = True
except Exception:
    _HAS_TF = False
import shutil


parser = argparse.ArgumentParser()
# === NEW/CHANGED === config + gpus
parser.add_argument("--config", type=str, help="YAML/JSON config file")
parser.add_argument("--gpus", type=int, default=1, help="number of GPUs for DataParallel (>=2 enables DP)")
#--data 
parser.add_argument("--train_csv", type=str, default="", help="CSV for unpaired training (provides A_input and B_exptC)")
parser.add_argument("--test_csv", type=str, default="", help="Optional paired test CSV for PSNR")
parser.add_argument("--val_csv", type=str, default="", help="CSV with val/test rows for PHILIPS & XR (for FID/KID)")

parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from, 0 starts from scratch, >0 starts from saved checkpoints")
parser.add_argument("--n_epochs", type=int, default=300, help="total number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="scanb_malmo", help="name of the dataset")

# === NEW/CHANGED === expose input_color_space (your code uses it later)
parser.add_argument("--input_color_space", type=str, default="sRGB", choices=["sRGB","XYZ"], help="input color space")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--lambda_pixel", type=float, default=1000, help="content preservation weight: 1000 for sRGB input, 10 for XYZ input")
parser.add_argument("--lambda_gp", type=float, default=10, help="gradient penalty weight in wgan-gp")
parser.add_argument("--lambda_smooth", type=float, default=1e-4, help="smooth regularization")
parser.add_argument("--lambda_monotonicity", type=float, default=10.0, help="monotonicity regularization: 10 for sRGB input, 100 for XYZ input (slightly better)")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_critic", type=int, default=1, help="number of training steps for discriminator per iter")
parser.add_argument("--output_dir", type=str, default="/mnt/ssd/bojing/Image-Adaptive-3DLUT/LUTs/unpaired/", help="path to save model")
parser.add_argument("--run_name", type=str, default="exp0", help="optional subfolder name; if empty, use output_dir")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints")

# FID/KID
parser.add_argument("--fid_every", type=int, default=25, help="Compute FID/KID every N epochs (0=disable)")
parser.add_argument("--val_b_corpus", type=str, default="", help="Optional folder of real XR validation images")
parser.add_argument("--export_dir", type=str, default="", help="Where to write exports; default <run_dir>/metrics")
parser.add_argument("--keep_fid_images", action="store_true", help="Keep exported fake images (default: delete)")
parser.add_argument("--export_ext", type=str, default="png", choices=["png","jpg","jpeg"], help="Output format for generated images (default: png)")


# === NEW/CHANGED === config merge + run dir
cfg, run_dir, eff = load_and_merge_config(parser)
save_configs(run_dir, eff, getattr(cfg, "config", None))
print(f"[Run dir] {run_dir}")
print(cfg)

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Loss functions
criterion_GAN = torch.nn.MSELoss().to(device)
criterion_pixelwise = torch.nn.MSELoss().to(device)

# Initialize generator and discriminator
LUT0 = Generator3DLUT_identity()
LUT1 = Generator3DLUT_zero()
LUT2 = Generator3DLUT_zero()
classifier = Classifier_unpaired()
discriminator = Discriminator()
TV3 = TV_3D()

if cuda:
    LUT0 = LUT0.cuda()
    LUT1 = LUT1.cuda()
    LUT2 = LUT2.cuda()
    classifier = classifier.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()
    discriminator = discriminator.cuda()
    TV3 = TV_3D().cuda()
    TV3.weight_r = TV3.weight_r.type(Tensor)
    TV3.weight_g = TV3.weight_g.type(Tensor)
    TV3.weight_b = TV3.weight_b.type(Tensor)

# === NEW/CHANGED === minimal wrapper module so we can DataParallel the generator path
class LUTGenerator(nn.Module):
    def __init__(self, lut0, lut1, lut2, classifier):
        super().__init__()
        self.LUT0 = lut0
        self.LUT1 = lut1
        self.LUT2 = lut2
        self.classifier = classifier

    def forward(self, img):
        pred = self.classifier(img)           # [B,3]
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        pred = pred.view(img.size(0), -1)
        weights_norm = torch.mean(pred ** 2)
        w0 = pred[:, 0].view(-1,1,1,1)
        w1 = pred[:, 1].view(-1,1,1,1)
        w2 = pred[:, 2].view(-1,1,1,1)
        out = w0 * self.LUT0(img) + w1 * self.LUT1(img) + w2 * self.LUT2(img)
        return out, weights_norm

# === NEW/CHANGED === build generator module and (optionally) wrap in DataParallel
generator_core = LUTGenerator(LUT0, LUT1, LUT2, classifier).to(device)
if cuda and cfg.gpus > 1 and torch.cuda.device_count() > 1:
    generator = nn.DataParallel(generator_core, device_ids=list(range(min(cfg.gpus, torch.cuda.device_count()))))
    discriminator = nn.DataParallel(discriminator, device_ids=list(range(min(cfg.gpus, torch.cuda.device_count()))))
    print(f"[Multi-GPU] DataParallel on {min(cfg.gpus, torch.cuda.device_count())} GPUs")
else:
    generator = generator_core
    print("[Multi-GPU] Disabled (single GPU or CPU)")

# === NEW/CHANGED === use cfg instead of opt/args for resume
if cfg.epoch != 0:
    # Load pretrained models from run_dir
    LUTs = torch.load(str(run_dir / f"LUTs_{cfg.epoch}.pth"), map_location=device)
    gen_ref = generator.module if isinstance(generator, nn.DataParallel) else generator
    gen_ref.LUT0.load_state_dict(LUTs["0"])
    gen_ref.LUT1.load_state_dict(LUTs["1"])
    gen_ref.LUT2.load_state_dict(LUTs["2"])
    gen_ref.classifier.load_state_dict(torch.load(str(run_dir / f"classifier_{cfg.epoch}.pth"), map_location=device))
else:
    # Initialize weights
    gen_ref = generator.module if isinstance(generator, nn.DataParallel) else generator
    gen_ref.classifier.apply(weights_init_normal_classifier)
    torch.nn.init.constant_(gen_ref.classifier.model[12].bias.data, 1.0)
    disc_ref = discriminator.module if isinstance(discriminator, nn.DataParallel) else discriminator
    disc_ref.apply(weights_init_normal_classifier)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(
        (generator.parameters())  # works for DP or single
    ),
    lr=cfg.lr, betas=(cfg.b1, cfg.b2)
)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))

# === NEW/CHANGED === use cfg.* and keep your dataset wiring
if cfg.input_color_space == 'sRGB':
    dataloader = DataLoader(
        ImageDataset_sRGB_unpaired_CSV(cfg.train_csv, mode="train"),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.n_cpu,
        pin_memory=True, persistent_workers=(cfg.n_cpu > 0), prefetch_factor=(4 if cfg.n_cpu > 0 else None),
    )

    ## === NEW/CHANGED === test/psnr loader (only if cfg.test_csv is set)
    
    psnr_dataloader = None
    """
    if getattr(cfg, "val_csv", None):
        psnr_dataloader = DataLoader(
            ImageDataset_sRGB_unpaired_CSV(cfg.val_csv, mode="test", test_domain=cfg.test_domain),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=(device.type == "cuda"),
        )
    """
else:
    dataloader = DataLoader(
        ImageDataset_XYZ_unpaired("../data/%s" % cfg.dataset_name, mode="train"),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.n_cpu,
    )

    psnr_dataloader = DataLoader(
        ImageDataset_XYZ_unpaired("../data/%s" % cfg.dataset_name,  mode="test"),
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

# ========= NEW: safely image  with max quality

def save_max_quality(t: torch.Tensor, out_stem: Path, fmt: str):
    """
    t: [C,H,W] in [0,1]; out_stem: path WITHOUT extension
    fmt: 'png' or 'jpg'/'jpeg'
    """
    fmt = fmt.lower()
    if fmt == "jpeg": fmt = "jpg"

    img = torch.nan_to_num(t.detach().cpu(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
    pil = to_pil_image(img).convert("RGB")

    out_path = Path(out_stem).with_suffix("." + fmt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "jpg":
        pil.save(out_path, format="JPEG", quality=100, subsampling=0, optimize=True, progressive=True)
    else:  # PNG is lossless; compress level only affects size/speed
        pil.save(out_path, format="PNG", optimize=True, compress_level=0)


# ========= NEW: validation loaders for FID/KID (from SAME CSV) =========
val_A_loader = val_B_loader = None
if getattr(cfg, "val_csv", ""):
    val_A_loader = DataLoader(
        ImageDataset_sRGB_unpaired_CSV(cfg.val_csv, mode="test", test_domain="PHILIPS"),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=max(1, cfg.n_cpu//2), pin_memory=(device.type=="cuda")
    )
    val_B_loader = DataLoader(
        ImageDataset_sRGB_unpaired_CSV(cfg.val_csv, mode="test", test_domain="XR"),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=max(1, cfg.n_cpu//2), pin_memory=(device.type=="cuda")
    )



# --- PSNR helper (skips if no psnr_loader/no paired data) ---
def calculate_psnr():
    if psnr_dataloader is None:
        return None
    # classifier is inside generator; eval generator is enough
    was_train = generator.training
    generator.eval()
    avg_psnr = 0
    for i, batch in enumerate(psnr_dataloader):
        real_A = Variable(batch["A_input"].type(Tensor))
        real_B = Variable(batch["A_exptC"].type(Tensor))
        fake_B, weights_norm = generator(real_A)
        fake_B = torch.round(fake_B*255)
        real_B = torch.round(real_B*255)
        mse = criterion_pixelwise(fake_B, real_B)
        psnr = 10 * math.log10(255.0 * 255.0 / max(mse.item(), 1e-12))
        avg_psnr += psnr
    if was_train:
        generator.train()
    return avg_psnr/ max(len(psnr_dataloader), 1)

# --- WGAN-GP helper (device-safe tensors) ---
def compute_gradient_penalty(D, real_samples, fake_samples):
    device = real_samples.device; dtype = real_samples.dtype
    alpha = torch.rand((real_samples.size(0),1,1,1), device=device, dtype=dtype)
    interpolates = (alpha*real_samples + (1-alpha)*fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones((real_samples.size(0),1,1,1), device=device, dtype=dtype)
    gradients = autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


"""
# original
def compute_gradient_penalty(D, real_samples, fake_samples):
    #Calculates the gradient penalty loss for WGAN GP
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1, 1, 1).fill_(1.0), requires_grad=False)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
"""
# filename helpers (preserve original names safely)

def _unique_path(dest_dir: Path, filename: str) -> Path:
    out = dest_dir / filename
    if not out.exists():
        return out
    stem, ext, i = out.stem, out.suffix, 1
    while True:
        cand = dest_dir / f"{stem}_{i}{ext}"
        if not cand.exists():
            return cand
        i += 1
"""
@torch.no_grad()
def export_generated_from_valA(val_loader, generator, out_dir: Path, Tensor):
    out_dir.mkdir(parents=True, exist_ok=True)
    was = generator.training
    generator.eval()
    idx = 0
    for batch in val_loader:
        x = batch["A_input"].type(Tensor)
        y, _ = generator(x)
        names = batch.get("input_name", None)
        for k in range(y.size(0)):
            if names is not None:
                # use original filename as-is, just strip any path
                fname = os.path.basename(str(names[k]))
                # default extension if missing
                if not os.path.splitext(fname)[1]:
                    fname = fname + ".jpg"
            else:
                fname = f"{idx:06d}.jpg"
            out_path = _unique_path(out_dir, fname)
            img = y[k].detach().cpu().clamp(0, 1)  # guarantees [0,1]
            save_image(img, out_path, normalize=False)
            idx += 1
    if was: generator.train()
    return idx
"""

@torch.no_grad()
def export_generated_from_valA(val_loader, generator, out_dir: Path, Tensor, export_ext: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    was = generator.training
    generator.eval()
    idx = 0
    for batch in val_loader:
        x = batch["A_input"].type(Tensor)
        y, _ = generator(x)
        names = batch.get("input_name", None)
        for k in range(y.size(0)):
            base = os.path.basename(str(names[k])) if names is not None else f"{idx:06d}"
            stem, _ = os.path.splitext(base)
            # pass cfg.export_ext here
            save_max_quality(y[k], out_dir / stem, export_ext)
            idx += 1
    if was: generator.train()
    return idx

"""
@torch.no_grad()
def export_real_B_from_valB(val_loader, out_dir: Path, Tensor):
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    for batch in val_loader:
        xr = batch["A_input"].type(Tensor)
        names = batch.get("input_name", None)
        for k in range(xr.size(0)):
            if names is not None:
                fname = os.path.basename(str(names[k]))
                if not os.path.splitext(fname)[1]:
                    fname = fname + ".jpg"
            else:
                fname = f"{idx:06d}.jpg"
            out_path = _unique_path(out_dir, fname)
            img = xr[k].detach().cpu().clamp(0, 1)  # guarantees [0,1]
            save_image(img, out_path, normalize=False)
            idx += 1
    return idx
"""

@torch.no_grad()
def export_real_B_from_valB(val_loader, out_dir: Path, Tensor, export_ext: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    for batch in val_loader:
        # Prefer copying the original JPEGs if we have their paths
        srcs = batch.get("src_path", None) # here is buggy, copying does not seem to work
        names = batch.get("input_name", None)
        if srcs is not None:
            for k in range(len(srcs)):
                src = Path(str(srcs[k]))
                base = os.path.basename(str(names[k])) if names is not None else src.name
                dst = _unique_path(out_dir, base)
                shutil.copy2(src, dst)  # no re-encode (ideal for FID/KID)
                idx += 1
        else:
            # fallback: save tensor with requested format
            xr = batch["A_input"].type(Tensor)
            for k in range(xr.size(0)):
                base = os.path.basename(str(names[k])) if names is not None else f"{idx:06d}"
                stem, _ = os.path.splitext(base)
                save_max_quality(xr[k], out_dir / stem, export_ext)
                idx += 1
    return idx

def _compute_fid_kid(dir_fake: Path, dir_real: Path):
    metrics = _tf_calculate_metrics(
        input1=str(dir_fake), input2=str(dir_real),
        cuda=cuda, isc=False, fid=True, kid=True, verbose=False,
        batch_size=64, num_workers=max(1, cfg.n_cpu//2)
    )
    fid = float(metrics["frechet_inception_distance"])
    kid = float(metrics["kernel_inception_distance_mean"])
    return fid, kid



def visualize_result(epoch):
    """Saves a generated sample from the validation set"""
    os.makedirs("images/LUTs/" +str(epoch), exist_ok=True)
    was_train = generator.training
    generator.eval()
    for i, batch in enumerate(psnr_dataloader):
        real_A = Variable(batch["A_input"].type(Tensor))
        real_B = Variable(batch["A_exptC"].type(Tensor))
        img_name = batch["input_name"]
        fake_B, weights_norm = generator(real_A)
        img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -1)
        fake_B = torch.round(fake_B*255)
        real_B = torch.round(real_B*255)
        mse = criterion_pixelwise(fake_B, real_B)
        psnr = 10 * math.log10(255.0 * 255.0 / max(mse.item(), 1e-12))
        save_image(img_sample, "images/LUTs/%s/%s.jpg" % (epoch, img_name[0]+'_'+str(psnr)[:5]), nrow=3, normalize=False)
    if was_train:
        generator.train()


# ----------
#  Training
# ----------
avg_psnr = calculate_psnr()
print(avg_psnr)
best_fid = float('inf')     # initialize best tracker
best_fid_epoch = None
prev_time = time.time()
max_psnr = 0
max_epoch = 0

for epoch in range(cfg.epoch, cfg.n_epochs):
    loss_D_avg = 0
    loss_G_avg = 0
    loss_pixel_avg = 0
    cnt = 0
    psnr_avg = 0
    # classifier now lives inside generator
    generator.train()
    for i, batch in enumerate(dataloader):

        # Model inputs
        real_A = Variable(batch["A_input"].type(Tensor))
        real_B = Variable(batch["B_exptC"].type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        fake_B, weights_norm = generator(real_A)
        # === NEW/CHANGED ===↓↓↓ Minimal fix: reduce DP-stacked scalars to one scalar
        if isinstance(weights_norm, torch.Tensor) and weights_norm.dim() > 0:
            weights_norm = weights_norm.mean()
        pred_real = discriminator(real_B)
        pred_fake = discriminator(fake_B.detach())
        gradient_penalty = compute_gradient_penalty(discriminator, real_B, fake_B.detach())
        loss_D = -torch.mean(pred_real) + torch.mean(pred_fake) + cfg.lambda_gp * gradient_penalty
        loss_D.backward()
        optimizer_D.step()
        loss_D_avg += (-torch.mean(pred_real) + torch.mean(pred_fake)) / 2

        # ------------------
        #  Train Generators
        # ------------------
        if i % cfg.n_critic == 0:
            optimizer_G.zero_grad()
            fake_B, weights_norm = generator(real_A)
            # === NEW/CHANGED === ↓↓↓ Minimal fix: reduce DP-stacked scalars to one scalar
            if isinstance(weights_norm, torch.Tensor) and weights_norm.dim() > 0:
                weights_norm = weights_norm.mean()
            pred_fake = discriminator(fake_B)
            # Pixel-wise loss (unpaired): keep close to input
            loss_pixel = criterion_pixelwise(fake_B, real_A)

            # === NEW/CHANGED === TV/monotonicity on the underlying module (DP-safe)
            gen_ref = generator.module if isinstance(generator, nn.DataParallel) else generator
            tv0, mn0 = TV3(gen_ref.LUT0)
            tv1, mn1 = TV3(gen_ref.LUT1)
            tv2, mn2 = TV3(gen_ref.LUT2)
            tv_cons = tv0 + tv1 + tv2
            mn_cons = mn0 + mn1 + mn2

            loss_G = -torch.mean(pred_fake) + cfg.lambda_pixel * loss_pixel + cfg.lambda_smooth * (weights_norm + tv_cons) + cfg.lambda_monotonicity * mn_cons
            loss_G.backward()
            optimizer_G.step()

            cnt += 1
            loss_G_avg += -torch.mean(pred_fake)
            loss_pixel_avg += loss_pixel
            psnr_avg += 10 * math.log10(1 / max(loss_pixel.item(), 1e-12))

        # --------------
        #  Log Progress
        # --------------
        batches_done = epoch * len(dataloader) + i
        batches_left = cfg.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D: %f, G: %f] [pixel: %f] [tv: %f, wnorm: %f, mn: %f] ETA: %s"
            % (
                epoch,
                cfg.n_epochs,
                i,
                len(dataloader),
                loss_D_avg.item() / max(cnt,1),
                loss_G_avg.item() / max(cnt,1),
                loss_pixel_avg.item() / max(cnt,1),
                tv_cons, weights_norm, mn_cons,
                time_left,
            )
        )
    # PSNR (paired test only)
    avg_psnr = calculate_psnr()
    if avg_psnr is None:
        sys.stdout.write(" [PSNR: skipped]\n")

    else:
        if avg_psnr > max_psnr:
            max_psnr = avg_psnr
            max_epoch = epoch
        sys.stdout.write(" [PSNR: %f] [max PSNR: %f, epoch: %d]\n"% (avg_psnr, max_psnr, max_epoch))

    # FID/KID (unpaired val)

    if _HAS_TF and cfg.fid_every > 0 and (val_A_loader is not None) and (val_B_loader is not None):
        if (epoch % cfg.fid_every == 0) or (epoch == cfg.n_epochs - 1):
            export_root = Path(cfg.export_dir) if getattr(cfg, "export_dir", "") else (Path(run_dir) / "metrics")
            fake_dir = export_root / "fake_tmp"
            real_dir = Path(cfg.val_b_corpus) if getattr(cfg, "val_b_corpus", "") else (export_root / "realB_cache")

            # clean/reuse fake_tmp
            shutil.rmtree(fake_dir, ignore_errors=True)
            fake_dir.mkdir(parents=True, exist_ok=True)

            n_fake = export_generated_from_valA(val_A_loader, generator, fake_dir, Tensor, cfg.export_ext)


            # build real XR cache once if not provided
            if not getattr(cfg, "val_b_corpus", ""):
                if not real_dir.exists() or not any(real_dir.iterdir()):
                    real_dir.mkdir(parents=True, exist_ok=True)
                    export_real_B_from_valB(val_B_loader, real_dir, Tensor, cfg.export_ext)

            if n_fake > 0 and real_dir.exists() and any(real_dir.iterdir()):
                metrics = _tf_calculate_metrics(
                    input1=str(fake_dir), input2=str(real_dir),
                    cuda=cuda, isc=False, fid=True, kid=True, verbose=False,
                    batch_size=64, num_workers=max(1, cfg.n_cpu // 2),
                )
                fid = float(metrics["frechet_inception_distance"])
                kid = float(metrics["kernel_inception_distance_mean"])
                print(f"\n[VAL] epoch {epoch}  FID: {fid:.3f}  KID: {kid:.5f}")
                with open(Path(run_dir) / "fid_kid.csv", "a") as f:
                    if epoch == cfg.epoch:
                        f.write("epoch,fid,kid\n")
                    f.write(f"{epoch},{fid:.6f},{kid:.6f}\n")

                if fid < best_fid:
                    best_fid = fid
                    gen_ref = generator.module if isinstance(generator, nn.DataParallel) else generator
                    LUTs = {"0": gen_ref.LUT0.state_dict(), "1": gen_ref.LUT1.state_dict(), "2": gen_ref.LUT2.state_dict()}
                    torch.save(LUTs, str(run_dir / "LUTs_best_fid.pth"))
                    torch.save(gen_ref.classifier.state_dict(), str(run_dir / "classifier_best_fid.pth"))

            # cleanup fakes unless asked to keep
            if not getattr(cfg, "keep_fid_images", False):
                shutil.rmtree(fake_dir, ignore_errors=True)

    if epoch % cfg.checkpoint_interval == 0:
        # Save model checkpoints
        gen_ref = generator.module if isinstance(generator, nn.DataParallel) else generator
        LUTs = {"0": gen_ref.LUT0.state_dict(), "1": gen_ref.LUT1.state_dict(), "2": gen_ref.LUT2.state_dict()}
        torch.save(LUTs, str(run_dir / f"LUTs_{epoch}.pth"))
        torch.save(gen_ref.classifier.state_dict(), str(run_dir / f"classifier_{epoch}.pth"))
        with open(run_dir / "result.txt", "a") as f:
            if avg_psnr is None:
                f.write("[PSNR: skipped]\n")
            else:
                f.write(f"[PSNR: {avg_psnr:.6f}] [max PSNR: {max_psnr:.6f}, epoch: {max_epoch}]\n")