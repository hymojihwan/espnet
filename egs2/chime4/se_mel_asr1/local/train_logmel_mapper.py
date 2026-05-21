#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch_complex.tensor import ComplexTensor

from espnet2.layers.log_mel import LogMel
from espnet2.layers.stft import Stft


def fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def read_scp(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            utt, value = line.rstrip().split(maxsplit=1)
            data[utt] = value
    return data


class PairedWaveDataset(Dataset):
    def __init__(self, wav_scp, ref_scp, max_items=None):
        noisy = read_scp(wav_scp)
        clean = read_scp(ref_scp)
        self.items = [(utt, noisy[utt], clean[utt]) for utt in noisy.keys() if utt in clean]
        if max_items is not None:
            self.items = self.items[:max_items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        utt, noisy_path, clean_path = self.items[idx]
        noisy, _ = sf.read(noisy_path, dtype="float32")
        clean, _ = sf.read(clean_path, dtype="float32")
        noisy = torch.from_numpy(noisy).float()
        clean = torch.from_numpy(clean).float()
        return utt, noisy, clean


def collate(batch):
    utts, noisy_list, clean_list = zip(*batch)
    noisy_lens = torch.tensor([x.numel() for x in noisy_list], dtype=torch.long)
    clean_lens = torch.tensor([x.numel() for x in clean_list], dtype=torch.long)
    max_len = max(max(noisy_lens).item(), max(clean_lens).item())
    noisy = torch.zeros(len(batch), max_len)
    clean = torch.zeros(len(batch), max_len)
    for i, (n, c) in enumerate(zip(noisy_list, clean_list)):
        noisy[i, : n.numel()] = n
        clean[i, : c.numel()] = c
    return utts, noisy, noisy_lens, clean, clean_lens


class ChannelwiseLayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels, eps=1e-8)

    def forward(self, x):
        return self.norm(x)


class CleanMelBlock(nn.Module):
    def __init__(self, channels, time_kernel_size, freq_kernel_size, dropout):
        super().__init__()
        time_padding = (time_kernel_size - 1) // 2
        freq_padding = (freq_kernel_size - 1) // 2
        self.narrow_band = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(time_kernel_size, 1),
                padding=(time_padding, 0),
                groups=channels,
                bias=False,
            ),
            nn.PReLU(),
            ChannelwiseLayerNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.PReLU(),
            ChannelwiseLayerNorm2d(channels),
        )
        self.cross_band = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, freq_kernel_size),
                padding=(0, freq_padding),
                bias=False,
            ),
            nn.PReLU(),
            ChannelwiseLayerNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.PReLU(),
            ChannelwiseLayerNorm2d(channels),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        residual = x
        x = self.narrow_band(x)
        x = self.cross_band(x)
        return residual + x


class CleanMelMapper(nn.Module):
    def __init__(
        self,
        output_dim=80,
        hidden_dim=128,
        num_blocks=6,
        dropout=0.1,
        time_kernel_size=5,
        freq_kernel_size=5,
        mapper_mode="map",
    ):
        super().__init__()
        self.output_dim = output_dim
        self.mapper_mode = mapper_mode
        self.input_proj = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=1, bias=False),
            nn.PReLU(),
            ChannelwiseLayerNorm2d(hidden_dim),
        )
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                CleanMelBlock(
                    channels=hidden_dim,
                    time_kernel_size=time_kernel_size,
                    freq_kernel_size=freq_kernel_size,
                    dropout=dropout,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.PReLU(),
            ChannelwiseLayerNorm2d(hidden_dim),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, mel, lengths):
        x = mel.unsqueeze(1)
        x = self.input_proj(x)
        x = self.blocks(x)
        pred = self.output_proj(x).squeeze(1)
        if self.mapper_mode == "mask":
            pred = mel * torch.sigmoid(pred)
        else:
            pred = mel + pred
        return pred, lengths


class FeatureExtractor(nn.Module):
    def __init__(self, fs=16000, n_fft=512, hop_length=128, win_length=512, n_mels=80):
        super().__init__()
        self.stft = Stft(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        self.logmel = LogMel(fs=fs, n_fft=n_fft, n_mels=n_mels)

    def forward(self, wav, lengths):
        spec, spec_lens = self.stft(wav, lengths)
        spec = ComplexTensor(spec[..., 0], spec[..., 1])
        power = spec.real ** 2 + spec.imag ** 2
        return self.logmel(power, spec_lens)


def masked_regression_loss(pred, target, lengths):
    max_t = pred.size(1)
    pred = pred[:, :max_t, :]
    target = target[:, :max_t, :]
    mask = (torch.arange(max_t, device=lengths.device)[None, :] < lengths[:, None]).unsqueeze(-1)
    mask_f = mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    l1 = (torch.abs(pred - target) * mask_f).sum() / denom
    mse = (((pred - target) ** 2) * mask_f).sum() / denom
    return l1 + mse, {"l1": float(l1.detach()), "mse": float(mse.detach())}


def unwrap_state_dict(model):
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def save_loss_plot(history, outdir):
    if not history:
        return
    epochs = [x["epoch"] for x in history]
    train_losses = [x["train_loss"] for x in history]
    valid_losses = [x["valid_loss"] for x in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, valid_losses, label="valid_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("CleanMel Mapper Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "loss.png")
    plt.close()


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_wav_scp", required=True)
    parser.add_argument("--train_ref_scp", required=True)
    parser.add_argument("--valid_wav_scp", required=True)
    parser.add_argument("--valid_ref_scp", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ngpu", type=int, default=1)
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument("--num_train_utts", type=int, default=None)
    parser.add_argument("--num_valid_utts", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    set_all_seeds(conf.get("seed", 0))

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(conf, f, sort_keys=False)
    writer = SummaryWriter(log_dir=str(outdir / "tensorboard"))

    use_cuda = torch.cuda.is_available() and args.ngpu > 0
    device = torch.device("cuda" if use_cuda else "cpu")
    feat_extractor = FeatureExtractor(
        fs=conf["fs"],
        n_fft=conf["n_fft"],
        hop_length=conf["hop_length"],
        win_length=conf["win_length"],
        n_mels=conf["n_mels"],
    ).to(device)
    model = CleanMelMapper(
        output_dim=conf["n_mels"],
        hidden_dim=conf["hidden_dim"],
        num_blocks=conf["num_layers"],
        dropout=conf["dropout"],
        time_kernel_size=conf.get("time_kernel_size", 5),
        freq_kernel_size=conf.get("freq_kernel_size", 5),
        mapper_mode=conf.get("mapper_mode", "map"),
    ).to(device)
    if use_cuda and args.ngpu > 1 and torch.cuda.device_count() >= args.ngpu:
        device_ids = list(range(args.ngpu))
        feat_extractor = nn.DataParallel(feat_extractor, device_ids=device_ids)
        model = nn.DataParallel(model, device_ids=device_ids)

    train_loader = DataLoader(
        PairedWaveDataset(args.train_wav_scp, args.train_ref_scp, args.num_train_utts),
        batch_size=conf["batch_size"],
        shuffle=True,
        num_workers=conf["num_workers"],
        collate_fn=collate,
    )
    valid_loader = DataLoader(
        PairedWaveDataset(args.valid_wav_scp, args.valid_ref_scp, args.num_valid_utts),
        batch_size=conf["batch_size"],
        shuffle=False,
        num_workers=conf["num_workers"],
        collate_fn=collate,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=conf["lr"], weight_decay=conf["weight_decay"])
    best_loss = float("inf")
    history = []
    log_interval = conf.get("log_interval", 100)

    max_epoch = args.max_epoch if args.max_epoch is not None else conf["max_epoch"]
    for epoch in range(1, max_epoch + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0
        epoch_start = time.time()
        window_start = time.time()
        window_loss = 0.0
        window_forward_time = 0.0
        window_backward_time = 0.0
        window_optim_step_time = 0.0
        window_grad_norm = 0.0
        window_batches = 0
        prev_batch_idx = 1
        total_train_batches = len(train_loader)
        for batch_idx, (_, noisy, noisy_lens, clean, clean_lens) in enumerate(train_loader, 1):
            noisy = noisy.to(device)
            noisy_lens = noisy_lens.to(device)
            clean = clean.to(device)
            clean_lens = clean_lens.to(device)
            t0 = time.time()
            with torch.no_grad():
                noisy_mel, noisy_feat_lens = feat_extractor(noisy, noisy_lens)
                clean_mel, clean_feat_lens = feat_extractor(clean, clean_lens)
            pred, pred_lens = model(noisy_mel, noisy_feat_lens)
            feat_lens = torch.minimum(torch.minimum(pred_lens, noisy_feat_lens), clean_feat_lens)
            max_len = int(feat_lens.max().item())
            loss, _ = masked_regression_loss(pred[:, :max_len], clean_mel[:, :max_len], feat_lens)
            t1 = time.time()
            optimizer.zero_grad()
            loss.backward()
            t2 = time.time()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            t3 = time.time()
            train_loss += float(loss.detach())
            train_batches += 1
            window_loss += float(loss.detach())
            window_forward_time += t1 - t0
            window_backward_time += t2 - t1
            window_optim_step_time += t3 - t2
            window_grad_norm += float(grad_norm)
            window_batches += 1

            if batch_idx % log_interval == 0 or batch_idx == total_train_batches:
                train_time = time.time() - window_start
                avg_loss = window_loss / max(window_batches, 1)
                print(
                    f"{epoch}epoch:train:{prev_batch_idx}-{batch_idx}batch: "
                    f"iter_time={fmt(train_time / max(window_batches, 1))}, "
                    f"forward_time={fmt(window_forward_time / max(window_batches, 1))}, "
                    f"loss={fmt(avg_loss)}, "
                    f"backward_time={fmt(window_backward_time / max(window_batches, 1))}, "
                    f"grad_norm={fmt(window_grad_norm / max(window_batches, 1))}, "
                    f"optim_step_time={fmt(window_optim_step_time / max(window_batches, 1))}, "
                    f"optim0_lr0={fmt(optimizer.param_groups[0]['lr'])}, "
                    f"train_time={fmt(train_time)}",
                    flush=True,
                )
                prev_batch_idx = batch_idx + 1
                window_start = time.time()
                window_loss = 0.0
                window_forward_time = 0.0
                window_backward_time = 0.0
                window_optim_step_time = 0.0
                window_grad_norm = 0.0
                window_batches = 0

        model.eval()
        valid_loss = 0.0
        valid_batches = 0
        with torch.no_grad():
            for _, noisy, noisy_lens, clean, clean_lens in valid_loader:
                noisy = noisy.to(device)
                noisy_lens = noisy_lens.to(device)
                clean = clean.to(device)
                clean_lens = clean_lens.to(device)
                noisy_mel, noisy_feat_lens = feat_extractor(noisy, noisy_lens)
                clean_mel, clean_feat_lens = feat_extractor(clean, clean_lens)
                pred, pred_lens = model(noisy_mel, noisy_feat_lens)
                feat_lens = torch.minimum(torch.minimum(pred_lens, noisy_feat_lens), clean_feat_lens)
                max_len = int(feat_lens.max().item())
                loss, _ = masked_regression_loss(pred[:, :max_len], clean_mel[:, :max_len], feat_lens)
                valid_loss += float(loss.detach())
                valid_batches += 1

        train_loss /= max(train_batches, 1)
        valid_loss /= max(valid_batches, 1)
        epoch_time = time.time() - epoch_start
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/valid", valid_loss, epoch)
        if use_cuda:
            writer.add_scalar("resource/gpu_max_cached_mem_GB", torch.cuda.max_memory_reserved() / (1024 ** 3), epoch)
        if use_cuda:
            gpu_mem = torch.cuda.max_memory_reserved() / (1024 ** 3)
            print(
                f"{epoch}epoch results: [train] loss={train_loss:.4f}, total_count={train_batches}, "
                f"[valid] loss={valid_loss:.4f}, total_count={valid_batches}, "
                f"time={epoch_time:.2f} seconds, gpu_max_cached_mem_GB={gpu_mem:.3f}",
                flush=True,
            )
        else:
            print(
                f"{epoch}epoch results: [train] loss={train_loss:.4f}, total_count={train_batches}, "
                f"[valid] loss={valid_loss:.4f}, total_count={valid_batches}, time={epoch_time:.2f} seconds",
                flush=True,
            )
        torch.save({"model": unwrap_state_dict(model), "config": conf}, outdir / "latest.pth")
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save({"model": unwrap_state_dict(model), "config": conf}, outdir / "valid.loss.best.pth")

    with open(outdir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    save_loss_plot(history, outdir)
    writer.close()


if __name__ == "__main__":
    main()
