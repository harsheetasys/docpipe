"""Train the text-region U-Net segmenter with MLflow tracking + registry.

Masks are derived from the line boxes in the lake (every pixel inside a text-line box is
"text"). Images are resized to 256x512 (receipts are tall), grayscale.

Usage: python -m docpipe.train.train_segmenter --epochs 6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from docpipe.config import CHALLENGER_ALIAS, LAKE_DIR, RAW_DIR, SEED, SEGMENTER_MODEL_NAME
from docpipe.models.unet import UNet, dice_loss, iou_score
from docpipe.tracking import log_environment, setup_mlflow

SIZE = (160, 320)  # (W, H) - receipts are tall; this keeps CPU training to minutes


class ReceiptMaskDataset(Dataset):
    def __init__(self, split: str, lake_dir: Path = LAKE_DIR, raw_dir: Path = RAW_DIR, limit: int | None = None):
        con = duckdb.connect()
        self.docs = con.execute(
            f"SELECT doc_id, width, height FROM read_parquet('{lake_dir / 'docs.parquet'}/*.parquet') WHERE split=? ORDER BY doc_id",
            [split],
        ).fetchall()[:limit]
        self.boxes = {}
        for doc_id, x1, y1, x2, y2 in con.execute(
            f"SELECT doc_id, x1, y1, x2, y2 FROM read_parquet('{lake_dir / 'lines.parquet'}/*.parquet')"
        ).fetchall():
            self.boxes.setdefault(doc_id, []).append((x1, y1, x2, y2))
        self.raw_dir = raw_dir
        # pre-render everything once: ~0.2 MB per doc, avoids re-drawing masks every epoch
        self.items = [self._load(i) for i in range(len(self.docs))]

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, i):
        return self.items[i]

    def _load(self, i):
        doc_id, w, h = self.docs[i]
        img = Image.open(self.raw_dir / "img" / f"{doc_id}.jpg").convert("L").resize(SIZE, Image.BILINEAR)
        mask = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(mask)
        for b in self.boxes.get(doc_id, []):
            d.rectangle(b, fill=255)
        mask = mask.resize(SIZE, Image.NEAREST)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)[None]
        y = (torch.from_numpy(np.asarray(mask, dtype=np.float32)) > 127).float()[None]
        return x, y


def evaluate(model, loader, device="cpu") -> dict:
    model.eval()
    ious, losses, confs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            lg = model(x)
            losses.append(float(dice_loss(lg, y)))
            ious.append(iou_score(lg, y))
            confs.append(float(torch.sigmoid(lg).sub(0.5).abs().mean() * 2))  # 1 = confident everywhere
    return {"dice_loss": float(np.mean(losses)), "iou": float(np.mean(ious)), "mean_confidence": float(np.mean(confs))}


def train(epochs: int = 6, batch: int = 8, lr: float = 2e-3, base: int = 16, limit: int | None = None, register: bool = True) -> dict:
    torch.manual_seed(SEED)
    setup_mlflow()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr = DataLoader(ReceiptMaskDataset("train", limit=limit), batch_size=batch, shuffle=True, num_workers=0)
    va = DataLoader(ReceiptMaskDataset("val", limit=limit), batch_size=batch, num_workers=0)
    te = DataLoader(ReceiptMaskDataset("test", limit=limit), batch_size=batch, num_workers=0)

    model = UNet(base=base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(tr))
    bce = torch.nn.BCEWithLogitsLoss()

    with mlflow.start_run(run_name="segmenter-unet") as run:
        log_environment()
        mlflow.set_tags({"model": "segmenter", "arch": f"unet-base{base}"})
        mlflow.log_params({"epochs": epochs, "batch": batch, "lr": lr, "base_channels": base, "input_size": f"{SIZE[0]}x{SIZE[1]}",
                           "loss": "bce+dice", "n_train": len(tr.dataset), "n_val": len(va.dataset), "n_test": len(te.dataset), "seed": SEED})
        best_iou, best_state = -1.0, None
        for ep in range(epochs):
            model.train()
            tot = 0.0
            for x, y in tr:
                x, y = x.to(device), y.to(device)
                lg = model(x)
                loss = bce(lg, y) + dice_loss(lg, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                tot += float(loss.detach()) * len(x)
            val = evaluate(model, va, device)
            mlflow.log_metrics({"train_loss": tot / len(tr.dataset), "val_iou": val["iou"], "val_dice_loss": val["dice_loss"]}, step=ep)
            print(f"epoch {ep+1}/{epochs} train_loss={tot/len(tr.dataset):.4f} val_iou={val['iou']:.4f}", flush=True)
            if val["iou"] > best_iou:
                best_iou, best_state = val["iou"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        test = evaluate(model, te, device)
        final = {"test_iou": test["iou"], "test_dice_loss": test["dice_loss"], "test_mean_confidence": test["mean_confidence"], "best_val_iou": best_iou}
        mlflow.log_metrics(final)

        example = torch.zeros(1, 1, SIZE[1], SIZE[0])
        mlflow.pytorch.log_model(model.cpu(), name="model", input_example=example.numpy(), pip_requirements=["torch", "numpy", "pillow"])
        version = None
        if register:
            mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", SEGMENTER_MODEL_NAME)
            client = mlflow.MlflowClient()
            client.set_registered_model_alias(SEGMENTER_MODEL_NAME, CHALLENGER_ALIAS, mv.version)
            client.set_model_version_tag(SEGMENTER_MODEL_NAME, mv.version, "test_iou", f"{test['iou']:.4f}")
            version = mv.version
        out = {"run_id": run.info.run_id, "model_version": version, **final}
        print(json.dumps(out, indent=2))
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--limit", type=int, default=None, help="debug: cap docs per split")
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args()
    train(epochs=a.epochs, batch=a.batch, lr=a.lr, limit=a.limit, register=not a.no_register)
