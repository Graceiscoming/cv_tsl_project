"""
SPOTER on TSL-ONE-S (Thai Sign Language, 184 glosses): train + evaluate in one script.
Optimised for a CUDA GPU (batched training, data kept on GPU, bf16), falls back to CPU.

Usage:
  python spoter_tslone.py --epochs 100 --out out
  python spoter_tslone.py --resume --epochs 100 --out out      # continue after Ctrl+C / crash / sleep
  python spoter_tslone.py --eval_only --out out                # only metrics + plots from out/best.pt
  python spoter_tslone.py --epochs 1 --limit_train 200 --out test_out   # 1-minute smoke test

Expected folder layout (after the setup commands):
  tslone/tsl_one_s_dataset.json   pose/TSL-ONE-Pose/*.npy   spoter/spoter/spoter_model.py

Progress files written to --out:
  status.json     what the script is doing right now (stage, epoch, batch, ETA, last update time)
  train_log.csv   one row per epoch
  last.pt         checkpoint of the latest epoch (for --resume)       best.pt   best validation epoch
"""
import argparse, csv, json, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn

BODY = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16]  # MediaPipe: nose, eyes, ears, shoulders, elbows, wrists


# ------------------------------------------------------------------ small utilities
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Status:
    """Writes out/status.json (at most every 2 s) so you can always see whether the run is alive."""

    def __init__(self, path):
        self.path, self.last, self.d = path, 0.0, {}

    def set(self, force=False, **kw):
        self.d.update(kw)
        now = time.time()
        if force or now - self.last > 2:
            self.d["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.d, f, indent=1)
            os.replace(tmp, self.path)
            self.last = now


class Progress:
    """One-line progress bar (updates in place on a terminal, or every 10% when output is redirected)."""

    def __init__(self, total, prefix):
        self.total, self.prefix, self.t0, self.last = total, prefix, time.time(), 0.0
        self.tty, self.step = sys.stdout.isatty(), max(1, total // 10)

    def update(self, i, extra=""):
        now, done = time.time(), i == self.total
        el = now - self.t0
        eta = el / i * (self.total - i) if i else 0
        line = f"{self.prefix} {i}/{self.total} ({100*i/self.total:3.0f}%) {el:5.0f}s elapsed, ETA {eta:4.0f}s {extra}"
        if self.tty:
            if now - self.last > 0.5 or done:
                sys.stdout.write("\r" + line + " " * 4 + ("\n" if done else "")); sys.stdout.flush(); self.last = now
        elif i % self.step == 0 or done:
            print(line, flush=True)


def fmt_min(sec):
    return f"{int(sec // 3600)}h{int(sec % 3600 // 60):02d}m" if sec >= 3600 else f"{sec/60:.1f}m"


# ------------------------------------------------------------------ data
def load_clip(pose_dir, vid):
    # file = (T, 150) = 75 MediaPipe landmarks (33 pose + 21 + 21 hands) x (x, y)
    a = np.load(f"{pose_dir}/{vid}.npy").reshape(-1, 75, 2).astype(np.float32)
    neck = (a[:, 11] + a[:, 12]) / 2
    j = np.concatenate([a[:, BODY[:1]], neck[:, None], a[:, BODY[1:]], a[:, 33:75]], axis=1)  # (T, 54, 2)
    scale = np.linalg.norm(a[:, 11] - a[:, 12], axis=-1).mean() + 1e-6
    return ((j - neck[:, None]) / (2 * scale)).astype(np.float32)  # centre on neck, scale by shoulder width


def load_data(args, dev):
    meta = json.load(open(args.meta))
    glosses = [g["gloss"] for g in meta]
    items = {"train": [], "val": [], "test": []}
    skipped = []
    for ci, g in enumerate(meta):
        for ins in g["instances"]:
            if not os.path.exists(f"{args.pose_dir}/{ins['video_id']}.npy"):
                skipped.append(ins["video_id"]); continue  # 4 ids in the JSON have no matching .npy
            items[ins["split"]].append((ins["video_id"], ci))
    log(f"clips without a .npy file (skipped): {skipped}")
    if args.limit_train:
        random.shuffle(items["train"]); items["train"] = items["train"][:args.limit_train]
    out = {}
    for name, lst in items.items():
        prog = Progress(len(lst), f"loading {name}")
        clips = []
        for i, (v, c) in enumerate(lst, 1):
            clips.append(load_clip(args.pose_dir, v)); prog.update(i)
        lens = torch.tensor([len(c) for c in clips])
        X = torch.zeros(len(clips), int(lens.max()), 54, 2)
        for i, c in enumerate(clips):
            X[i, :len(c)] = torch.from_numpy(c)
        out[name] = dict(X=X.to(dev), L=lens.to(dev), y=torch.tensor([c for _, c in lst]).to(dev),
                         vids=[v for v, _ in lst])
        log(f"{name}: {len(lst)} clips, longest {int(lens.max())} frames, {X.numel()*4/1e6:.0f} MB on {dev}")
    return out, glosses


# ------------------------------------------------------------------ model
class SimpleDecoder(nn.Module):
    """New PyTorch's TransformerDecoder reads layers[0].self_attn, which SPOTER deletes.
    Looping over SPOTER's layers directly is the same computation as the original."""

    def __init__(self, layers, norm):
        super().__init__(); self.layers, self.norm = layers, norm

    def forward(self, tgt, memory, *a, memory_key_padding_mask=None, **k):
        for layer in self.layers:
            tgt = layer(tgt, memory, memory_key_padding_mask=memory_key_padding_mask)
        return self.norm(tgt) if self.norm is not None else tgt


class GRUClassifier(nn.Module):
    """Baseline: 2-layer bidirectional GRU over the same 54x2 pose sequence, masked mean+max pooling, linear head."""
    def __init__(self, num_classes, in_dim=108, hidden=256, layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * hidden, num_classes)

    def forward(self, x, pad_mask):
        x = x.flatten(2).float()
        lens = (~pad_mask).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        valid = (~pad_mask).unsqueeze(-1)
        mean = (out * valid).sum(1) / valid.sum(1).clamp(min=1)
        mx = out.masked_fill(~valid, -1e4).max(1).values
        return self.fc(self.drop(torch.cat([mean, mx], -1)))


def build_model(spoter_dir, num_classes, kind="spoter"):
    if kind == "gru":
        return GRUClassifier(num_classes)
    sys.path.insert(0, spoter_dir)
    from spoter_model import SPOTER
    m = SPOTER(num_classes=num_classes, hidden_dim=108)
    m.transformer.decoder = SimpleDecoder(m.transformer.decoder.layers, m.transformer.decoder.norm)
    return m


def forward_batch(model, x, pad_mask):
    """Batched SPOTER forward. x: (B,T,54,2), pad_mask: (B,T) True on padded frames -> logits (B,num_classes).
    Same computation as SPOTER.forward on one clip, but padded frames are masked out of attention."""
    if isinstance(model, GRUClassifier):
        return model(x, pad_mask)
    B = x.shape[0]
    src = (model.pos + x.flatten(2)).transpose(0, 1)                    # (T,B,108)
    tgt = model.class_query.unsqueeze(0).expand(1, B, -1)               # (1,B,108)
    memory = model.transformer.encoder(src, src_key_padding_mask=pad_mask)
    out = model.transformer.decoder(tgt, memory, memory_key_padding_mask=pad_mask)
    return model.linear_class(out.transpose(0, 1)).squeeze(1)


def get_batch(D, idx):
    lens = D["L"][idx]
    T = int(lens.max())
    return D["X"][idx, :T], torch.arange(T, device=lens.device)[None] >= lens[:, None], D["y"][idx]


def augment(x):
    """Per-clip random rotation (+-10 deg), scale (0.9-1.1) and Gaussian noise, all on the GPU."""
    B, dev = x.shape[0], x.device
    th = (torch.rand(B, device=dev) * 2 - 1) * (10 * np.pi / 180)
    s = 0.9 + 0.2 * torch.rand(B, device=dev)
    R = torch.stack([torch.stack([th.cos(), -th.sin()], -1), torch.stack([th.sin(), th.cos()], -1)], -2)  # (B,2,2)
    x = torch.einsum("btjc,bdc->btjd", x, R) * s[:, None, None, None]
    return x + 0.01 * torch.randn_like(x)


@torch.no_grad()
def predict(model, D, dev, amp, bs=128):
    model.eval(); P, loss_sum = [], 0.0
    for i in range(0, len(D["y"]), bs):
        x, m, y = get_batch(D, torch.arange(i, min(i + bs, len(D["y"])), device=dev))
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
            lg = forward_batch(model, x, m)
        lg = lg.float()
        loss_sum += nn.functional.cross_entropy(lg, y, reduction="sum").item()
        P.append(torch.softmax(lg, -1).cpu())
    return torch.cat(P).numpy(), loss_sum / len(D["y"])


# ------------------------------------------------------------------ training
def train(args, model, data, dev, amp, status):
    D = data["train"]
    N = len(D["y"])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=8)
    crit = nn.CrossEntropyLoss()
    hist, best, start_ep, elapsed_prev, bad, step = [], -1.0, 1, 0.0, 0, 0
    ck = f"{args.out}/last.pt"
    if args.resume and os.path.exists(ck):
        s = torch.load(ck, map_location=dev)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        hist, best, start_ep, elapsed_prev, bad, step = s["hist"], s["best"], s["epoch"] + 1, s["elapsed"], s["bad"], s["step"]
        log(f"RESUMED from epoch {s['epoch']} (best val acc so far {best:.4f})")
    elif args.resume:
        log("--resume given but no last.pt found, starting from scratch")
    nb = (N + args.batch_size - 1) // args.batch_size
    log(f"training: {N} clips, batch {args.batch_size} -> {nb} steps/epoch, lr {args.lr}, bf16={amp}, epochs {start_ep}..{args.epochs}")
    t0 = time.time()
    logf = open(f"{args.out}/train_log.csv", "a", newline="")
    lw = csv.writer(logf)
    if logf.tell() == 0:
        lw.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_sec", "gpu_peak_mb"])

    def save_ck(ep):
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), hist=hist, best=best,
                        epoch=ep, elapsed=elapsed_prev + time.time() - t0, bad=bad, step=step), ck)

    ep = start_ep - 1
    try:
        for ep in range(start_ep, args.epochs + 1):
            te = time.time(); model.train()
            perm = torch.randperm(N, device=dev)
            tl_sum = torch.zeros((), device=dev); correct = torch.zeros((), device=dev)
            prog = Progress(nb, f"epoch {ep:3d}/{args.epochs}")
            for b in range(nb):
                x, m, y = get_batch(D, perm[b * args.batch_size:(b + 1) * args.batch_size])
                step += 1
                if step <= args.warmup:  # linear LR warm-up
                    for g in opt.param_groups: g["lr"] = args.lr * step / args.warmup
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
                    lg = forward_batch(model, augment(x), m)
                loss = crit(lg.float(), y)
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
                tl_sum += loss.detach() * len(y); correct += (lg.argmax(-1) == y).sum()
                if b == 0 or (b + 1) % 5 == 0 or b + 1 == nb:  # .item() syncs the GPU, so only occasionally
                    lv = loss.item()
                    if not np.isfinite(lv):
                        raise RuntimeError("loss became NaN/inf - stop and re-run with --lr 1e-4 or --no_amp")
                    prog.update(b + 1, f"loss {lv:.3f}")
                    status.set(stage="training", epoch=ep, epochs=args.epochs, batch=b + 1, batches=nb, loss=round(lv, 4))
                if ep == start_ep and b == 0:
                    log(f"first batch OK ({time.time()-te:.1f}s incl. warm-up) - training is running")
            tr_loss, tr_acc = tl_sum.item() / N, correct.item() / N
            status.set(stage="validating", epoch=ep, force=True)
            P, vl = predict(model, data["val"], dev, amp)
            va = float((P.argmax(1) == data["val"]["y"].cpu().numpy()).mean())
            sched.step(va)
            sec = time.time() - te
            peak = torch.cuda.max_memory_allocated() / 1e6 if dev.type == "cuda" else 0
            hist.append((ep, tr_loss, va)); improved = va > best
            if improved:
                best, bad = va, 0; torch.save(model.state_dict(), f"{args.out}/best.pt")
            else:
                bad += 1
            lw.writerow([ep, f"{tr_loss:.4f}", f"{tr_acc:.4f}", f"{vl:.4f}", f"{va:.4f}", f"{opt.param_groups[0]['lr']:.2e}",
                         f"{sec:.1f}", f"{peak:.0f}"]); logf.flush()
            run_el = elapsed_prev + time.time() - t0
            eta = sec * (args.epochs - ep)
            log(f"epoch {ep:3d}/{args.epochs} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | val loss {vl:.3f} acc {va:.3f}"
                f"{' *best*' if improved else ''} | {sec:.0f}s/epoch | ETA {fmt_min(eta)} | GPU peak {peak:.0f} MB")
            save_ck(ep)
            json.dump({"hist": hist, "train_minutes": run_el / 60}, open(f"{args.out}/history.json", "w"))
            status.set(stage="epoch done", epoch=ep, val_acc=round(va, 4), best_val_acc=round(best, 4), eta=fmt_min(eta), force=True)
            if args.patience and bad >= args.patience:
                log(f"early stop: no val improvement for {args.patience} epochs"); break
    except KeyboardInterrupt:
        log(f"interrupted - checkpoint of the last finished epoch is saved. Continue with:  --resume")
        status.set(stage="interrupted", force=True)
        raise SystemExit(1)
    finally:
        logf.close()
    json.dump({"hist": hist, "train_minutes": (elapsed_prev + time.time() - t0) / 60}, open(f"{args.out}/history.json", "w"))


# ------------------------------------------------------------------ evaluation (table + figures for the slides)
def evaluate(args, model, data, glosses, dev, amp, status):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, roc_auc_score,
                                 confusion_matrix, roc_curve, auc)
    status.set(stage="evaluating on test set", force=True); log("evaluating best.pt on the test set ...")
    NC = len(glosses)
    model.load_state_dict(torch.load(f"{args.out}/best.pt", map_location=dev))
    P, _ = predict(model, data["test"], dev, amp)
    y = data["test"]["y"].cpu().numpy(); pred = P.argmax(1); vids = data["test"]["vids"]

    res = {"n_test": int(len(y)), "n_classes_in_test": int(len(set(y)))}
    res["accuracy_top1"] = float(accuracy_score(y, pred))
    res["accuracy_top5"] = float(np.mean([y[i] in np.argsort(-P[i])[:5] for i in range(len(y))]))
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y, pred, average=avg, zero_division=0)
        res[f"precision_{avg}"], res[f"recall_{avg}"], res[f"f1_{avg}"] = float(p), float(r), float(f)
    Y = np.zeros_like(P); Y[np.arange(len(y)), y] = 1  # one-vs-rest ROC/AUC over classes present in test
    cls = [c for c in range(NC) if 0 < Y[:, c].sum() < len(y)]
    res["auc_macro_ovr"] = float(np.mean([roc_auc_score(Y[:, c], P[:, c]) for c in cls]))
    fpr_m, tpr_m, _ = roc_curve(Y[:, cls].ravel(), P[:, cls].ravel())
    res["auc_micro"] = float(auc(fpr_m, tpr_m))
    hist = json.load(open(f"{args.out}/history.json"))
    res["train_minutes"] = hist.get("train_minutes")
    json.dump(res, open(f"{args.out}/metrics.json", "w"), indent=2)
    log("test metrics:\n" + json.dumps(res, indent=2))

    grid = np.linspace(0, 1, 200)
    mean_tpr = np.mean([np.interp(grid, *roc_curve(Y[:, c], P[:, c])[:2]) for c in cls], axis=0)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr_m, tpr_m, label=f"micro-average (AUC={res['auc_micro']:.3f})")
    plt.plot(grid, mean_tpr, label=f"macro-average (AUC={res['auc_macro_ovr']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title(f"ROC Curve ({args.model.upper()}, TSL-ONE-S test)")
    plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(f"{args.out}/roc_curve.png", dpi=200); plt.close()

    cm = confusion_matrix(y, pred, labels=range(NC))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    plt.figure(figsize=(9, 8)); plt.imshow(cmn, cmap="Blues", vmin=0, vmax=1); plt.colorbar(label="row-normalised")
    plt.xlabel("Predicted class id"); plt.ylabel("True class id"); plt.title("Confusion Matrix (184 classes)")
    plt.tight_layout(); plt.savefig(f"{args.out}/confusion_matrix.png", dpi=200); plt.close()
    off = sorted(((cm[i, j], glosses[i], glosses[j]) for i in range(NC) for j in range(NC) if i != j and cm[i, j] > 0), reverse=True)
    with open(f"{args.out}/top_confusions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["count", "true", "predicted"]); w.writerows(off[:15])

    with open(f"{args.out}/test_predictions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["video_id", "true", "pred", "confidence", "correct"])
        for v, t, p, pr in zip(vids, y, pred, P.max(1)):
            w.writerow([v, glosses[t], glosses[p], f"{pr:.3f}", int(t == p)])

    plt.figure(figsize=(9, 3.5)); e = [h[0] for h in hist["hist"]]
    plt.subplot(1, 2, 1); plt.plot(e, [h[1] for h in hist["hist"]]); plt.xlabel("epoch"); plt.title("Train loss")
    plt.subplot(1, 2, 2); plt.plot(e, [h[2] for h in hist["hist"]]); plt.xlabel("epoch"); plt.title("Val accuracy")
    plt.tight_layout(); plt.savefig(f"{args.out}/training_curves.png", dpi=200); plt.close()
    status.set(stage="finished", force=True); log(f"ALL DONE -> {args.out}/")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="tslone/tsl_one_s_dataset.json")
    ap.add_argument("--pose_dir", default="pose/TSL-ONE-Pose")
    ap.add_argument("--spoter_dir", default="spoter/spoter")
    ap.add_argument("--out", default="out")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=300, help="LR warm-up steps")
    ap.add_argument("--patience", type=int, default=25, help="early stop after N epochs without val gain (0 = off)")
    ap.add_argument("--no_amp", action="store_true", help="disable bf16 mixed precision")
    ap.add_argument("--limit_train", type=int, default=0, help="debug: use only N training clips")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", choices=["spoter", "gru"], default="spoter")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    status = Status(f"{args.out}/status.json")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda" and torch.cuda.is_bf16_supported() and not args.no_amp
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        log(f"device: cuda - {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB VRAM")
    else:
        torch.set_num_threads(os.cpu_count() or 4)
        log("device: CPU  (WARNING: no CUDA GPU found - this will be slow; install the CUDA build of PyTorch)")

    status.set(stage="loading data", force=True)
    data, glosses = load_data(args, dev)
    model = build_model(args.spoter_dir, len(glosses), args.model).to(dev)
    log(f"model: {args.model.upper()}, {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters, {len(glosses)} classes")
    if not args.eval_only:
        train(args, model, data, dev, amp, status)
    evaluate(args, model, data, glosses, dev, amp, status)


if __name__ == "__main__":
    main()
