from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


DEFAULT_LOSS_TAG = "train_loss"
FALLBACK_LOSS_TAGS = ["loss/epoch"]
DEFAULT_ACC_TAGS = ["accuracy/val", "accuracy/test"]


def resolve_path(p: str) -> Path:
    x = Path(p).expanduser()
    if not x.is_absolute():
        x = (Path.cwd() / x)
    return x.resolve()


def get_run_dir(path: Path) -> Path:
    return path.parent if path.is_file() else path


def load_event_accumulator(run_dir: Path) -> event_accumulator.EventAccumulator:
    ea = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={event_accumulator.SCALARS: 0},  # load all scalars
    )
    ea.Reload()
    return ea


def load_scalar_df(ea: event_accumulator.EventAccumulator, tag: str) -> pd.DataFrame:
    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        raise KeyError(f"Scalar tag '{tag}' not found. Available: {tags}")

    events = ea.Scalars(tag)
    return pd.DataFrame(
        {
            "step": [e.step for e in events],
            "wall_time": [e.wall_time for e in events],
            "value": [e.value for e in events],
        }
    )


def plot_loss(df: pd.DataFrame, tag: str, out_path: Path, title: str | None = None):
    plt.figure()
    plt.plot(df["step"], df["value"], label=tag)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(title or tag)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_accuracy(dfs: dict[str, pd.DataFrame], out_path: Path, title: str | None = None):
    plt.figure()
    for tag, df in dfs.items():
        plt.plot(df["step"], df["value"], label=tag)
    plt.xlabel("step")
    plt.ylabel("accuracy")
    plt.title(title or "accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "path",
        help="Path to events.out.tfevents.* file OR the run directory containing it",
    )
    ap.add_argument(
        "--loss_tag",
        default=DEFAULT_LOSS_TAG,
        help=f"Loss scalar tag (default: {DEFAULT_LOSS_TAG})",
    )
    ap.add_argument(
        "--loss_fallbacks",
        nargs="*",
        default=FALLBACK_LOSS_TAGS,
        help="Fallback loss tags to try if --loss_tag not found",
    )
    ap.add_argument(
        "--acc_tags",
        nargs="*",
        default=DEFAULT_ACC_TAGS,
        help=f"Accuracy scalar tags (default: {' '.join(DEFAULT_ACC_TAGS)})",
    )
    ap.add_argument(
        "--print_tags",
        action="store_true",
        help="Print available scalar tags",
    )
    args = ap.parse_args()

    p = resolve_path(args.path)
    run_dir = get_run_dir(p)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    ea = load_event_accumulator(run_dir)
    available = ea.Tags().get("scalars", [])

    if args.print_tags:
        print(f"[run_dir] {run_dir}")
        print("[available scalars]")
        for t in available:
            print("  -", t)

    # ---- loss ----
    loss_png = run_dir / "loss_curve.png"
    try:
        loss_df = load_scalar_df(ea, args.loss_tag)
        if loss_df.empty:
            print(f"[warn] '{args.loss_tag}' is empty, skip loss plot.")
        else:
            plot_loss(loss_df, args.loss_tag, loss_png, title=args.loss_tag)
            print(f"[ok] saved: {loss_png}")
    except KeyError as e:
        print(f"[warn] {e}. Try fallbacks.")
        loss_df = None
        for tag in args.loss_fallbacks:
            if tag not in available:
                continue
            df = load_scalar_df(ea, tag)
            if df.empty:
                continue
            loss_df = df
            plot_loss(loss_df, tag, loss_png, title=tag)
            print(f"[ok] saved: {loss_png}")
            break
        if loss_df is None:
            print("[warn] no loss curve plotted (no valid loss tags).")

    # ---- accuracy (val/test etc) ----
    acc_png = run_dir / "acc_curve.png"
    acc_dfs = {}
    for tag in args.acc_tags:
        if tag not in available:
            print(f"[warn] accuracy tag '{tag}' not found, skip.")
            continue
        df = load_scalar_df(ea, tag)
        if df.empty:
            print(f"[warn] accuracy tag '{tag}' is empty, skip.")
            continue
        acc_dfs[tag] = df

    if acc_dfs:
        plot_accuracy(acc_dfs, acc_png, title="accuracy")
        print(f"[ok] saved: {acc_png}")
    else:
        print("[warn] no accuracy curves plotted (no valid acc tags).")


if __name__ == "__main__":
    main()
