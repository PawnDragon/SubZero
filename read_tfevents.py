from pathlib import Path
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def load_scalar(run_dir: Path, tag: str) -> pd.DataFrame:
    ea = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        raise KeyError(f"Tag '{tag}' not found. Available scalars: {tags}")

    events = ea.Scalars(tag)
    return pd.DataFrame(
        {"step": [e.step for e in events], "wall_time": [e.wall_time for e in events], "value": [e.value for e in events]}
    )


def summarize_best_and_last(df: pd.DataFrame):
    best_row = df.loc[df["value"].idxmax()]
    last_row = df.iloc[-1]
    return {
        "best_step": int(best_row["step"]),
        "best_value": float(best_row["value"]),
        "last_step": int(last_row["step"]),
        "last_value": float(last_row["value"]),
        "n_points": int(len(df)),
    }


if __name__ == "__main__":
    event_file = Path(
        "large_models/result/SST2/opt-1.3b/prompt/subzo_adamu/"
        "subzo_adamu-SST2-0-opt-1.3b-OPTIM_prompt-STEP20000-adamw-momen0.0-LR0.001-constant-ZOEPS0.01-T2000-gauss_rank8-Q1-bs16-gradAccumulation1/"
        "2026-02-08_06-51-24/"
        "events.out.tfevents.1770504684.autodl-container-e999448ba2-4300a65f.4338.0"
    ).resolve()

    run_dir = event_file.parent
    print("run_dir:", run_dir)

    ea = event_accumulator.EventAccumulator(str(run_dir))
    ea.Reload()
    print("available scalars:", ea.Tags().get("scalars", []))

    # your real tags
    val_tag = "accuracy/val"
    test_tag = "accuracy/test"
    loss_tag = "train_loss"

    val_df = load_scalar(run_dir, val_tag)
    test_df = load_scalar(run_dir, test_tag)
    loss_df = load_scalar(run_dir, loss_tag)

    print("\n[accuracy/val]", summarize_best_and_last(val_df))
    print("[accuracy/test]", summarize_best_and_last(test_df))

    # for loss, usually care about minimum + last
    min_row = loss_df.loc[loss_df["value"].idxmin()]
    last_row = loss_df.iloc[-1]
    print("\n[train_loss]", {
        "min_step": int(min_row["step"]),
        "min_value": float(min_row["value"]),
        "last_step": int(last_row["step"]),
        "last_value": float(last_row["value"]),
        "n_points": int(len(loss_df)),
    })
