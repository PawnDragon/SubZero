from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
from pathlib import Path


def load_scalars_from_run(
    run_dir,
    scalar_names=None,
    print_tags=True,
):
    """
    Load scalar data from a TensorBoard run directory.

    Args:
        run_dir (str or Path): directory containing events.out.tfevents.*
        scalar_names (list[str] or None): which scalars to load; None = load all
        print_tags (bool): whether to print available tags

    Returns:
        dict[str, pd.DataFrame]: {scalar_name: DataFrame(step, wall_time, value)}
    """
    run_dir = Path(run_dir)

    ea = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={
            event_accumulator.SCALARS: 0,  # load all
        },
    )
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    if print_tags:
        print(f"\n[Scalars in {run_dir.name}]")
        for t in tags:
            print("  -", t)

    if scalar_names is None:
        scalar_names = tags

    results = {}
    for name in scalar_names:
        if name not in tags:
            print(f"[Warning] scalar '{name}' not found, skip.")
            continue

        events = ea.Scalars(name)
        df = pd.DataFrame(
            {
                "step": [e.step for e in events],
                "wall_time": [e.wall_time for e in events],
                "value": [e.value for e in events],
            }
        )
        results[name] = df

    return results


def summarize_scalar(df, mode="max"):
    """
    Simple scalar summary for paper/debug.

    mode: "max" | "min" | "last"
    """
    if df.empty:
        return None

    if mode == "max":
        row = df.loc[df["value"].idxmax()]
    elif mode == "min":
        row = df.loc[df["value"].idxmin()]
    elif mode == "last":
        row = df.iloc[-1]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {
        "step": int(row["step"]),
        "value": float(row["value"]),
    }


if __name__ == "__main__":
    run_path = (
        "large_models/result/SST2/opt-1.3b/prompt/subzo_adamu/2026-02-08_06-51-24"
    )

    scalars = load_scalars_from_run(
        run_path,
        scalar_names=[
            "train/loss",
            "eval/accuracy",
        ],
    )

    # Example: summarize eval accuracy
    if "eval/accuracy" in scalars:
        best = summarize_scalar(scalars["eval/accuracy"], mode="max")
        last = summarize_scalar(scalars["eval/accuracy"], mode="last")

        print("\n[Eval Accuracy Summary]")
        print("Best :", best)
        print("Last :", last)
