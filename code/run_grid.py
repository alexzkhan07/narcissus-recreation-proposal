"""Run the CIFAR-10 Figure-3 sweep and append resumable CSV rows.

The output CSV schema matches `plot_figure3.py`:

    method,poison_ratio,seed,tar_acc,asr

`poison_ratio`, `tar_acc`, and `asr` are written as percentages. The training
API expects poison ratios as fractions, so a command-line ratio of `0.5` means
0.5% of the target class and is passed to training as `0.005`.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch

from attacks.badnets import make_trigger_fn as make_badnets_trigger_fn
from attacks.blend import make_trigger_fn as make_blend_trigger_fn
from attacks.label_consistent import make_trigger_fns as make_lc_trigger_fns
from attacks.narcissus import make_trigger_fn as make_narcissus_trigger_fn
from train_eval import TrainConfig, TriggerFn, train_and_eval

CSV_FIELDS = ("method", "poison_ratio", "seed", "tar_acc", "asr")
DEFAULT_RATIOS = "0,0.5,5,10,20,30,70"
DEFAULT_SEEDS = "0,1,2"
DEFAULT_METHODS = "BadNets,Blend,Label-Consistent,Ours"


@dataclass(frozen=True)
class MethodSpec:
    name: str
    poison_trigger_fn: TriggerFn
    eval_trigger_fn: TriggerFn | None = None


def _parse_float_list(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one comma-separated number")
    return parsed


def _parse_int_list(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return parsed


def _parse_methods(value: str) -> list[str]:
    aliases = {
        "badnets": "BadNets",
        "badnets-c": "BadNets",
        "blend": "Blend",
        "blend-c": "Blend",
        "lc": "Label-Consistent",
        "label-consistent": "Label-Consistent",
        "label_consistent": "Label-Consistent",
        "ours": "Ours",
        "narcissus": "Ours",
    }
    methods = []
    for item in value.split(","):
        key = item.strip()
        if not key:
            continue
        normalized = aliases.get(key.lower())
        if normalized is None:
            valid = ", ".join(sorted(set(aliases.values())))
            raise argparse.ArgumentTypeError(f"unknown method {key!r}; choose from {valid}")
        methods.append(normalized)
    if not methods:
        raise argparse.ArgumentTypeError("expected at least one comma-separated method")
    return list(dict.fromkeys(methods))


def _existing_keys(path: Path) -> set[tuple[str, float, int]]:
    if not path.exists():
        return set()

    keys: set[tuple[str, float, int]] = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = set(CSV_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise SystemExit(f"{path} is missing CSV columns: {sorted(missing)}")
        for row in reader:
            keys.add((
                row["method"],
                round(float(row["poison_ratio"]), 10),
                int(row["seed"]),
            ))
    return keys


def _format_percent(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _build_method(name: str, args: argparse.Namespace) -> MethodSpec:
    if name == "BadNets":
        trigger_fn = make_badnets_trigger_fn(
            patch_size=args.patch_size,
            location=args.patch_location,
            color=args.patch_color,
        )
        return MethodSpec(name=name, poison_trigger_fn=trigger_fn)

    if name == "Blend":
        trigger_fn = make_blend_trigger_fn(
            alpha=args.blend_alpha,
            pattern_seed=args.blend_seed,
        )
        return MethodSpec(name=name, poison_trigger_fn=trigger_fn)

    if name == "Label-Consistent":
        poison_trigger_fn, eval_trigger_fn = make_lc_trigger_fns(
            epsilon=args.lc_epsilon,
            perturb_seed=args.lc_perturb_seed,
            patch_size=args.patch_size,
            location=args.patch_location,
            color=args.patch_color,
        )
        return MethodSpec(
            name=name,
            poison_trigger_fn=poison_trigger_fn,
            eval_trigger_fn=eval_trigger_fn,
        )

    if name == "Ours":
        trigger_fn = make_narcissus_trigger_fn(args.narcissus_trigger)
        return MethodSpec(name=name, poison_trigger_fn=trigger_fn)

    raise AssertionError(f"unhandled method {name!r}")


def _iter_runs(
    methods: Iterable[str],
    ratios: Iterable[float],
    seeds: Iterable[int],
) -> Iterable[tuple[str, float, int]]:
    for method in methods:
        for ratio in ratios:
            for seed in seeds:
                yield method, ratio, seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="./data", help="CIFAR-10 cache directory")
    parser.add_argument("--out", type=Path, required=True, help="results CSV to append")
    parser.add_argument("--methods", type=_parse_methods, default=_parse_methods(DEFAULT_METHODS))
    parser.add_argument("--ratios", type=_parse_float_list, default=_parse_float_list(DEFAULT_RATIOS))
    parser.add_argument("--seeds", type=_parse_int_list, default=_parse_int_list(DEFAULT_SEEDS))
    parser.add_argument("--target-class", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--allow-cpu", action="store_true", help="allow slow CPU runs")
    parser.add_argument("--dry-run", action="store_true", help="print pending runs without training")
    parser.add_argument("--max-runs", type=int, default=None, help="stop after N new runs")
    parser.add_argument("--patch-size", type=int, default=3)
    parser.add_argument("--patch-location", default="bottom-right")
    parser.add_argument("--patch-color", type=float, default=1.0)
    parser.add_argument("--blend-alpha", type=float, default=0.2)
    parser.add_argument("--blend-seed", type=int, default=1234)
    parser.add_argument("--lc-epsilon", type=float, default=16 / 255)
    parser.add_argument("--lc-perturb-seed", type=int, default=2024)
    parser.add_argument("--narcissus-trigger", default=None, help="optional .npy trigger path")
    parser.add_argument("--restart", action="store_true", help="overwrite --out before running")
    args = parser.parse_args()

    if args.restart and args.out.exists() and not args.dry_run:
        args.out.unlink()

    existing = set() if args.restart else _existing_keys(args.out)
    pending = [
        run for run in _iter_runs(args.methods, args.ratios, args.seeds)
        if (run[0], round(run[1], 10), run[2]) not in existing
    ]
    if args.max_runs is not None:
        pending = pending[:args.max_runs]

    print(f"results CSV : {args.out}")
    print(f"existing    : {len(existing)} completed rows")
    print(f"pending     : {len(pending)} rows")
    for method, ratio, seed in pending:
        print(f"  {method:16s} ratio={ratio:g}% seed={seed}")

    if args.dry_run or not pending:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise SystemExit("CUDA is not available; rerun with --allow-cpu for a slow local test.")
    print(f"device      : {device}")

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not args.out.exists() or args.out.stat().st_size == 0

    with args.out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()

        for i, (method_name, ratio_pct, seed) in enumerate(pending, start=1):
            spec = _build_method(method_name, args)
            print(
                f"\n[{i}/{len(pending)}] {method_name} "
                f"target={args.target_class} ratio={ratio_pct:g}% seed={seed}"
            )
            tar_acc, asr = train_and_eval(
                trigger_fn=spec.poison_trigger_fn,
                eval_trigger_fn=spec.eval_trigger_fn,
                target_class=args.target_class,
                target_class_poison_ratio=ratio_pct / 100.0,
                seed=seed,
                root=args.root,
                cfg=cfg,
                device=device,
            )
            row = {
                "method": method_name,
                "poison_ratio": _format_percent(ratio_pct),
                "seed": seed,
                "tar_acc": _format_percent(tar_acc * 100),
                "asr": _format_percent(asr * 100),
            }
            writer.writerow(row)
            f.flush()
            print(
                f"  wrote row: tar_acc={row['tar_acc']}% "
                f"asr={row['asr']}%"
            )


if __name__ == "__main__":
    main()
