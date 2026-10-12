"""Turn an EMOE training log into a Markdown results table.

Usage: python summarize.py [LOG] [--out results_table.md]

Per finished seed two rows are reported:
  selected      - the final TEST line: the best-validation-Loss checkpoint
                  evaluated on the test set (no test data used to pick it).
  peak on test  - each metric at its own best epoch on the test set.
                  Optimistic: the epoch is chosen by looking at the test set.
"""
import argparse
import re
import statistics

METRIC_RE = re.compile(r"(Acc_2|F1_score|Acc_7|MAE|Loss): (-?\d+(?:\.\d+)?|nan)")
SEED_RE = re.compile(r"seed: (\d+)")
COLUMNS = ["Acc_7", "Acc_2", "F1_score", "MAE"]
HIGHER_IS_BETTER = {"Acc_7": True, "Acc_2": True, "F1_score": True, "MAE": False}
PAPER = {"Acc_7": 0.477, "Acc_2": 0.854, "F1_score": 0.854, "MAE": 0.710}


def parse_metrics(line):
    return {k: float(v) for k, v in METRIC_RE.findall(line)}


def parse_log(path):
    """Split the log into runs, one per 'seed: n' line, keeping the
    TRAIN / VAL / TEST lines of each run in order."""
    runs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = SEED_RE.search(line)
            if m:
                runs.append({"seed": int(m.group(1)), "events": []})
            elif not runs:
                continue
            elif "TRAIN-(" in line:
                runs[-1]["events"].append(("TRAIN", parse_metrics(line)))
            elif "VAL-(" in line:
                runs[-1]["events"].append(("VAL", parse_metrics(line)))
            elif "TEST-(" in line:
                runs[-1]["events"].append(("TEST", parse_metrics(line)))
    return runs


def summarize_run(run):
    """Return (selected, peak, best_epoch) or None if the run did not finish.
    A finished run has n epochs (TRAIN, VAL, TEST each) plus one final TEST."""
    kinds = [k for k, _ in run["events"]]
    n_epochs = kinds.count("TRAIN")
    if n_epochs < 1 or kinds != ["TRAIN", "VAL", "TEST"] * n_epochs + ["TEST"]:
        return None
    val = [m for k, m in run["events"] if k == "VAL"]
    epoch_test = [m for k, m in run["events"] if k == "TEST"][:-1]
    selected = run["events"][-1][1]

    # Same rule as the trainer: first epoch with the lowest validation Loss
    losses = [m["Loss"] for m in val]
    best_epoch = losses.index(min(losses)) + 1
    if any(abs(epoch_test[best_epoch - 1][c] - selected[c]) > 1e-4 for c in COLUMNS):
        print(f"WARNING: seed {run['seed']}: final TEST differs from the TEST line of "
              f"epoch {best_epoch} (lowest validation Loss)")

    peak = {}
    for c in COLUMNS:
        values = [m[c] for m in epoch_test]
        best = max(values) if HIGHER_IS_BETTER[c] else min(values)
        peak[c] = (best, values.index(best) + 1)
    return selected, peak, best_epoch


def fmt(metric, value):
    return f"{value:.3f}" if metric == "MAE" else f"{value * 100:.1f}"


def fmt_std(metric, value):
    return f"± {value:.3f}" if metric == "MAE" else f"± {value * 100:.1f}"


def build_table(rows_selected, rows_peak):
    header = "| Row | Seed | Epoch | Acc_7 | Acc_2 | F1 | MAE |"
    lines = [header, "|---|---|---|---|---|---|---|"]
    lines.append("| Paper (reported) | | | " + " | ".join(fmt(c, PAPER[c]) for c in COLUMNS) + " |")

    for seed, sel, epoch in rows_selected:
        lines.append(f"| selected | {seed} | {epoch} | " + " | ".join(fmt(c, sel[c]) for c in COLUMNS) + " |")
    for seed, peak in rows_peak:
        cells = [f"{fmt(c, peak[c][0])} (ep {peak[c][1]})" for c in COLUMNS]
        lines.append(f"| peak on test (optimistic) | {seed} | per metric | " + " | ".join(cells) + " |")

    if len(rows_selected) > 1:
        for name, values in (
            ("selected", [{c: sel[c] for c in COLUMNS} for _, sel, _ in rows_selected]),
            ("peak on test (optimistic)", [{c: peak[c][0] for c in COLUMNS} for _, peak in rows_peak]),
        ):
            n = len(values)
            means = {c: statistics.mean(v[c] for v in values) for c in COLUMNS}
            stds = {c: statistics.stdev(v[c] for v in values) for c in COLUMNS}
            lines.append(f"| {name}: mean | {n} seeds | | " + " | ".join(fmt(c, means[c]) for c in COLUMNS) + " |")
            lines.append(f"| {name}: std (n-1) | {n} seeds | | " + " | ".join(fmt_std(c, stds[c]) for c in COLUMNS) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?", default="log/emoe-mosi.log")
    parser.add_argument("--out", default="results_table.md")
    cli = parser.parse_args()

    rows_selected, rows_peak = [], []
    for run in parse_log(cli.log):
        result = summarize_run(run)
        if result is None:
            print(f"Skipping seed {run['seed']}: run did not finish.")
            continue
        selected, peak, best_epoch = result
        rows_selected.append((run["seed"], selected, best_epoch))
        rows_peak.append((run["seed"], peak))

    seeds = [s for s, _, _ in rows_selected]
    if len(seeds) != len(set(seeds)):
        print("WARNING: some seeds appear more than once (the log is appended across runs).")

    table = build_table(rows_selected, rows_peak)
    note = ("\n\nAcc_7, Acc_2, F1 in %; MAE on the label scale. Acc_2 and F1 exclude clips labelled 0.\n"
            "selected = best-validation-Loss checkpoint on test. "
            "peak on test = each metric at its best test epoch (optimistic, chosen using the test set).\n")
    print(table + note)
    with open(cli.out, "w", encoding="utf-8") as f:
        f.write(f"Source log: {cli.log}\n\n" + table + note)
    print(f"Saved to {cli.out}")


if __name__ == "__main__":
    main()
