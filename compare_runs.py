"""Put the router experiments side by side in one table.

Usage: python compare_runs.py           (reads every log*/emoe-mosi.log: log/ is EMOE as
                                         published, log_<name>/ is a variant such as log_self/)
       python compare_runs.py --smoke   (reads the smoke-test logs instead)

For each variant it reports:
  - model size: total parameters and the router's share
  - the selected test result (best-validation-Loss checkpoint), averaged over seeds
  - Acc_2 and MAE when the voice or face is damaged on purpose (ROBUST lines)
  - the WEIGHTS line: what weights the router gave the test clips
Saves the table to compare_table.md.
"""
import argparse
import re
import statistics
from pathlib import Path

from summarize import parse_log, summarize_run, parse_metrics, fmt

CORRUPTIONS = ['voice_zero', 'face_zero', 'noise']
PARAMS_RE = re.compile(r"router: (\w+) \| parameters: total ([\d,]+), router ([\d,]+)")
COST_RE = re.compile(r"cost: training ([\d.]+) min \| peak GPU memory ([\d.]+|nan) GB")


def find_logs(smoke):
    """Every training log here: log/ (or log_smoke/) is EMOE as published, called 'orig';
    log_<name>/ (or log_smoke_<name>/) is a variant called <name>."""
    prefix = 'log_smoke' if smoke else 'log'
    found = []
    for path in sorted(Path('.').glob('log*/emoe-mosi.log')):
        folder = path.parent.name
        if folder.startswith('log_smoke') != smoke:
            continue
        name = folder[len(prefix):].lstrip('_') or 'orig'
        found.append((name, path))
    return found


def read_extra_lines(path):
    """Collect the lines summarize.py ignores: model size, training cost, ROBUST results, WEIGHTS summary."""
    params, weights, costs = None, [], []
    robust = {how: [] for how in CORRUPTIONS}
    for line in path.read_text(encoding='utf-8').splitlines():
        m = PARAMS_RE.search(line)
        if m:
            params = (m.group(2), m.group(3))
        c = COST_RE.search(line)
        if c:
            costs.append((float(c.group(1)), float(c.group(2))))
        for how in CORRUPTIONS:
            if f"ROBUST {how}-(" in line:
                robust[how].append(parse_metrics(line))
        if "WEIGHTS-(" in line:
            weights.append(line.split(">>", 1)[1].strip())
    return params, robust, weights, costs


def mean_of(results, metric):
    return statistics.mean(r[metric] for r in results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='compare the smoke-test logs')
    parser.add_argument('--out', default='compare_table.md')
    cli = parser.parse_args()

    header = ("| Variant | Parameters (total / router) | Train min | Peak GPU GB | Seeds | Acc_7 | Acc_2 | F1 | MAE "
              "| Acc_2 voice zeroed | Acc_2 face zeroed | Acc_2 noise | MAE noise |")
    lines = [header, "|" + "---|" * 13]
    notes = []
    for router, path in find_logs(cli.smoke):
        finished = [summarize_run(run) for run in parse_log(path)]
        selected = [result[0] for result in finished if result is not None]
        if not selected:
            print(f"Router '{router}': no finished run in {path}, skipping.")
            continue
        params, robust, weights, costs = read_extra_lines(path)
        size = f"{params[0]} / {params[1]}" if params else "?"
        minutes = f"{statistics.mean(c[0] for c in costs):.1f}" if costs else "-"
        memory = f"{max(c[1] for c in costs):.2f}" if costs else "-"
        cells = [fmt(c, mean_of(selected, c)) for c in ['Acc_7', 'Acc_2', 'F1_score', 'MAE']]
        for how in CORRUPTIONS:
            cells.append(fmt('Acc_2', mean_of(robust[how], 'Acc_2')) if robust[how] else "-")
        cells.append(fmt('MAE', mean_of(robust['noise'], 'MAE')) if robust['noise'] else "-")
        lines.append(f"| {router} | {size} | {minutes} | {memory} | {len(selected)} | " + " | ".join(cells) + " |")
        for w in weights:
            notes.append(f"- {router}: {w}")

    table = "\n".join(lines)
    text = (table + "\n\nAcc_7, Acc_2, F1 in %, MAE on the label scale; averaged over the finished seeds.\n"
            "Robustness columns: the same best model tested with the voice or face damaged on purpose.\n\n"
            "Router weights on the test clips (corr_with_error: negative means bigger weights go to the "
            "modality whose own prediction is better on that clip):\n" + "\n".join(notes) + "\n")
    print(text)
    Path(cli.out).write_text(text, encoding='utf-8')
    print(f"Saved to {cli.out}")


if __name__ == '__main__':
    main()
