"""Put the router experiments side by side.

Usage: python compare_runs.py           (reads every log*/emoe-mosi.log: log/ is EMOE as
                                         published, log_<name>/ is a variant such as log_self/)
       python compare_runs.py --smoke   (reads the smoke-test logs instead)

Three tables, each averaged over the finished seeds:
  1. Results: model size, training cost and the selected test result (best-validation-Loss
     checkpoint). With more than one seed a cell reads mean ± std (n-1).
  2. Checks on the same best model, as Acc_2 / MAE: voice or face damaged on purpose, text
     only, and every clip given the same fixed weights (1/3 each, or the model's own average).
  3. Peak on test: each metric at its best test epoch, the way the paper reports (optimistic).
Then the router weights: on the clean test clips (WEIGHTS lines, one per seed) and on
average under each kind of damage. Saves everything to compare_table.md.
"""
import argparse
import re
import statistics
from pathlib import Path

from summarize import parse_log, summarize_run, parse_metrics, fmt, COLUMNS

ROUTERS = ('orig', 'equal', 'light', 'self')
CHECKS = [('ROBUST voice_zero', 'voice zeroed'), ('ROBUST face_zero', 'face zeroed'),
          ('ROBUST both_zero', 'text only'), ('ROBUST noise', 'noise'),
          ('FORCED equal', 'forced 1/3 each'), ('FORCED average', 'forced average')]
PARAMS_RE = re.compile(r"router: (\w+) \| parameters: total ([\d,]+), router ([\d,]+)")
COST_RE = re.compile(r"cost: training ([\d.]+) min \| peak GPU memory ([\d.]+|nan) GB")
DAMAGE_W_RE = re.compile(r"WEIGHTS ROBUST (\w+) >> text ([\d.]+) face ([\d.]+) voice ([\d.]+)")
AVG_W_RE = re.compile(r"FORCED average uses the validation-clip average: text ([\d.]+) face ([\d.]+) voice ([\d.]+)")


def find_logs(smoke):
    """Every training log here: log/ (or log_smoke/) is EMOE as published, called 'orig';
    log_<name>/ (or log_smoke_<name>/) is a variant called <name>. A folder that only adds a
    training switch to EMOE's own router, such as log_batchbal/, is called orig_<switch>."""
    prefix = 'log_smoke' if smoke else 'log'
    found = []
    for path in sorted(Path('.').glob('log*/emoe-mosi.log')):
        folder = path.parent.name
        if folder.startswith('log_smoke') != smoke:
            continue
        name = folder[len(prefix):].lstrip('_') or 'orig'
        if name.split('_')[0] not in ROUTERS:
            name = 'orig_' + name
        found.append((name, path))
    return found


def read_extra_lines(path):
    """The lines summarize.py ignores: model size, training cost, the checks (ROBUST and
    FORCED results), the WEIGHTS summaries and the average weights on damaged clips."""
    info = {'params': None, 'costs': [], 'checks': {key: [] for key, _ in CHECKS},
            'weights': [], 'damage_w': {}, 'avg_w': []}
    for line in path.read_text(encoding='utf-8').splitlines():
        m = PARAMS_RE.search(line)
        if m:
            info['params'] = (m.group(2), m.group(3))
        c = COST_RE.search(line)
        if c:
            info['costs'].append((float(c.group(1)), float(c.group(2))))
        for key, _ in CHECKS:
            if f"{key}-(" in line:
                info['checks'][key].append(parse_metrics(line))
        if "WEIGHTS-(" in line:
            info['weights'].append(line.split(">>", 1)[1].strip())
        d = DAMAGE_W_RE.search(line)
        if d:
            info['damage_w'].setdefault(d.group(1), []).append([float(x) for x in d.groups()[1:]])
        a = AVG_W_RE.search(line)
        if a:
            info['avg_w'].append([float(x) for x in a.groups()])
    return info


def cell(values, metric):
    """The mean, or mean ± std (n-1) when there is more than one seed."""
    mean = statistics.mean(values)
    if len(values) < 2:
        return fmt(metric, mean)
    return f"{fmt(metric, mean)} ± {fmt(metric, statistics.stdev(values))}"


def acc_mae(results):
    """Acc_2 / MAE averaged over the seeds, or '-' when the log has no such line."""
    if not results:
        return "-"
    return (f"{fmt('Acc_2', statistics.mean(r['Acc_2'] for r in results))} / "
            f"{fmt('MAE', statistics.mean(r['MAE'] for r in results))}")


def mean_weights(rows):
    """Average a list of [text, face, voice] weights."""
    means = [statistics.mean(r[i] for r in rows) for i in range(3)]
    return "text {:.3f} face {:.3f} voice {:.3f}".format(*means)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='compare the smoke-test logs')
    parser.add_argument('--out', default='compare_table.md')
    cli = parser.parse_args()

    results = ["| Variant | Parameters (total / router) | Train min | Peak GPU GB | Seeds | Acc_7 | Acc_2 | F1 | MAE |",
               "|" + "---|" * 9]
    checks = ["| Variant | " + " | ".join(label for _, label in CHECKS) + " |",
              "|" + "---|" * (len(CHECKS) + 1)]
    peaks = ["| Variant | Seeds | Acc_7 | Acc_2 | F1 | MAE |", "|" + "---|" * 6]
    notes, damage = [], []
    for name, path in find_logs(cli.smoke):
        finished = [r for r in (summarize_run(run) for run in parse_log(path)) if r is not None]
        if not finished:
            print(f"Variant '{name}': no finished run in {path}, skipping.")
            continue
        selected = [r[0] for r in finished]
        peak = [r[1] for r in finished]
        info = read_extra_lines(path)
        size = f"{info['params'][0]} / {info['params'][1]}" if info['params'] else "?"
        minutes = f"{statistics.mean(c[0] for c in info['costs']):.1f}" if info['costs'] else "-"
        memory = f"{max(c[1] for c in info['costs']):.2f}" if info['costs'] else "-"
        n = len(selected)
        results.append(f"| {name} | {size} | {minutes} | {memory} | {n} | "
                       + " | ".join(cell([s[c] for s in selected], c) for c in COLUMNS) + " |")
        checks.append(f"| {name} | " + " | ".join(acc_mae(info['checks'][key]) for key, _ in CHECKS) + " |")
        peaks.append(f"| {name} | {n} | " + " | ".join(cell([p[c][0] for p in peak], c) for c in COLUMNS) + " |")
        for w in info['weights']:
            notes.append(f"- {name}: {w}")
        parts = [f"{how}: {mean_weights(rows)}" for how, rows in info['damage_w'].items()]
        if info['avg_w']:
            parts.append(f"forced average used: {mean_weights(info['avg_w'])}")
        if parts:
            damage.append(f"- {name}: " + " | ".join(parts))

    text = ("Results (selected = best-validation-Loss checkpoint on test; mean ± std over seeds when there is more than one):\n\n"
            + "\n".join(results)
            + "\n\nChecks on the same best model, Acc_2 / MAE (text only = voice and face both zeroed; forced = every "
              "clip gets the same weights, 1/3 each or the model's own average on the validation clips):\n\n"
            + "\n".join(checks)
            + "\n\nPeak on test (each metric at its best test epoch, as the paper reports; optimistic):\n\n"
            + "\n".join(peaks)
            + "\n\nAcc_7, Acc_2, F1 in %, MAE on the label scale.\n\n"
              "Router weights on the clean test clips (corr_with_error: negative means bigger weights go to the "
              "modality whose own prediction is better on that clip):\n" + "\n".join(notes)
            + "\n\nAverage router weights on the damaged test clips:\n"
            + ("\n".join(damage) if damage else "- none logged") + "\n")
    print(text)
    Path(cli.out).write_text(text, encoding='utf-8')
    print(f"Saved to {cli.out}")


if __name__ == '__main__':
    main()
