#!/usr/bin/env python3
"""Render all retained quality comparisons from the published performance CSV.

Optional plotting dependency: matplotlib==3.11.2. No models or network needed.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'qwen3-coder-30b': 'Qwen3-Coder 30B', 'deepcoder-14b': 'DeepCoder 14B',
         'deepseek-r1-32b': 'DeepSeek-R1 32B', 'reader-lm-1.5b': 'Reader-LM 1.5B',
         'devstral-24b': 'Devstral 24B'}
ORDER = list(NAMES)
COLORS = {1: '#606870', 2: '#0072B2', 3: '#D55E00', 4: '#009E73'}
LABELS = {1: 'Original', 2: '2 loops', 3: '3 loops', 4: '4 loops'}
STUDIES = [
    ('qwen-lm-eval-v1', 'test', 'qwen-quality-performance',
     '100-question comparison: correct answers and time cost',
     'Same 100 questions per profile | 768-token output limit | One run per profile',
     'No accuracy change was significant after correction. Reader-LM: math/logic is outside its intended domain.\n'
     'Times sum measured requests; loading and warm-up excluded. Output lengths vary. Time axes differ by model.'),
    ('looplab-diagnostic-v2', 'selection', 'diagnostic-selection-performance',
     'Earlier diagnostic: candidate-selection questions',
     '12 selection questions per profile | Original runtime | Separate from the 100-question comparison',
     'These questions selected a candidate; they are not independent held-out evidence. Time axes differ by model.\n'
     '* Earlier Qwen3-Coder results survive only as scores/timings; original per-question outputs were overwritten.'),
    ('looplab-diagnostic-v2', 'holdout', 'diagnostic-holdout-performance',
     'Earlier diagnostic: held-out questions',
     '12 held-out questions | Only the selected 2-loop candidate was evaluated on this split',
     'No candidate met the automatic promotion threshold. Unmeasured 3/4-loop holdouts are omitted, not scored zero.\n'
     '* Earlier Qwen3-Coder results survive only as scores/timings; original per-question outputs were overwritten.'),
    ('looplab-diagnostic-v3-quick', None, 'qwen-quick-performance',
     'Qwen3-Coder: later quick diagnostic',
     '4 questions per split | Recovery runtime | Separate from the earlier diagnostic and main comparison',
     'A quick smoke/quality screen; no automatic promotion. Only 2 loops reached the held-out split.\n'
     'Times sum measured requests with varying output lengths; they exclude model loading.'),
    ('qwen-lm-eval-pilot', 'test', 'qwen-pilot-performance',
     'Qwen3-Coder: benchmark pilot',
     '4 questions per profile | Original official scorer | Excluded from the main comparison',
     'Historical scores are 3/4 for each profile. Later decimal normalization would give 4/4 for each.\n'
     'Times sum measured requests; loading and warm-up excluded. This tiny pilot does not establish a gain.'),
]


def plot(reports):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'looplab-research-2026-11-07', 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold',
                         'axes.labelcolor': '#273344', 'text.color': '#172337',
                         'axes.edgecolor': '#a8b2bf', 'xtick.color': '#273344', 'ytick.color': '#273344'})
    source = reports / 'research-data/performance.csv'
    with source.open(newline='') as f:
        rows = list(csv.DictReader(f))
    output = reports / 'charts'
    output.mkdir(exist_ok=True)
    included = set()
    files = []
    for study, split, filename, title, subtitle, footnote in STUDIES:
        subset = [(i, r) for i, r in enumerate(rows) if r['study'] == study and (split is None or r['split'] == split)]
        groups = {}
        for index, row in subset:
            included.add(index)
            groups.setdefault((row['model'], row['split']), []).append(row)
        ordered = sorted(groups, key=lambda k: (ORDER.index(k[0]), k[1] == 'holdout'))
        if not ordered:
            raise ValueError(f'Missing chart data: {study}/{split}')
        height = 2.75 * len(ordered) + 2.2
        fig, axes = plt.subplots(len(ordered), 2, figsize=(12, height), squeeze=False)
        fig.subplots_adjust(top=1 - 1.38 / height, bottom=0.85 / height, left=.085, right=.98,
                            hspace=.60, wspace=.26)
        fig.text(.085, 1 - .29 / height, title, size=17, weight='bold', va='top')
        fig.text(.085, 1 - .66 / height, subtitle, size=10.5, va='top', color='#465568')
        fig.text(.085, .11 / height, footnote, size=9.5, va='bottom', color='#465568', linespacing=1.6)
        for axes_row, key in zip(axes, ordered):
            profile_rows = sorted(groups[key], key=lambda r: int(r['passes']))
            counts = [int(r['passes']) for r in profile_rows]
            if len(set(counts)) != len(counts) or 1 not in counts:
                raise ValueError('Invalid comparison profiles')
            totals = {int(r['questions']) for r in profile_rows}
            if len(totals) != 1:
                raise ValueError('Cannot compare different question counts')
            total = totals.pop()
            correct = [int(r['correct']) for r in profile_rows]
            seconds = [float(r['request_seconds']) for r in profile_rows]
            baseline = seconds[counts.index(1)]
            minutes = study == 'qwen-lm-eval-v1'
            times = [s / 60 if minutes else s for s in seconds]
            labels = [LABELS[p] for p in counts]
            colors = [COLORS[p] for p in counts]
            model_title = NAMES[key[0]]
            if any(r['raw_answers'] == 'summary_only' for r in profile_rows):
                model_title += ' *'
            if split is None:
                model_title += ' — ' + ('selection' if key[1] == 'selection' else 'held-out')
            left, right = axes_row
            left.set_title(model_title, loc='left', pad=13)
            left.bar(labels, correct, color=colors, width=.65)
            left.set_ylim(0, total * 1.18)
            left.set_ylabel(f'Correct / {total}')
            left.set_yticks(range(0, total + 1, 20 if total == 100 else 2 if total == 12 else 1))
            for index, value in enumerate(correct):
                left.text(index, value + total * .025, str(value), ha='center', va='bottom', weight='bold')
            right.set_title('Time cost on the same questions', loc='left', pad=13, size=11)
            right.bar(labels, times, color=colors, width=.65)
            right.set_ylabel('Total request time (min)' if minutes else 'Total request time (s)')
            right.set_ylim(0, max(times) * 1.38)
            right.yaxis.set_major_locator(MaxNLocator(4))
            for index, value in enumerate(times):
                label = f'{value:.1f} {"min" if minutes else "s"}\n{seconds[index] / baseline:.2f}× original'
                right.text(index, value + max(times) * .035, label, ha='center', va='bottom', size=9.5)
            for ax in axes_row:
                ax.grid(axis='y', color='#dfe5ec', linewidth=.7)
                ax.set_axisbelow(True)
                ax.tick_params(axis='x', length=0, pad=7)
        for extension in ('svg', 'png'):
            path = output / f'{filename}.{extension}'
            metadata = {'Title': title, 'Description': subtitle + '\n' + footnote + '\nSource: reports/research-data/performance.csv'}
            if extension == 'svg':
                metadata['Date'] = None
            fig.savefig(path, dpi=150, facecolor='white', metadata=metadata)
            if extension == 'svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
            files.append(path.name)
        plt.close(fig)
    if included != set(range(len(rows))):
        raise ValueError('Some performance rows are missing from the charts')
    (output / 'manifest.json').write_text(json.dumps({'source': '../research-data/performance.csv',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'performance_rows': len(rows),
        'matplotlib_version': matplotlib.__version__,
        'plot_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'files': {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in files}}, indent=2) + '\n')
    print(json.dumps({'charts': len(STUDIES), 'performance_rows': len(rows), 'files': files}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, default=ROOT / 'reports')
    plot(parser.parse_args().reports)
