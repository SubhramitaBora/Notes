#!/usr/bin/env python3
"""Offline Hydra Experiment 09: aggregate experiment 08/08 v2 CSV metrics.

Usage: python 09_Hydra_Evaluation.py path/to/weight_search_v2_*.csv
Only Python standard library is required. All files remain local.
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def val(row, *keys):
    for key in keys:
        if key in row and row[key] is not None:
            return str(row[key]).strip()
    return ''


def number(s):
    try:
        x = float(s)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def flag(s):
    if str(s).strip().lower() in ('true', '1', 'yes'): return True
    if str(s).strip().lower() in ('false', '0', 'no'): return False
    return None


def pct(n, d):
    return f'{100*n/d:.2f}%' if d else 'N/A'


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values)/len(values) if values else None


def fmt(x, suffix=''):
    return f'{x:.3f}{suffix}' if x is not None else 'N/A'


def summary(rows):
    count = len(rows)
    same = [flag(val(r, 'answers_identical')) for r in rows]
    changed = sum(v is False for v in same)
    known = sum(v is not None for v in same)
    baseline = [number(val(r, 'baseline_seconds')) for r in rows]
    modified = [number(val(r, 'modified_seconds')) for r in rows]
    paired = [(a,b) for a,b in zip(baseline, modified) if a is not None and b is not None]
    a_mean = mean([a for a,b in paired]); b_mean = mean([b for a,b in paired])
    speed = ((a_mean-b_mean)/a_mean*100) if a_mean and b_mean is not None else None
    bt = [flag(val(r, 'baseline_possibly_truncated')) for r in rows]
    mt = [flag(val(r, 'modified_possibly_truncated')) for r in rows]
    return dict(rows=count, changed=changed, known=known, baseline_s=a_mean, modified_s=b_mean,
                speed_pct=speed, baseline_trunc=sum(x is True for x in bt),
                baseline_trunc_known=sum(x is not None for x in bt),
                modified_trunc=sum(x is True for x in mt),
                modified_trunc_known=sum(x is not None for x in mt))


def score_report(rows):
    # Scores are entered as 1/0 in review CSV; blank means unreviewed.
    fields = ['human_baseline_correct','human_modified_correct','human_modified_unsupported_claims',
              'human_baseline_unsupported_claims','human_baseline_complete','human_modified_complete']
    out = []
    for f in fields:
        values = [flag(val(r, f)) for r in rows]
        scored = [x for x in values if x is not None]
        out.append((f, len(scored), sum(scored), pct(sum(scored), len(scored))))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv_file', type=Path, help='Experiment 08 or 08 v2 CSV')
    args = p.parse_args()
    source = args.csv_file.expanduser().resolve()
    if not source.is_file(): p.error(f'File not found: {source}')
    with source.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames: p.error('CSV has no header')
        rows = list(reader)
    if not rows: p.error('CSV contains no result rows')
    needed = {'candidate', 'baseline_seconds', 'modified_seconds'}
    if not needed.issubset(set(reader.fieldnames)):
        p.error('Unexpected CSV format. Missing: '+', '.join(sorted(needed-set(reader.fieldnames))))

    groups = defaultdict(list)
    for r in rows: groups[val(r, 'candidate') or 'UNKNOWN'].append(r)
    s = summary(rows)
    lines = [
        'HYDRA EXPERIMENT 09 — OFFLINE EVALUATION',
        '='*52,
        f'Source: {source.name}',
        f'Comparisons: {s["rows"]}',
        f'Candidates: {len(groups)}',
        f'Unique test IDs: {len(set(val(r, "test_id") for r in rows if val(r, "test_id")))}',
        '',
        'AUTOMATIC METRICS (NOT ACCURACY)',
        f'Response change rate: {pct(s["changed"],s["known"])} ({s["changed"]}/{s["known"]} known)',
        f'Average baseline time: {fmt(s["baseline_s"]," s")}',
        f'Average modified time: {fmt(s["modified_s"]," s")}',
        f'Latency reduction (positive=faster): {fmt(s["speed_pct"],"%")}',
        f'Baseline truncation: {pct(s["baseline_trunc"],s["baseline_trunc_known"])}',
        f'Modified truncation: {pct(s["modified_trunc"],s["modified_trunc_known"])}',
        '', 'BY CANDIDATE',
    ]
    output_rows = []
    for name, group in sorted(groups.items()):
        t = summary(group)
        lines += [f'  {name}',
                  f'    N={t["rows"]}; changed={pct(t["changed"],t["known"])}; '
                  f'baseline={fmt(t["baseline_s"],"s")}; modified={fmt(t["modified_s"],"s")}; '
                  f'latency reduction={fmt(t["speed_pct"],"%")}; '
                  f'modified truncated={pct(t["modified_trunc"],t["modified_trunc_known"])}']
        output_rows.append(dict(candidate=name, comparisons=t['rows'], response_change_rate_pct=round(100*t['changed']/t['known'],2) if t['known'] else '',
                                baseline_seconds=round(t['baseline_s'],3) if t['baseline_s'] is not None else '',
                                modified_seconds=round(t['modified_s'],3) if t['modified_s'] is not None else '',
                                latency_reduction_pct=round(t['speed_pct'],2) if t['speed_pct'] is not None else '',
                                modified_truncation_pct=round(100*t['modified_trunc']/t['modified_trunc_known'],2) if t['modified_trunc_known'] else ''))

    lines += ['', 'QUALITY SCORES',
              'The source CSV may not contain scored answers. Do NOT interpret response changes as accuracy gains.']
    for f, n, yes, rate in score_report(rows):
        lines.append(f'{f}: {rate} (scored={n}, positive={yes})')
    lines += ['', 'SCORING GUIDE',
              'In the review CSV, enter 1=true, 0=false, blank=not reviewed.',
              'Correct: answer reaches supported conclusion.',
              'Unsupported claim: asserts a cause not supported by evidence.',
              'Complete: answers all parts of question.',
              'Score BOTH baseline and modified for each row.',
              'Do not claim a quality improvement until scores are reviewed and validated.',
              'Repeated answers to the same question are not independent benchmark questions.',
              'Latency is descriptive; compare matched prompts with identical generation settings.',
              'All files are generated locally; nothing is uploaded.',
              '']
    dest = source.parent
    report = dest / (source.stem + '_manager_summary.txt')
    table = dest / (source.stem + '_candidate_metrics.csv')
    review = dest / (source.stem + '_human_review.csv')
    report.write_text('\n'.join(lines), encoding='utf-8')
    with table.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(output_rows[0])); w.writeheader(); w.writerows(output_rows)
    review_fields=['candidate','test_id','requested_delta','actual_delta','question','baseline_answer','modified_answer',
                   'human_baseline_correct','human_modified_correct','human_baseline_unsupported_claims',
                   'human_modified_unsupported_claims','human_baseline_complete','human_modified_complete','human_notes']
    with review.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=review_fields); w.writeheader()
        for r in rows:
            w.writerow({k: val(r,k) if k in r else '' for k in review_fields})
    print('\n'.join(lines))
    print('Saved:', report, '\nSaved:', table, '\nSaved:', review, sep='\n')

if __name__ == '__main__': main()
