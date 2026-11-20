#!/usr/bin/env python3
"""Offline Hydra CSV evaluator, compatible with Experiment 08 and 08 v2.
Usage: python 09_Hydra_Evaluation_v2.py path/to/experiment.csv
No third-party packages, network calls, or model dependencies.
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

SCORE_FIELDS = [
    'human_baseline_correct_0_or_1', 'human_modified_correct_0_or_1',
    'human_baseline_complete_0_or_1', 'human_modified_complete_0_or_1',
    'human_baseline_unsupported_claim_count', 'human_modified_unsupported_claim_count',
]

def get(r, *names):
    for n in names:
        x = r.get(n)
        if x is not None and str(x).strip() != '':
            return str(x).strip()
    return ''

def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None

def boolean(x):
    x = str(x).strip().lower()
    if x in ('true', 'yes', '1'): return True
    if x in ('false', 'no', '0'): return False
    return None

def percent(n, d):
    return f'{100*n/d:.2f}%' if d else 'N/A'

def candidate(r):
    return get(r, 'candidate') or (get(r, 'parameter') + '|delta=' + get(r, 'requested_delta'))

def aggregate(rows):
    changes = [boolean(get(r, 'answers_identical')) for r in rows]
    known = [v for v in changes if v is not None]
    pairs = [(num(get(r,'baseline_seconds')), num(get(r,'modified_seconds'))) for r in rows]
    pairs = [(a,b) for a,b in pairs if a is not None and b is not None]
    a = sum(p[0] for p in pairs)/len(pairs) if pairs else None
    b = sum(p[1] for p in pairs)/len(pairs) if pairs else None
    latency = 100*(a-b)/a if a and b is not None else None
    bt = [boolean(get(r,'baseline_possibly_truncated')) for r in rows]
    mt = [boolean(get(r,'modified_possibly_truncated')) for r in rows]
    return dict(n=len(rows), changed=sum(v is False for v in known), known=len(known),
                baseline=a, modified=b, latency=latency,
                btrunc=sum(v is True for v in bt), bknown=sum(v is not None for v in bt),
                mtrunc=sum(v is True for v in mt), mknown=sum(v is not None for v in mt))

def metric_lines(rows):
    pairs = {
        'Accuracy': ('human_baseline_correct_0_or_1','human_modified_correct_0_or_1'),
        'Completeness': ('human_baseline_complete_0_or_1','human_modified_complete_0_or_1'),
        'Answers with unsupported claims': ('human_baseline_unsupported_claim_count','human_modified_unsupported_claim_count'),
    }
    lines=[]
    for label,(bk,mk) in pairs.items():
        scored=[]
        for r in rows:
            a=num(get(r,bk)); b=num(get(r,mk))
            if a is None or b is None: continue
            if label=='Answers with unsupported claims':
                if a<0 or b<0 or int(a)!=a or int(b)!=b: continue
                scored.append((a>0,b>0))
            else:
                if a not in (0,1) or b not in (0,1): continue
                scored.append((bool(a),bool(b)))
        if not scored:
            lines.append(f'{label}: NOT SCORED (0 paired reviews)')
            continue
        n=len(scored); a=sum(x for x,y in scored); b=sum(y for x,y in scored)
        lines.append(f'{label}: baseline {percent(a,n)} ({a}/{n}); modified {percent(b,n)} ({b}/{n}); difference {100*(b-a)/n:+.2f} percentage points')
        if label=='Answers with unsupported claims' and a:
            lines.append(f'  Relative unsupported-claim reduction: {100*(a-b)/a:+.2f}%')
    return lines

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv_file', type=Path)
    args=p.parse_args()
    path=args.csv_file.expanduser().resolve()
    if not path.is_file(): p.error(f'File not found: {path}')
    with path.open('r',newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        headers=reader.fieldnames or []
        rows=list(reader)
    if not rows: p.error('CSV has no data rows')
    required=['parameter','requested_delta','baseline_seconds','modified_seconds']
    missing=[k for k in required if k not in headers]
    if missing: p.error('Missing required columns: '+', '.join(missing)+'\nFound: '+', '.join(headers))
    groups=defaultdict(list)
    for r in rows: groups[candidate(r)].append(r)
    overall=aggregate(rows)
    cases={get(r,'case_id','test_id') for r in rows if get(r,'case_id','test_id')}
    lines=['HYDRA EXPERIMENT 09 — OFFLINE EVALUATION','='*56,
           f'Source: {path.name}',f'Comparisons: {len(rows)}',
           f'Candidates: {len(groups)}',f'Unique questions/cases: {len(cases)}','',
           'AUTOMATIC METRICS (NOT QUALITY SCORES)',
           f'Response change rate: {percent(overall["changed"],overall["known"])} ({overall["changed"]}/{overall["known"]})',
           f'Baseline average latency: {overall["baseline"]:.3f}s' if overall['baseline'] is not None else 'Baseline average latency: N/A',
           f'Modified average latency: {overall["modified"]:.3f}s' if overall['modified'] is not None else 'Modified average latency: N/A',
           f'Latency reduction (positive means faster): {overall["latency"]:+.2f}%' if overall['latency'] is not None else 'Latency reduction: N/A',
           f'Baseline truncated: {percent(overall["btrunc"],overall["bknown"])}',
           f'Modified truncated: {percent(overall["mtrunc"],overall["mknown"])}',
           '', 'QUALITY METRICS (PAIRED MANUAL SCORING)']
    lines.extend(metric_lines(rows))
    lines.extend(['','CANDIDATE COMPARISON'])
    out=[]
    for name,subset in sorted(groups.items()):
        s=aggregate(subset)
        lines.append(f'{name}: n={s["n"]}, changed={percent(s["changed"],s["known"])}, latency reduction={s["latency"]:+.2f}%' if s['latency'] is not None else f'{name}: n={s["n"]}, changed={percent(s["changed"],s["known"])}')
        out.append({'candidate':name,'comparisons':s['n'],
                    'response_change_rate_pct':round(100*s['changed']/s['known'],2) if s['known'] else '',
                    'baseline_seconds':round(s['baseline'],3) if s['baseline'] is not None else '',
                    'modified_seconds':round(s['modified'],3) if s['modified'] is not None else '',
                    'latency_reduction_pct':round(s['latency'],2) if s['latency'] is not None else '',
                    'modified_truncation_pct':round(100*s['mtrunc']/s['mknown'],2) if s['mknown'] else ''})
    lines.extend(['','NOTES','Response changes do not prove accuracy improvements.',
                  'Quality scores need human review using consistent criteria.',
                  'Repeated answers to the same question are not independent cases.',
                  'Latency improvements are descriptive, not evidence of faster inference generally.',
                  'All processing is local; no data is uploaded.'])
    stem=path.stem
    # A reviewed input can be re-evaluated without overwriting the original review sheet.
    if stem.endswith('_human_review'):
        stem=stem[:-len('_human_review')]
        make_review=False
    else:
        make_review=True
    report=path.parent/(stem+'_manager_summary.txt')
    table=path.parent/(stem+'_candidate_metrics.csv')
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with table.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    print('\n'.join(lines))
    print('\nSaved:',report,'\nSaved:',table)
    if make_review:
        review=path.parent/(stem+'_human_review.csv')
        if review.exists():
            print('Existing review file preserved:',review)
        else:
            cols=['parameter','requested_delta','actual_delta','case_id','repeat','question',
                  'baseline_answer','modified_answer','answers_identical','baseline_seconds',
                  'modified_seconds','baseline_possibly_truncated','modified_possibly_truncated']+SCORE_FIELDS+['human_notes']
            with review.open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
                for r in rows:
                    item={c:get(r,c) for c in cols}
                    for score in SCORE_FIELDS:
                        item[score]=get(r,score)
                    item['case_id']=get(r,'case_id','test_id')
                    w.writerow(item)
            print('Saved review worksheet:',review)

if __name__=='__main__': main()
