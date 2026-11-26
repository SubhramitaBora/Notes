"""Score independently reviewed paired free-text answers from a fresh test set."""
import argparse,csv
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('csv'); a=p.parse_args()
    with open(a.csv,newline='') as f: rows=list(csv.DictReader(f))
    if not rows or len({r['case_id'] for r in rows})!=len(rows): raise ValueError('Need unique paired test cases')
    cols=['baseline_correct','modified_correct','baseline_unsupported','modified_unsupported']
    for r in rows:
        if any(r[c] not in ('0','1') for c in cols): raise ValueError('All review fields must be 0 or 1')
    n=len(rows); b,m,bu,mu=[sum(int(r[c]) for r in rows) for c in cols]
    print('Cases:',n,'Baseline accuracy:',100*b/n,'Modified accuracy:',100*m/n,'Gain pp:',100*(m-b)/n)
    print('Unsupported-answer rates:',100*bu/n,100*mu/n)
    print('Rescued:',sum(r['baseline_correct']=='0' and r['modified_correct']=='1' for r in rows),'Regressed:',sum(r['baseline_correct']=='1' and r['modified_correct']=='0' for r in rows))
