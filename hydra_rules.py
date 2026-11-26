"""Conservative lexical diagnostics, not semantic grading."""
import re

def normalize(s):
    return ' '.join(s.casefold().split())

def inspect(row):
    answer=row['answer']; text=normalize(answer)
    coverage=[]
    for criterion in row['criteria']:
        terms=set(re.findall(r'[a-z0-9_]{4,}',criterion.lower()))-{'that','does','with','from','check','inspect','alone','rather','than','should','must','only','when','what'}
        hits=sorted(t for t in terms if t in text)
        coverage.append({'criterion':criterion,'hits':hits,'terms':sorted(terms)})
    return {'word_count':len(answer.split()),'over_100_words':len(answer.split())>100,'limit':row.get('possibly_truncated'),'example_literal_match':normalize(row['unsupported_example']) in text,'lexical_coverage':coverage}

def validate(obj,row):
    if not isinstance(obj,dict): raise ValueError('Expected object')
    for key in ('criteria','complete','unsupported_quotes','failure_type','uncertain','notes'):
        if key not in obj: raise ValueError('Missing '+key)
    c=obj['criteria']
    if not isinstance(c,list) or len(c)!=3 or any(type(x) is not int or x not in (0,1) for x in c): raise ValueError('criteria must be three binary integers')
    if type(obj['complete']) is not int or obj['complete'] not in (0,1) or type(obj['uncertain']) is not bool: raise ValueError('Invalid complete/uncertain')
    q=obj['unsupported_quotes']
    if not isinstance(q,list) or any(not isinstance(x,str) or not x.strip() or normalize(x) not in normalize(row['answer']) for x in q): raise ValueError('Quotes must occur in answer')
    if len(set(map(normalize,q)))!=len(q): raise ValueError('Duplicate quotes')
    if obj['failure_type'] not in ('none','unsupported_cause','factual_error','overconfidence','evidence_ignored','incorrect_diagnostic','fabrication','incomplete','other'): raise ValueError('Invalid failure type')
    if not isinstance(obj['notes'],str): raise ValueError('Invalid notes')
    correct=int(sum(c)==3 and not q and obj['failure_type']=='none')
    if obj['failure_type']=='none' and (q or not correct or not obj['complete']): raise ValueError('Inconsistent none label')
    if obj['failure_type'] in ('unsupported_cause','factual_error','overconfidence','fabrication') and not q: raise ValueError('Claim failure needs quote')
    return dict(obj,correct=correct,unsupported_count=len(q))
