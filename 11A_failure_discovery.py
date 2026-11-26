import json, re, time
from pathlib import Path
from datetime import datetime
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
ROOT=Path(__file__).resolve().parent
MODEL=ROOT.parent/'Models'/'gemma-4-e4b-mlx-4bit-transformers-default-v1'
CASES=[('A Jenkins process exits with code 137. Is its root cause confirmed? Explain.','SIGKILL likely; OOM not confirmed.'),('A Kubernetes pod is in CrashLoopBackOff. Can you identify the root cause from this alone?','No; inspect previous logs and termination details.'),('A server returns HTTP 503. Does this prove the database is down?','No; check service, proxy and dependencies.'),('A Docker build fails with exit code 1. Identify what is known and what needs verification.','Generic failure; inspect build output.'),('Disk usage is 90 percent. Does this prove disk exhaustion caused an outage?','No; check filesystem, inodes and logs.'),('An EC2 instance cannot be reached over SSH. Is the security group definitely responsible?','No; multiple network and host causes exist.'),('Linux load average is 12. Does that prove CPU saturation?','No; consider CPU count and uninterruptible tasks.'),('A Kubernetes pod is Pending. Does that necessarily indicate insufficient memory?','No; inspect scheduling events.'),('An application logs connection refused. Does this establish a DNS failure?','No; check listener, address and port.'),('A container has OOMKilled in its termination reason. What does this establish and what remains unknown?','OOM termination indicated; exact trigger needs analysis.'),('A CI pipeline reports exit code 2. Can this code alone identify the failing command?','No; meaning depends on command.'),('An AWS application receives AccessDenied. Does this prove an IAM policy is missing?','No; check identities, explicit denies and policies.')]
def repetition_score(text,n=8):
    words=re.findall(r'\S+',text.lower()); grams=[tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    return 1-len(set(grams))/len(grams) if grams else 0.0
if __name__=='__main__':
    model,processor=load(str(MODEL)); config=load_config(str(MODEL))
    out=ROOT/'results'; out.mkdir(exist_ok=True)
    path=out/f'failures_{datetime.now():%Y%m%d_%H%M%S}.jsonl'
    with path.open('w') as f:
        for i,(q,expected) in enumerate(CASES,1):
            prompt=apply_chat_template(processor,config,q,num_images=0)
            start=time.perf_counter(); result=generate(model,processor,prompt,max_tokens=256,temperature=0.0,verbose=False); mx.synchronize()
            answer=getattr(result,'text',str(result)); tokens=getattr(result,'generation_tokens',None)
            row=dict(id=f'failure_{i:02}',question=q,expected=expected,answer=answer,seconds=time.perf_counter()-start,generation_tokens=tokens,possibly_truncated=tokens>=256 if isinstance(tokens,int) else None,repetition_score=repetition_score(answer),human_correct=None,human_unsupported_claims=None,human_failure_type=None)
            f.write(json.dumps(row)+'\n'); f.flush(); print(i,q,answer,sep='\n')
    print('Saved:',path)
