#!/usr/bin/env python3
from __future__ import annotations
import json, os, signal, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
from urllib import request

ROOT = Path('/Users/yenhaohuang/Desktop/model-tester')
RES = ROOT / 'results/gemma-4-12b-coder-v1-q4-no-think'
RAW = RES / 'raw'
LOGS = RES / 'logs'
STATUS = RES / 'job_status.json'
MODEL = 'gemma4-coding-Q4_K_M.gguf'
BASE = 'http://127.0.0.1:8098/v1'

def now(): return datetime.now().astimezone().isoformat()
def status(progress, step, state='running'):
    STATUS.write_text(json.dumps({'job_id':'gemma-4-12b-coder-v1-q4-no-think-rerun-truthfulqa-humaneval','status':state,'progress':progress,'current_step':step,'updated_at':now()}, ensure_ascii=False, indent=2), encoding='utf-8')

def run(cmd, log_name, progress, step):
    status(progress, step)
    with (LOGS/log_name).open('w', encoding='utf-8') as f:
        p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
        while True:
            rc = p.poll()
            if rc is not None:
                if rc != 0: raise RuntimeError(f'{step} failed rc={rc}; see {LOGS/log_name}')
                return
            status(progress, step)
            time.sleep(20)

def ready_gate():
    payload = {'model':MODEL,'messages':[{'role':'user','content':'Reply with exactly: READY'}],'temperature':0,'max_tokens':8}
    for i in range(60):
        status(0.08, f'waiting ready gate {i+1}/60')
        try:
            req = request.Request(BASE+'/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'}, method='POST')
            with request.urlopen(req, timeout=30) as r:
                body=json.loads(r.read().decode())
            if body.get('choices'):
                (RES/'ready.json').write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding='utf-8')
                return
        except Exception as e:
            (LOGS/'ready.log').write_text(f'{datetime.now().isoformat()} {type(e).__name__}: {e}\n', encoding='utf-8')
        time.sleep(3)
    raise RuntimeError('server not ready')

def summarize():
    lines = ['# gemma-4-12b-coder-v1-q4-no-think', '', 'No-think rerun for the two suspicious low-score benchmarks.', '', 'Launch flags: `--reasoning off --chat-template-kwargs \'{"enable_thinking": false}\'`', '', '## Accuracy']
    for label, glob in [('truthfulqa','truthfulqa_mcq_*_truthfulqa.json'), ('humaneval','fast_textgen_eval_*_humaneval.json')]:
        files=sorted(RAW.glob(glob))
        if files:
            d=json.loads(files[-1].read_text())
            lines.append(f'- {label}: {d.get("accuracy_pct")} ({d.get("correct")}/{d.get("total")}) — `{files[-1].name}`')
            nonempty=sum(1 for r in d.get('results',[]) if (r.get('thinking') or '').strip())
            lines.append(f'  - non-empty thinking fields: {nonempty}/{len(d.get("results",[]))}')
    (RES/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')

try:
    RES.mkdir(parents=True, exist_ok=True); RAW.mkdir(exist_ok=True); LOGS.mkdir(exist_ok=True)
    status(0.01, 'stopping old port 8098 server if any')
    subprocess.run("pkill -f 'llama-server.*--port 8098' || true", shell=True)
    time.sleep(4)
    status(0.03, 'starting no-think llama-server')
    with (LOGS/'llama-server.log').open('w', encoding='utf-8') as f:
        subprocess.Popen(['nohup','/Users/yenhaohuang/Desktop/llama-bash/gemma-4-12b-coder-v1-q4-no-think.sh'], stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    ready_gate()
    run(['python3','utils/eval_truthfulqa_mcq.py','--model',MODEL,'--base-url',BASE,'--n','100','--max-tokens','64','--timeout-sec','180','--out-dir',str(RAW)], 'eval_truthfulqa.log', 0.25, 'running TruthfulQA no-think 100q')
    run(['python3','utils/eval_fast_textgen_eval.py','--benchmarks','humaneval','--model',MODEL,'--base-url',BASE,'--n','100','--max-tokens','12096','--timeout-sec','180','--out-dir',str(RAW),'--dataset-root','/Users/yenhaohuang/Desktop/datasets/full-textgen-evalset'], 'eval_humaneval.log', 0.65, 'running HumanEval no-think 100q')
    summarize()
    status(1.0, 'completed no-think rerun for TruthfulQA and HumanEval', 'completed')
except Exception as e:
    status(0.0, f'failed: {type(e).__name__}: {e}', 'failed')
    raise
