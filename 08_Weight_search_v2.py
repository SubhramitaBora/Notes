"""Hydra 08 v2: repeated, reversible Gemma MLX weight-edit evaluation.

Run: python 08_Weight_search_v2.py
Does not change checkpoint files or save edited weights.
"""
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path('/Users/s87633/Documents/Weight Space Programming/Models/gemma-4-e4b-mlx-4bit-transformers-default-v1')
RESULTS_DIR = BASE_DIR / 'evaluation_results' / 'weight_search_v2'
MAX_TOKENS = 800
TEMPERATURE = 0.0
REPEATS = 2
# Start small on 8 GB unified memory. Expand only after checking results.
TARGET_LAYERS = (0, 1)
DELTAS = (-0.02, 0.02)
WEIGHT_INDEX = 0

# These are DEVELOPMENT cases. Reserve different cases for final validation.
TEST_CASES = [
    ('jenkins_exit_137', 'Jenkins reports process exited with code 137. What is known and what should be checked?', '137 commonly indicates SIGKILL; OOM is possible, not proven.'),
    ('jenkins_maven', 'Jenkins Maven build says Could not find artifact com.acme:core:jar:1.2. What can be concluded?', 'Dependency resolution failed; repository, version, credentials and settings need checking.'),
    ('kubernetes_crashloop', 'A Kubernetes pod shows CrashLoopBackOff but no logs or events are provided. Identify the root cause.', 'Root cause unknown; request describe pod, previous logs and events.'),
    ('kubernetes_pending', 'A Kubernetes pod remains Pending. No describe output is available. What are possible causes?', 'Do not assert a cause; check scheduler events, resources, taints, PVCs.'),
    ('docker_pull', 'docker pull returns pull access denied. Is it definitely an authentication problem?', 'No; image name, tag, permissions and registry configuration may matter.'),
    ('docker_exit', 'A Docker container exited with code 1 and no logs are available. What failed?', 'Nonzero exit indicates failure but does not identify root cause.'),
    ('linux_disk', 'Linux reports No space left on device. Is the disk definitely full?', 'Could be blocks, inodes, quotas, or filesystem constraints; inspect df -h and df -i.'),
    ('linux_oom', 'A Linux process was killed with SIGKILL. Is OOM confirmed?', 'No; check kernel logs, cgroup events, and explicit termination.'),
    ('aws_access_denied', 'An AWS API call returns AccessDenied. Is the IAM identity policy definitely the cause?', 'No; resource policies, SCPs, boundaries, session policies and other controls may apply.'),
    ('insufficient_evidence', 'A service log only says ERROR: request failed. State the root cause.', 'Cannot establish cause; ask for timestamps, traces, nearby logs and dependency data.'),
]

INSTRUCTIONS = (
    'You are assisting with production troubleshooting. Answer in 100-150 words maximum. '
    'Use exactly three headings: Known, Possible, Next checks. '
    'Distinguish observations from hypotheses. Never invent logs, metrics, or confirmed root causes. '
    'If evidence is insufficient, explicitly say the root cause is unknown.\n\n'
)


def run_question(model, processor, config, question):
    prompt = apply_chat_template(processor, config, INSTRUCTIONS + question, num_images=0)
    start = time.perf_counter()
    result = generate(model=model, processor=processor, prompt=prompt,
                      max_tokens=MAX_TOKENS, temperature=TEMPERATURE, verbose=False)
    elapsed = time.perf_counter() - start
    answer = getattr(result, 'text', str(result))
    tokens = getattr(result, 'generation_tokens', None)
    if tokens is None:
        tokens = getattr(result, 'output_tokens', None)
    finish = getattr(result, 'finish_reason', None)
    truncated = (finish == 'length') or (tokens is not None and tokens >= MAX_TOKENS)
    return dict(answer=answer, seconds=round(elapsed, 3), output_tokens=tokens,
                finish_reason=finish, possibly_truncated=truncated,
                chars=len(answer))


def update_one(model, name, value):
    model.update(tree_unflatten([(name, value)]))
    mx.eval(value)


def edited_value(original, delta):
    flat = original.reshape((-1,))
    changed = mx.where(mx.arange(flat.size) == WEIGHT_INDEX,
                       flat.astype(mx.float32) + delta,
                       flat.astype(mx.float32)).astype(original.dtype)
    return changed.reshape(original.shape)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Model not found: {MODEL_PATH}')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print('Loading original Gemma model...')
    model, processor = load(str(MODEL_PATH))
    config = load_config(str(MODEL_PATH))
    parameters = dict(tree_flatten(model.parameters()))
    names = []
    for layer in TARGET_LAYERS:
        name = f'language_model.model.layers.{layer}.self_attn.q_norm.weight'
        tensor = parameters.get(name)
        if tensor is not None and tensor.size > WEIGHT_INDEX and 'float' in str(tensor.dtype):
            names.append(name)
        else:
            print(f'Skipping missing/non-floating tensor: {name}')
    if not names:
        raise RuntimeError('No compatible normalization tensors found')
    originals = {name: parameters[name] for name in names}
    mx.eval(list(originals.values()))

    print(f'Baseline: {len(TEST_CASES)} questions x {REPEATS} repeats')
    baseline = {}
    for case_id, question, notes in TEST_CASES:
        for repeat in range(1, REPEATS + 1):
            print(f'  BASE {case_id} repeat {repeat}')
            baseline[(case_id, repeat)] = run_question(model, processor, config, question)

    rows = []
    for name in names:
        original = originals[name]
        old = float(original.reshape((-1,))[WEIGHT_INDEX].item())
        for delta in DELTAS:
            edited = edited_value(original, delta)
            mx.eval(edited)
            actual = float(edited.reshape((-1,))[WEIGHT_INDEX].item()) - old
            if actual == 0:
                print(f'Skipping rounded-away delta {name} {delta:+.3f}')
                continue
            print(f'Candidate: layer {name}, delta {delta:+.3f} (actual {actual:+.6f})')
            try:
                update_one(model, name, edited)
                for case_id, question, notes in TEST_CASES:
                    for repeat in range(1, REPEATS + 1):
                        modified = run_question(model, processor, config, question)
                        base = baseline[(case_id, repeat)]
                        rows.append({
                            'parameter': name, 'requested_delta': delta, 'actual_delta': actual,
                            'case_id': case_id, 'repeat': repeat, 'question': question,
                            'expected_review_notes': notes,
                            'baseline_answer': base['answer'], 'modified_answer': modified['answer'],
                            'answers_identical': base['answer'] == modified['answer'],
                            'baseline_seconds': base['seconds'], 'modified_seconds': modified['seconds'],
                            'baseline_tokens': base['output_tokens'], 'modified_tokens': modified['output_tokens'],
                            'baseline_finish_reason': base['finish_reason'],
                            'modified_finish_reason': modified['finish_reason'],
                            'baseline_possibly_truncated': base['possibly_truncated'],
                            'modified_possibly_truncated': modified['possibly_truncated'],
                            # Fill manually after reading both answers. 1=yes, 0=no; counts >=0.
                            'human_baseline_correct_0_or_1': '',
                            'human_modified_correct_0_or_1': '',
                            'human_baseline_unsupported_claim_count': '',
                            'human_modified_unsupported_claim_count': '',
                            'human_baseline_complete_0_or_1': '',
                            'human_modified_complete_0_or_1': '',
                            'human_notes': '',
                        })
                        print(f'  {case_id} run {repeat}: {modified["seconds"]:.1f}s, '
                              f'tokens={modified["output_tokens"]}, '
                              f'truncated={modified["possibly_truncated"]}')
            finally:
                update_one(model, name, original)

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = RESULTS_DIR / f'weight_search_v2_{stamp}'
    csv_path = prefix.with_suffix('.csv')
    json_path = prefix.with_suffix('.json')
    txt_path = prefix.with_suffix('.txt')
    with csv_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({
        'settings': {'max_tokens': MAX_TOKENS, 'temperature': TEMPERATURE,
                     'repeats': REPEATS, 'target_layers': TARGET_LAYERS, 'deltas': DELTAS,
                     'cases': len(TEST_CASES), 'candidate_count': len(names) * len(DELTAS)},
        'baseline': [{'case_id': case_id, 'repeat': repeat, **result}
                     for (case_id, repeat), result in baseline.items()],
        'comparisons': rows,
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    with txt_path.open('w', encoding='utf-8') as file:
        file.write('HYDRA EXPERIMENT 08 V2 — HUMAN REVIEW REPORT\n')
        file.write(f'{len(rows)} comparisons; {MAX_TOKENS} max tokens; {REPEATS} repeats\n')
        file.write('Human accuracy/unsupported-claim/completeness fields are in the CSV.\n\n')
        for i, row in enumerate(rows, 1):
            file.write('=' * 78 + '\n')
            file.write(f'{i}. {row["case_id"]} / repeat {row["repeat"]}\n')
            file.write(f'{row["parameter"]} / delta {row["requested_delta"]:+.3f}\n')
            file.write(f'QUESTION: {row["question"]}\n')
            file.write(f'REVIEW GUIDE: {row["expected_review_notes"]}\n\n')
            file.write('ORIGINAL:\n' + row['baseline_answer'] + '\n\n')
            file.write('MODIFIED:\n' + row['modified_answer'] + '\n\n')
            file.write(f'BASE: {row["baseline_seconds"]}s / {row["baseline_tokens"]} tokens '
                       f'/ truncated={row["baseline_possibly_truncated"]}\n')
            file.write(f'EDIT: {row["modified_seconds"]}s / {row["modified_tokens"]} tokens '
                       f'/ truncated={row["modified_possibly_truncated"]}\n\n')
    print('\nSaved results:')
    for path in (txt_path, csv_path, json_path):
        print(path)
    print('\nReview the CSV manually. Do not select a winner based only on speed or text differences.')


if __name__ == '__main__':
    main()
