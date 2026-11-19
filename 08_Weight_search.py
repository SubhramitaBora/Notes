"""Hydra experiment 08: reversible search over Gemma normalization-weight edits.

Run from the Hydra project virtual environment:
    python 08_Weight_search.py

This script does not overwrite the model or save any modified model weights.
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
MODEL_PATH = Path(
    "/Users/s87633/Documents/Weight Space Programming/Models/"
    "gemma-4-e4b-mlx-4bit-transformers-default-v1"
)
RESULTS_DIR = BASE_DIR / "evaluation_results" / "weight_search"
MAX_TOKENS = 350
TEMPERATURE = 0.0

# Three layers x four deltas = 12 candidates, each tested on two questions.
TARGET_LAYERS = (0, 1, 2)
DELTAS = (-0.10, -0.02, 0.02, 0.10)
WEIGHT_INDEX = 0

# Development questions only. Do not use these same cases for final validation.
TEST_CASES = [
    {
        "id": "jenkins_exit_137",
        "question": (
            "A Jenkins pipeline reports 'Process exited with code 137'. "
            "What can be concluded from this evidence alone? "
            "What would you check next? Avoid asserting unverified causes."
        ),
        "review_notes": (
            "137 often means SIGKILL (128+9). OOM is a possibility, "
            "not confirmed without memory metrics, events or logs."
        ),
    },
    {
        "id": "insufficient_evidence",
        "question": (
            "A production service log contains only: 'ERROR: request failed'. "
            "Identify the root cause and the next troubleshooting steps. "
            "Clearly separate known facts from hypotheses."
        ),
        "review_notes": (
            "Root cause cannot be established from this single line; "
            "request timestamps, surrounding logs, tracing and dependency data."
        ),
    },
]


def format_prompt(processor, config, question):
    if hasattr(processor, "tokenizer"):
        tokenizer = processor.tokenizer
    else:
        tokenizer = processor
    return apply_chat_template(
        processor,
        config,
        question,
        num_images=0,
    )


def run_question(model, processor, config, question):
    prompt = format_prompt(processor, config, question)
    started = time.perf_counter()
    result = generate(
        model=model,
        processor=processor,
        prompt=prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        verbose=False,
    )
    elapsed = time.perf_counter() - started
    answer = getattr(result, "text", str(result))
    finish_reason = getattr(result, "finish_reason", None)
    # The mlx-vlm version may not expose finish_reason. This is only a
    # heuristic; an answer ending at the token cap must be reviewed.
    output_tokens = getattr(result, "generation_tokens", None)
    if output_tokens is None:
        output_tokens = getattr(result, "output_tokens", None)
    possibly_truncated = (finish_reason == "length") or (
        finish_reason is None
        and output_tokens is not None
        and output_tokens >= MAX_TOKENS
    )
    return {
        "answer": answer,
        "elapsed_seconds": round(elapsed, 3),
        "output_tokens": output_tokens,
        "finish_reason": finish_reason,
        "possibly_truncated": possibly_truncated,
        "characters": len(answer),
    }


def find_target_names(parameters):
    """Choose existing q_norm tensors, not packed quantized matrices."""
    selected = []
    for layer in TARGET_LAYERS:
        name = f"language_model.model.layers.{layer}.self_attn.q_norm.weight"
        if name in parameters:
            tensor = parameters[name]
            if tensor.size > WEIGHT_INDEX and "float" in str(tensor.dtype):
                selected.append(name)
            else:
                print(f"Skipping unsupported tensor: {name} ({tensor.dtype})")
        else:
            print(f"Not found: {name}")
    return selected


def patched_tensor(original, delta):
    # Compute in float32, then cast to original dtype (e.g. bfloat16).
    # Tiny deltas may round away at bfloat16 precision.
    mask = mx.arange(original.size) == WEIGHT_INDEX
    flat = original.reshape((-1,))
    edited = mx.where(
        mask,
        flat.astype(mx.float32) + delta,
        flat.astype(mx.float32),
    ).astype(original.dtype)
    return edited.reshape(original.shape)


def set_parameter(model, name, tensor):
    model.update(tree_unflatten([(name, tensor)]))
    mx.eval(tensor)


def make_candidate_rows(model, processor, config, names, originals, baseline):
    rows = []
    for name in names:
        original = originals[name]
        for delta in DELTAS:
            edited = patched_tensor(original, delta)
            mx.eval(edited)
            old_value = float(original.reshape((-1,))[WEIGHT_INDEX].item())
            new_value = float(edited.reshape((-1,))[WEIGHT_INDEX].item())
            actual_delta = new_value - old_value
            if actual_delta == 0:
                print(f"Skipping {name} {delta:+.3f}: rounded to no change")
                continue
            candidate_id = f"{name}|delta={delta:+.3f}"
            print(f"\nCandidate: {candidate_id} (actual delta={actual_delta:+.6f})")
            try:
                set_parameter(model, name, edited)
                for case in TEST_CASES:
                    modified = run_question(
                        model, processor, config, case["question"]
                    )
                    base = baseline[case["id"]]
                    rows.append({
                        "candidate": candidate_id,
                        "parameter": name,
                        "requested_delta": delta,
                        "actual_delta": actual_delta,
                        "test_id": case["id"],
                        "question": case["question"],
                        "review_notes": case["review_notes"],
                        "baseline_answer": base["answer"],
                        "modified_answer": modified["answer"],
                        "answers_identical": base["answer"] == modified["answer"],
                        "baseline_seconds": base["elapsed_seconds"],
                        "modified_seconds": modified["elapsed_seconds"],
                        "baseline_output_tokens": base["output_tokens"],
                        "modified_output_tokens": modified["output_tokens"],
                        "baseline_characters": base["characters"],
                        "modified_characters": modified["characters"],
                        "baseline_finish_reason": base["finish_reason"],
                        "modified_finish_reason": modified["finish_reason"],
                        "baseline_possibly_truncated": base["possibly_truncated"],
                        "modified_possibly_truncated": modified["possibly_truncated"],
                        "human_baseline_correct": "",
                        "human_modified_correct": "",
                        "human_modified_unsupported_claims": "",
                        "human_notes": "",
                    })
                    print(
                        f"  {case['id']}: {modified['elapsed_seconds']:.1f}s, "
                        f"{modified['characters']} chars"
                    )
            finally:
                # Always restore before the next candidate, even on failure.
                set_parameter(model, name, original)
    return rows


def write_reports(rows, baseline):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = RESULTS_DIR / f"weight_search_{stamp}"
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    txt_path = prefix.with_suffix(".txt")

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("No candidate rows generated.\n", encoding="utf-8")

    json_path.write_text(
        json.dumps({
            "model_path": str(MODEL_PATH),
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "target_layers": TARGET_LAYERS,
            "deltas": DELTAS,
            "baseline": baseline,
            "results": rows,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("HYDRA EXPERIMENT 08: WEIGHT SEARCH\n")
        f.write("=" * 76 + "\n")
        f.write(f"Candidates tested: {len(set(r['candidate'] for r in rows))}\n")
        f.write(f"Total comparisons: {len(rows)}\n")
        f.write("No automatic accuracy ranking: review all unsupported claims.\n\n")
        for row in rows:
            f.write("=" * 76 + "\n")
            f.write(f"CANDIDATE: {row['candidate']}\n")
            f.write(f"TEST: {row['test_id']}\n")
            f.write(f"ACTUAL DELTA: {row['actual_delta']:+.8f}\n\n")
            f.write(f"QUESTION:\n{row['question']}\n\n")
            f.write(f"REVIEW NOTES:\n{row['review_notes']}\n\n")
            f.write(f"ORIGINAL:\n{row['baseline_answer']}\n\n")
            f.write(f"MODIFIED:\n{row['modified_answer']}\n\n")
            f.write(
                "METRICS: "
                f"baseline={row['baseline_seconds']}s, "
                f"modified={row['modified_seconds']}s; "
                f"chars={row['baseline_characters']} -> {row['modified_characters']}; "
                f"possible truncation={row['modified_possibly_truncated']}\n\n"
            )
    print("\nReports saved:")
    print("TXT: ", txt_path)
    print("CSV: ", csv_path)
    print("JSON:", json_path)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model folder does not exist: {MODEL_PATH}")
    print("Loading Gemma model...")
    model, processor = load(str(MODEL_PATH))
    config = load_config(str(MODEL_PATH))
    params = dict(tree_flatten(model.parameters()))
    names = find_target_names(params)
    if not names:
        raise RuntimeError("No compatible q_norm weights found. Inspect model names.")
    originals = {name: params[name] for name in names}
    print(f"Selected {len(names)} tensors: {names}")

    print("\nGenerating original-model baseline once per question...")
    baseline = {}
    for case in TEST_CASES:
        baseline[case["id"]] = run_question(
            model, processor, config, case["question"]
        )
        print(f"  {case['id']}: {baseline[case['id']]['elapsed_seconds']:.1f}s")

    try:
        rows = make_candidate_rows(
            model, processor, config, names, originals, baseline
        )
    finally:
        # Defensive restoration of every tensor targeted by this experiment.
        for name, original in originals.items():
            set_parameter(model, name, original)

    write_reports(rows, baseline)
    print("\nNext: manually review the TXT report; then validate candidates on NEW cases.")


if __name__ == "__main__":
    main()
