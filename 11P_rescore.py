
import csv
import re
import sys
from pathlib import Path
from collections import Counter

run = Path(sys.argv[1])

with (run / "responses.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 300, "Expected 300 responses"

patterns = [
    r"^\s*([AB])(?=\s*[.:)]|\s*$)",
    r"^\s*(?:the\s+)?(?:correct|best|recommended)\b"
    r"[^:\n]{0,100}?\b(?:is|would be)\s*:?\s*"
    r"(?:option\s+)?([AB])\b",
    r"^\s*(?:answer|choice|option)\s*:?\s*([AB])\b",
    r"^\s*(?:I\s+)?(?:choose|select)\s+([AB])\b",
]

def extract_choice(answer):
    text = re.sub(r"[*_`#]", "", answer).strip()
    first_line = text.splitlines()[0] if text else ""

    matches = set()

    for pattern in patterns:
        match = re.search(
            pattern, first_line, re.IGNORECASE
        )
        if match:
            matches.add(match.group(1).upper())

    if len(matches) != 1:
        return ""

    choice = matches.pop()

    # Do not automatically accept an ambiguous choice.
    if re.search(
        r"\b[AB]\s+(?:or|and)\s+[AB]\b",
        first_line,
        re.IGNORECASE
    ):
        return ""

    return choice


# Verify the formats shown in the terminal.
assert extract_choice("**A. Inspect memory.**") == "A"
assert extract_choice("Answer: B") == "B"
assert extract_choice(
    "The correct action to take is **B. Review rules.**"
) == "B"
assert extract_choice(
    "The correct first step is **A. Inspect memory.**"
) == "A"
assert extract_choice("Answer: A or B") == ""
assert extract_choice("Both options are possible.") == ""

keys = {
    (r["id"], r["rep"], r["condition"])
    for r in rows
}
assert len(keys) == 300, "Duplicate evaluations"

for row in rows:
    expected = row["expected"].strip().upper()
    assert expected in ("A", "B")

    choice = extract_choice(row["answer"])

    row["rescored_choice"] = choice
    row["rescored_correct"] = (
        int(choice == expected) if choice else ""
    )
    row["rescore_status"] = (
        "parsed" if choice else "review_required"
    )

output = run / "rescored_responses.csv"

with output.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(rows[0])
    )
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "HYDRA 11P - CORRECTED CHOICE SCORING",
    "Total responses: 300",
    "Semantic quality remains unverified.",
]

for condition in ["P0", "P1", "P2", "P3", "P4"]:
    group = [
        r for r in rows
        if r["condition"] == condition
    ]

    assert len(group) == 60

    parsed = [
        r for r in group
        if r["rescored_choice"]
    ]

    correct = sum(
        r["rescored_correct"] == 1
        for r in parsed
    )

    wrong = len(parsed) - correct
    unresolved = len(group) - len(parsed)

    lines.extend([
        "",
        f"Configuration: {condition}",
        f"Total: {len(group)}",
        f"Parsed: {len(parsed)}",
        f"Correct: {correct}",
        f"Incorrect: {wrong}",
        f"Review required: {unresolved}",
        f"Coverage: {100*len(parsed)/len(group):.2f}%",
        f"Accuracy lower bound: {100*correct/len(group):.2f}%",
        f"Accuracy upper bound: {100*(correct+unresolved)/len(group):.2f}%",
    ])

    if unresolved == 0:
        lines.append(
            f"Choice accuracy: {100*correct/len(group):.2f}%"
        )

    lines.append(
        f"Detected choices: "
        f"{dict(Counter(r['rescored_choice'] for r in group))}"
    )

pending = [
    r for r in rows
    if r["rescore_status"] == "review_required"
]

with (run / "rescore_pending.csv").open(
    "w", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(rows[0])
    )
    writer.writeheader()
    writer.writerows(pending)

report = "\n".join(lines)
(run / "rescored_report.txt").write_text(report)

print(report)
print("\nOriginal results preserved.")
print("Output:", output)
