Hydra 11E.1 — copy the two scripts and JSON into Experiment11 using approved means.
From project root with existing .venv active:
python Experiment11/11E_1_discover.py --smoke
python Experiment11/11E_1_discover.py
RUN=$(ls -td Experiment11/results/genuine_* | head -n 1)
cat "$RUN/report.txt"
python Experiment11/11E_1_review.py prepare "$RUN"
open "$RUN/semantic_review.csv"
python Experiment11/11E_1_review.py score "$RUN"
Resume interrupted full run: python Experiment11/11E_1_discover.py --resume "$RUN"
Resume smoke: add --smoke. Use exact interrupted folder, not a newer run.
Review: semantic_correct=1 if explanation is materially accurate; complete=1 if it addresses the decisive evidence; unsupported_count=number of distinct unsupported claims; quote one verbatim example if count positive; decision_verified=1 only if initial parsed letter accurately represents the model's decision and explanation does not contradict it. Use 0 otherwise. Leave uncertain rows blank, do not guess. Save CSV as UTF-8. Never change source columns. Review labels are not generated automatically.
Discovery 30, preliminary check 20. Do not use check rows to fit directions. Final evaluation needs new unseen questions. Labels A/B are balanced; semantic options are manually authored, not guaranteed equally difficult. The margin is for single-token labels, not full explanations. Generation may use different tokenization or hidden reasoning; replay is a teacher-forced prompt state, not an actual generation trace. At the same prompt position both candidate next tokens share the same activation; do not subtract identical states. Compare across verified failures and matched successes, or use controlled activation interventions. No automatic weight edit approval.
