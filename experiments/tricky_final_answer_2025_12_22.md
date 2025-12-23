# Tricky Final-Answer Evaluation (2025-12-22)

Purpose: Evaluate final-answer-only decoding on 10 tricky logic queries (tests 34-43)
with a 1024 token decode budget across quantization and decode strategies.

Setup:
- Model: Qwen/Qwen3-0.6B
- Evolution: chains=3, generations=3, survivors=2
- Quantization: none, auto, 4bit
- Decode strategy: best, combined
- Final-answer-only prompt injected via
  eval_results/latent_matrix_tricky_final_1024/run_latent_matrix_tricky_final_1024.py

Artifacts (gitignored):
- eval_results/latent_matrix_tricky_final_1024 (summary.json, summary.csv, per-test outputs)

Top findings:
- test42_logic_grid_houses errored for all configs: "cannot convert float infinity to integer".
- Final-answer-only prompt did not suppress <think>; most outputs are partial reasoning
  and stop at max_generations.
- Most tests have no explicit final answer in any config.
- Only explicit correct final answer observed: test41_bag_probability in 4bit+combined
  ("3/4") in eval_results/latent_matrix_tricky_final_1024/test41_bag_probability_4bit_combined.json.
- Some outputs include clearly invalid claims (e.g., test43 water jug saying 6L in the 5L jug).

Per-test notes (examples):
- test34_code_lock: all configs truncated mid-deduction; no final code stated.
  Example: eval_results/latent_matrix_tricky_final_1024/test34_code_lock_none_best.json.
- test35_bridge_torch: confusion about torch-return rules and no final sequence/time.
  Example: eval_results/latent_matrix_tricky_final_1024/test35_bridge_torch_none_best.json.
- test36_resource_schedule: partial critical-path reasoning without a final schedule.
  Example: eval_results/latent_matrix_tricky_final_1024/test36_resource_schedule_auto_combined.json.
- test37_constrained_shortest_path: partial path enumeration; no final path/cost.
  Example: eval_results/latent_matrix_tricky_final_1024/test37_constrained_shortest_path_none_best.json.
- test38_stack_machine: inconsistent reasoning about DUP semantics; no final stack.
  Example: eval_results/latent_matrix_tricky_final_1024/test38_stack_machine_none_combined.json.
- test39_knights_knaves_two: casework started; no final assignment.
  Example: eval_results/latent_matrix_tricky_final_1024/test39_knights_knaves_two_none_combined.json.
- test40_number_theory_crt: CRT steps started; no final N.
  Example: eval_results/latent_matrix_tricky_final_1024/test40_number_theory_crt_none_best.json.
- test41_bag_probability: 4bit+combined returns "The probability is 3/4" (correct).
  Example: eval_results/latent_matrix_tricky_final_1024/test41_bag_probability_4bit_combined.json.
- test42_logic_grid_houses: all configs error.
  Example: eval_results/latent_matrix_tricky_final_1024/test42_logic_grid_houses_none_best.json.
- test43_water_jug_8_5: 4bit+best claims "Fill the 8L jug... resulting in 6L in the 5L jug"
  (invalid). Example: eval_results/latent_matrix_tricky_final_1024/test43_water_jug_8_5_4bit_best.json.

Notes:
- stop_reason is consistently "max_generations" on non-error runs; outputs are cut off
  even with max_tokens=1024.
- No clear quality differences between quantization levels or decode strategy based on
  answer completeness in this run.

## Tricky Plan Evaluation (2025-12-22, 2048 tokens)
Purpose: Evaluate plan-quality decoding (not final-answer-only) on tests 34-43 with a
2048 token decode budget across quantization and decode strategies.

Setup:
- Models: Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B
- Evolution: chains=3, generations=3, survivors=2
- Quantization: none, 4bit
- Decode strategy: best, combined

Artifacts (gitignored):
- eval_results/latent_matrix_tricky_2048_best
- eval_results/latent_matrix_tricky_2048_combined

Top findings:
- Combined did not consistently improve answer quality vs best; results were mixed or tied.
- Qwen3-1.7B generally outperformed Qwen3-0.6B on answer correctness/coverage.
- 4bit tracked close to none overall; best+4bit is the default going forward.
- Example corrects: Qwen3-1.7B + 4bit + best reached the correct CRT solution (N=1103)
  and the correct 6-step water jug sequence; Qwen3-1.7B + 4bit + combined produced
  the correct 3/4 for test41 and a correct water jug sequence.
- The inf/NaN seed conversion error was fixed in src/latent_reasoning/core/encoder.py,
  and the 2048 runs completed without that failure.
