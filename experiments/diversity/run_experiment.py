"""
Diversity Experiment: Comparing LR vs Baseline on Output Homogeneity

This experiment tests whether Latent Space Reasoning produces more diverse
outputs than standard LLM generation, inspired by the "Artificial Hivemind"
paper (arXiv:2510.22954).

Experiments:
1. Intra-method diversity: Does LR produce more varied outputs across runs?
2. Inter-model homogeneity: Do different models converge to similar outputs?
   Does LR break this pattern?
"""

import json
import os
import sys
import time
import gc
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from itertools import combinations

import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from latent_reasoning import Engine, Config


# Configuration
FAST_CONFIG = True  # Use lighter evolution settings for faster runs

MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "ibm-granite/granite-4.0-h-1b",
    "meta-llama/Llama-3.2-1B-Instruct",
]

RUNS_PER_CONFIG = 3
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Paths
EXPERIMENT_DIR = Path(__file__).parent
QUERIES_PATH = EXPERIMENT_DIR / "queries.json"
RAW_RESULTS_DIR = EXPERIMENT_DIR / "results" / "raw"
METRICS_DIR = EXPERIMENT_DIR / "results" / "metrics"
ANALYSIS_DIR = EXPERIMENT_DIR / "analysis"


@dataclass
class GenerationResult:
    """Single generation output."""
    query_id: str
    query: str
    model: str
    method: str  # "baseline" or "lr"
    run: int
    output: str
    timestamp: float
    generation_time: float


def load_queries() -> list[dict]:
    """Load queries from JSON file."""
    with open(QUERIES_PATH) as f:
        data = json.load(f)
    return data["queries"]


def save_result(result: GenerationResult):
    """Save a single generation result to disk."""
    filename = f"{result.query_id}_{result.model.replace('/', '_')}_{result.method}_run{result.run}.json"
    filepath = RAW_RESULTS_DIR / filename
    with open(filepath, "w") as f:
        json.dump(asdict(result), f, indent=2)


def load_all_results() -> list[GenerationResult]:
    """Load all saved results from disk."""
    results = []
    for filepath in RAW_RESULTS_DIR.glob("*.json"):
        with open(filepath) as f:
            data = json.load(f)
            results.append(GenerationResult(**data))
    return results


def get_existing_results() -> set[str]:
    """Get set of existing result filenames (for resuming)."""
    return {f.stem for f in RAW_RESULTS_DIR.glob("*.json")}


def create_engine(model: str) -> Engine:
    """Create engine with appropriate config settings."""
    if FAST_CONFIG:
        config = Config()
        config.encoder.model = model
        config.evolution.chains = 3
        config.evolution.generations = 5
        return Engine(config=config, verbosity="silent")
    else:
        return Engine(encoder=model, verbosity="silent")


def run_generation(
    query: str,
    query_id: str,
    model: str,
    method: str,
    run: int,
    engine: Optional[Engine] = None,
) -> GenerationResult:
    """Run a single generation (baseline or LR)."""

    # Create engine if not provided
    if engine is None:
        engine = create_engine(model)

    start_time = time.time()

    if method == "baseline":
        output = engine.run_baseline(query)
    else:  # lr
        result = engine.run(query)
        output = result.plan

    generation_time = time.time() - start_time

    return GenerationResult(
        query_id=query_id,
        query=query,
        model=model,
        method=method,
        run=run,
        output=output,
        timestamp=time.time(),
        generation_time=generation_time,
    )


def run_all_generations(resume: bool = True):
    """Run all generations for the experiment."""
    queries = load_queries()
    existing = get_existing_results() if resume else set()

    total_runs = len(queries) * len(MODELS) * 2 * RUNS_PER_CONFIG  # 2 methods
    completed = len(existing)

    print(f"Total runs needed: {total_runs}")
    print(f"Already completed: {completed}")
    print(f"Remaining: {total_runs - completed}")
    print("-" * 50)

    for model in MODELS:
        print(f"\n{'='*50}")
        print(f"Model: {model}")
        print(f"{'='*50}")

        # Create engine once per model (reuse for all queries)
        engine = None

        try:
            for query_data in queries:
                query_id = query_data["id"]
                query = query_data["query"]

                for method in ["baseline", "lr"]:
                    for run in range(1, RUNS_PER_CONFIG + 1):
                        # Check if already done
                        filename = f"{query_id}_{model.replace('/', '_')}_{method}_run{run}"
                        if filename in existing:
                            continue

                        # Create engine on first use for this model
                        if engine is None:
                            print(f"  Loading model...")
                            engine = create_engine(model)

                        print(f"  [{query_id}] {method} run {run}...", end=" ", flush=True)

                        try:
                            result = run_generation(
                                query=query,
                                query_id=query_id,
                                model=model,
                                method=method,
                                run=run,
                                engine=engine,
                            )
                            save_result(result)
                            print(f"done ({result.generation_time:.1f}s)")
                            completed += 1

                        except Exception as e:
                            print(f"ERROR: {e}")
                            continue

        finally:
            # Clean up model to free VRAM before loading next
            if engine is not None:
                del engine
                gc.collect()
                torch.cuda.empty_cache()

    print(f"\n{'='*50}")
    print(f"Generation complete: {completed}/{total_runs}")


def compute_embeddings(texts: list[str]) -> np.ndarray:
    """Compute sentence embeddings for a list of texts."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings


def compute_pairwise_similarity(embeddings: np.ndarray) -> float:
    """Compute average pairwise cosine similarity."""
    if len(embeddings) < 2:
        return 1.0  # Single item has perfect self-similarity

    sim_matrix = cosine_similarity(embeddings)

    # Get upper triangle (excluding diagonal)
    n = len(embeddings)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(sim_matrix[i, j])

    return np.mean(pairs)


def analyze_intra_method_diversity(results: list[GenerationResult]) -> dict:
    """
    Experiment 1: Measure diversity within each (query, model, method) group.

    For each group of 3 runs, compute pairwise similarity.
    Lower similarity = more diversity.
    """
    print("\nAnalyzing intra-method diversity...")

    # Group results by (query_id, model, method)
    groups = {}
    for r in results:
        key = (r.query_id, r.model, r.method)
        if key not in groups:
            groups[key] = []
        groups[key].append(r.output)

    # Compute similarity for each group
    similarities = {"baseline": [], "lr": []}

    for (query_id, model, method), outputs in groups.items():
        if len(outputs) < 2:
            continue

        embeddings = compute_embeddings(outputs)
        sim = compute_pairwise_similarity(embeddings)
        similarities[method].append({
            "query_id": query_id,
            "model": model,
            "similarity": sim,
        })

    # Aggregate
    baseline_sims = [x["similarity"] for x in similarities["baseline"]]
    lr_sims = [x["similarity"] for x in similarities["lr"]]

    analysis = {
        "baseline": {
            "mean_similarity": np.mean(baseline_sims),
            "std_similarity": np.std(baseline_sims),
            "per_config": similarities["baseline"],
        },
        "lr": {
            "mean_similarity": np.mean(lr_sims),
            "std_similarity": np.std(lr_sims),
            "per_config": similarities["lr"],
        },
        "comparison": {
            "baseline_mean": np.mean(baseline_sims),
            "lr_mean": np.mean(lr_sims),
            "difference": np.mean(baseline_sims) - np.mean(lr_sims),
            "lr_more_diverse": np.mean(lr_sims) < np.mean(baseline_sims),
        }
    }

    return analysis


def analyze_inter_model_homogeneity(results: list[GenerationResult]) -> dict:
    """
    Experiment 3: Measure whether different models produce similar outputs.

    For each query and method, compare outputs across models.
    High similarity = "hivemind" effect.
    """
    print("\nAnalyzing inter-model homogeneity...")

    # Group results by (query_id, method)
    # Take first run from each model for fair comparison
    groups = {}
    for r in results:
        if r.run != 1:  # Only use first run for cross-model comparison
            continue
        key = (r.query_id, r.method)
        if key not in groups:
            groups[key] = {}
        groups[key][r.model] = r.output

    # Compute cross-model similarity for each query
    similarities = {"baseline": [], "lr": []}

    for (query_id, method), model_outputs in groups.items():
        if len(model_outputs) < 2:
            continue

        outputs = list(model_outputs.values())
        models = list(model_outputs.keys())

        embeddings = compute_embeddings(outputs)

        # Compute all pairwise similarities between models
        sim_matrix = cosine_similarity(embeddings)

        pair_sims = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                pair_sims.append({
                    "model_a": models[i],
                    "model_b": models[j],
                    "similarity": sim_matrix[i, j],
                })

        avg_sim = np.mean([p["similarity"] for p in pair_sims])

        similarities[method].append({
            "query_id": query_id,
            "avg_cross_model_similarity": avg_sim,
            "pairs": pair_sims,
        })

    # Aggregate
    baseline_sims = [x["avg_cross_model_similarity"] for x in similarities["baseline"]]
    lr_sims = [x["avg_cross_model_similarity"] for x in similarities["lr"]]

    analysis = {
        "baseline": {
            "mean_cross_model_similarity": np.mean(baseline_sims),
            "std_cross_model_similarity": np.std(baseline_sims),
            "per_query": similarities["baseline"],
        },
        "lr": {
            "mean_cross_model_similarity": np.mean(lr_sims),
            "std_cross_model_similarity": np.std(lr_sims),
            "per_query": similarities["lr"],
        },
        "comparison": {
            "baseline_mean": np.mean(baseline_sims),
            "lr_mean": np.mean(lr_sims),
            "difference": np.mean(baseline_sims) - np.mean(lr_sims),
            "lr_breaks_homogeneity": np.mean(lr_sims) < np.mean(baseline_sims),
        }
    }

    return analysis


def analyze_by_category(results: list[GenerationResult]) -> dict:
    """Break down diversity analysis by query category."""
    queries = load_queries()
    query_categories = {q["id"]: q["category"] for q in queries}

    # Group results by category
    by_category = {}
    for r in results:
        cat = query_categories.get(r.query_id, "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    # Analyze each category
    category_analysis = {}
    for cat, cat_results in by_category.items():
        intra = analyze_intra_method_diversity(cat_results)
        inter = analyze_inter_model_homogeneity(cat_results)
        category_analysis[cat] = {
            "intra_method": intra["comparison"],
            "inter_model": inter["comparison"],
        }

    return category_analysis


def analyze_by_model(results: list[GenerationResult]) -> dict:
    """Break down intra-method diversity by model."""
    by_model = {}
    for r in results:
        if r.model not in by_model:
            by_model[r.model] = []
        by_model[r.model].append(r)

    model_analysis = {}
    for model, model_results in by_model.items():
        intra = analyze_intra_method_diversity(model_results)
        model_analysis[model] = intra["comparison"]

    return model_analysis


def generate_report(
    intra_analysis: dict,
    inter_analysis: dict,
    category_analysis: dict,
    model_analysis: dict,
) -> str:
    """Generate markdown report from analysis results."""

    report = """# Diversity Experiment Results

## Overview

This experiment compares Latent Space Reasoning (LR) against baseline LLM generation
on output diversity, inspired by the "Artificial Hivemind" paper (arXiv:2510.22954).

**Key Questions:**
1. Does LR produce more diverse outputs across repeated runs? (Intra-method diversity)
2. Do different models converge to similar outputs? Does LR break this? (Inter-model homogeneity)

---

## Experiment 1: Intra-Method Diversity

**Question:** When we run the same query multiple times, how similar are the outputs?

*Lower similarity = more diversity (better)*

| Method | Mean Similarity | Std Dev |
|--------|-----------------|---------|
| Baseline | {baseline_intra_mean:.4f} | {baseline_intra_std:.4f} |
| LR | {lr_intra_mean:.4f} | {lr_intra_std:.4f} |

**Difference:** {intra_diff:+.4f} ({"LR more diverse" if intra_analysis["comparison"]["lr_more_diverse"] else "Baseline more diverse"})

---

## Experiment 2: Inter-Model Homogeneity

**Question:** Do different models produce similar outputs on the same query?

*High similarity = "hivemind" effect (bad)*
*Lower similarity = models escape homogeneity (good)*

| Method | Mean Cross-Model Similarity | Std Dev |
|--------|----------------------------|---------|
| Baseline | {baseline_inter_mean:.4f} | {baseline_inter_std:.4f} |
| LR | {lr_inter_mean:.4f} | {lr_inter_std:.4f} |

**Difference:** {inter_diff:+.4f} ({"LR breaks homogeneity" if inter_analysis["comparison"]["lr_breaks_homogeneity"] else "No improvement"})

---

## Results by Category

| Category | Baseline Intra | LR Intra | Δ Intra | Baseline Inter | LR Inter | Δ Inter |
|----------|----------------|----------|---------|----------------|----------|---------|
""".format(
        baseline_intra_mean=intra_analysis["baseline"]["mean_similarity"],
        baseline_intra_std=intra_analysis["baseline"]["std_similarity"],
        lr_intra_mean=intra_analysis["lr"]["mean_similarity"],
        lr_intra_std=intra_analysis["lr"]["std_similarity"],
        intra_diff=intra_analysis["comparison"]["difference"],
        baseline_inter_mean=inter_analysis["baseline"]["mean_cross_model_similarity"],
        baseline_inter_std=inter_analysis["baseline"]["std_cross_model_similarity"],
        lr_inter_mean=inter_analysis["lr"]["mean_cross_model_similarity"],
        lr_inter_std=inter_analysis["lr"]["std_cross_model_similarity"],
        inter_diff=inter_analysis["comparison"]["difference"],
    )

    for cat, data in category_analysis.items():
        report += "| {cat} | {b_intra:.4f} | {l_intra:.4f} | {d_intra:+.4f} | {b_inter:.4f} | {l_inter:.4f} | {d_inter:+.4f} |\n".format(
            cat=cat,
            b_intra=data["intra_method"]["baseline_mean"],
            l_intra=data["intra_method"]["lr_mean"],
            d_intra=data["intra_method"]["difference"],
            b_inter=data["inter_model"]["baseline_mean"],
            l_inter=data["inter_model"]["lr_mean"],
            d_inter=data["inter_model"]["difference"],
        )

    report += """
---

## Results by Model

| Model | Baseline Similarity | LR Similarity | Difference |
|-------|---------------------|---------------|------------|
"""

    for model, data in model_analysis.items():
        report += "| {model} | {baseline:.4f} | {lr:.4f} | {diff:+.4f} |\n".format(
            model=model.split("/")[-1],
            baseline=data["baseline_mean"],
            lr=data["lr_mean"],
            diff=data["difference"],
        )

    report += """
---

## Interpretation

### Intra-Method Diversity
"""

    if intra_analysis["comparison"]["lr_more_diverse"]:
        report += """
**LR produces more diverse outputs** than baseline across repeated runs.
This suggests that evolutionary exploration in latent space leads to different
solutions each time, rather than converging to the same "default" response.
"""
    else:
        report += """
**Baseline produces more diverse outputs** than LR across repeated runs.
This may indicate that LR's optimization converges to similar high-quality
solutions, while baseline's randomness produces more variation (though
potentially lower quality variation).
"""

    report += """
### Inter-Model Homogeneity
"""

    if inter_analysis["comparison"]["lr_breaks_homogeneity"]:
        report += """
**LR helps break inter-model homogeneity.** Different models produce more
distinct outputs when using LR compared to baseline. This suggests LR helps
each model find its own optimal solution rather than converging to the
same "generic" response that the "Artificial Hivemind" paper warns about.
"""
    else:
        report += """
**LR does not significantly reduce inter-model homogeneity.** Different models
still produce similar outputs whether using baseline or LR. The "hivemind"
effect persists regardless of the generation method.
"""

    report += """
---

## Methodology

- **Queries:** 30 open-ended queries across 6 categories (brainstorming, ideation, creative writing, analysis, planning, advice)
- **Models:** 6 models (Qwen3-0.6B/1.7B/4B, DeepSeek-R1-Distill, Granite, Llama-3.2)
- **Runs:** 3 runs per (query, model, method) configuration
- **Embedding:** sentence-transformers/all-MiniLM-L6-v2
- **Metric:** Pairwise cosine similarity (lower = more diverse)

---

*Generated by Latent Space Reasoning diversity experiment*
"""

    return report


def run_analysis():
    """Run full analysis on collected results."""
    print("Loading results...")
    results = load_all_results()
    print(f"Loaded {len(results)} results")

    if len(results) == 0:
        print("No results found. Run generation first.")
        return

    # Run analyses
    intra_analysis = analyze_intra_method_diversity(results)
    inter_analysis = analyze_inter_model_homogeneity(results)
    category_analysis = analyze_by_category(results)
    model_analysis = analyze_by_model(results)

    # Save metrics
    metrics = {
        "intra_method_diversity": intra_analysis,
        "inter_model_homogeneity": inter_analysis,
        "by_category": category_analysis,
        "by_model": model_analysis,
    }

    with open(METRICS_DIR / "diversity_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    print(f"Metrics saved to {METRICS_DIR / 'diversity_metrics.json'}")

    # Generate report
    report = generate_report(intra_analysis, inter_analysis, category_analysis, model_analysis)

    with open(ANALYSIS_DIR / "diversity_report.md", "w") as f:
        f.write(report)

    print(f"Report saved to {ANALYSIS_DIR / 'diversity_report.md'}")

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"\nIntra-Method Diversity:")
    print(f"  Baseline mean similarity: {intra_analysis['baseline']['mean_similarity']:.4f}")
    print(f"  LR mean similarity:       {intra_analysis['lr']['mean_similarity']:.4f}")
    print(f"  LR more diverse: {intra_analysis['comparison']['lr_more_diverse']}")

    print(f"\nInter-Model Homogeneity:")
    print(f"  Baseline cross-model similarity: {inter_analysis['baseline']['mean_cross_model_similarity']:.4f}")
    print(f"  LR cross-model similarity:       {inter_analysis['lr']['mean_cross_model_similarity']:.4f}")
    print(f"  LR breaks homogeneity: {inter_analysis['comparison']['lr_breaks_homogeneity']}")


def run_single_model(model: str, resume: bool = True):
    """Run generations for a single model only."""
    queries = load_queries()
    existing = get_existing_results() if resume else set()

    total_runs = len(queries) * 2 * RUNS_PER_CONFIG  # 2 methods
    model_existing = sum(1 for e in existing if model.replace('/', '_') in e)

    print(f"Model: {model}")
    print(f"Total runs needed: {total_runs}")
    print(f"Already completed: {model_existing}")
    print(f"Remaining: {total_runs - model_existing}")
    print("-" * 50)

    engine = None

    try:
        for query_data in queries:
            query_id = query_data["id"]
            query = query_data["query"]

            for method in ["baseline", "lr"]:
                for run in range(1, RUNS_PER_CONFIG + 1):
                    filename = f"{query_id}_{model.replace('/', '_')}_{method}_run{run}"
                    if filename in existing:
                        continue

                    if engine is None:
                        print(f"Loading model...")
                        engine = Engine(encoder=model, verbosity="silent")

                    print(f"[{query_id}] {method} run {run}...", end=" ", flush=True)

                    try:
                        result = run_generation(
                            query=query,
                            query_id=query_id,
                            model=model,
                            method=method,
                            run=run,
                            engine=engine,
                        )
                        save_result(result)
                        print(f"done ({result.generation_time:.1f}s)")

                    except Exception as e:
                        print(f"ERROR: {e}")
                        continue

    finally:
        if engine is not None:
            del engine
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\nModel {model} complete!")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Diversity experiment for LR vs Baseline")
    parser.add_argument("command", choices=["generate", "analyze", "all"],
                       help="Command to run")
    parser.add_argument("--no-resume", action="store_true",
                       help="Start fresh (don't resume from existing results)")
    parser.add_argument("--model", type=str, default=None,
                       help="Run only this model (for parallel execution)")

    args = parser.parse_args()

    # Ensure directories exist
    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    if args.command == "generate":
        if args.model:
            run_single_model(args.model, resume=not args.no_resume)
        else:
            run_all_generations(resume=not args.no_resume)
    elif args.command == "analyze":
        run_analysis()
    elif args.command == "all":
        if args.model:
            run_single_model(args.model, resume=not args.no_resume)
        else:
            run_all_generations(resume=not args.no_resume)
        run_analysis()


if __name__ == "__main__":
    main()
