#!/usr/bin/env python
"""
FASS Model Evaluation & Uncertainty Calibration
==================================================
Comprehensive evaluation covering:
    1. MC-Dropout uncertainty decomposition (epistemic vs aleatoric)
    2. Expected Calibration Error (ECE) with reliability diagrams
    3. Safety/performance KPIs (collision accuracy, FN rate, latency)
    4. Scenario coverage scoring
    5. Latency benchmarking (p50/p95/p99)

Supports:
    --synthetic     : evaluate on randomly generated features
    --carla-recorded: evaluate on collected CARLA .npz data
    --checkpoint    : path to saved model checkpoint

Output: JSON evaluation report + console summary.

Usage:
    python -m fass_ml.training.evaluate --synthetic --checkpoint ./fass_checkpoints/best_model.pt
    python -m fass_ml.training.evaluate --carla-recorded ./fass_data --checkpoint ./fass_checkpoints/best_model.pt

ISO 26262 Note:
    Evaluation metrics (especially ECE and FN rate) are critical for
    safety argument.  Results should be reviewed before deploying
    the model in any FASS integration.
"""

import os
import sys
import json
import time
import argparse
import glob
import numpy as np
import torch

from .config import FASSConfig, DEFAULT_CONFIG
from ..models.risk_model import FASSRiskNet


# ============================================================================
# Expected Calibration Error (ECE)
# ============================================================================

def compute_ece(pred_risks: np.ndarray, true_risks: np.ndarray,
                n_bins: int = 10, threshold: float = 0.5) -> dict:
    """Compute Expected Calibration Error with per-bin gap analysis.

    ECE measures how well the predicted probabilities match the observed
    frequency of positive (high-risk) outcomes.  A perfectly calibrated
    model has ECE = 0.  For publication quality, ECE < 0.10 is preferred;
    for safety-critical deployment, ECE < 0.05 is recommended.

    Per-bin gaps identify WHICH confidence ranges are miscalibrated,
    guiding targeted recalibration (e.g., Platt scaling, isotonic regression).

    Parameters
    ----------
    pred_risks : (N,) predicted risk scores [0, 1].
    true_risks : (N,) ground-truth risk labels [0, 1].
    n_bins : number of equally-spaced bins.
    threshold : binarization threshold for "high-risk".

    Returns
    -------
    dict with 'ece', 'mce' (max calibration error), 'bin_accuracies',
    'bin_confidences', 'bin_counts', 'bin_gaps', 'calibration_quality',
    'recommendation'.
    """
    pred_binary = (pred_risks > threshold).astype(float)
    true_binary = (true_risks > threshold).astype(float)
    confidences = np.abs(pred_risks - threshold) + 0.5  # confidence in prediction

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    bin_gaps = []

    ece = 0.0
    mce = 0.0
    total = len(pred_risks)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        count = mask.sum()
        bin_counts.append(int(count))

        if count > 0:
            acc = np.mean(pred_binary[mask] == true_binary[mask])
            conf = np.mean(confidences[mask])
            gap = abs(acc - conf)
            bin_accuracies.append(float(acc))
            bin_confidences.append(float(conf))
            bin_gaps.append(round(float(gap), 6))
            ece += (count / total) * gap
            mce = max(mce, gap)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
            bin_gaps.append(0.0)

    # Calibration quality assessment
    if ece < 0.05:
        quality = 'EXCELLENT'
        recommendation = 'Calibration suitable for safety-critical deployment.'
    elif ece < 0.10:
        quality = 'GOOD'
        recommendation = ('Publication-quality calibration. Consider isotonic '
                          'regression for further improvement.')
    elif ece < 0.20:
        quality = 'FAIR'
        recommendation = ('Train for more epochs with real CARLA data. Apply '
                          'temperature scaling or Platt scaling post-hoc.')
    else:
        quality = 'POOR'
        recommendation = ('Significant miscalibration. Recommend: (1) train '
                          '50+ epochs on diverse CARLA scenarios, (2) apply '
                          'post-hoc calibration (temperature scaling), '
                          '(3) verify data quality and label correctness.')

    return {
        'ece': round(float(ece), 6),
        'mce': round(float(mce), 6),
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts,
        'bin_gaps': bin_gaps,
        'calibration_quality': quality,
        'recommendation': recommendation,
    }


# ============================================================================
# Uncertainty decomposition
# ============================================================================

def evaluate_uncertainty(model: FASSRiskNet, features: np.ndarray,
                         n_samples: int = 30, device: str = 'cpu') -> dict:
    """Run MC-Dropout evaluation and decompose uncertainty.

    Returns per-sample epistemic, aleatoric, and total uncertainty.
    """
    model.to(device)
    model.enable_mc_dropout()

    x = torch.from_numpy(features).to(device)
    n = len(features)

    all_risks = np.zeros((n_samples, n))
    all_vars = np.zeros((n_samples, n))

    with torch.no_grad():
        for s in range(n_samples):
            risk, log_var = model(x)
            all_risks[s] = risk.cpu().numpy().flatten()
            all_vars[s] = np.exp(log_var.cpu().numpy().flatten())

    model.eval()

    # Decomposition
    risk_mean = np.mean(all_risks, axis=0)           # (N,)
    epistemic = np.var(all_risks, axis=0)             # variance of means
    aleatoric = np.mean(all_vars, axis=0)             # mean of variances
    total = epistemic + aleatoric

    return {
        'risk_mean': risk_mean,
        'risk_std': np.std(all_risks, axis=0),
        'epistemic': epistemic,
        'aleatoric': aleatoric,
        'total_uncertainty': total,
        'avg_epistemic': float(np.mean(epistemic)),
        'avg_aleatoric': float(np.mean(aleatoric)),
        'avg_total': float(np.mean(total)),
    }


# ============================================================================
# Safety / performance KPIs
# ============================================================================

def compute_safety_kpis(pred_risks: np.ndarray, true_risks: np.ndarray,
                        threshold: float = 0.5) -> dict:
    """Compute safety-critical performance indicators.

    Returns
    -------
    dict with:
        collision_accuracy  : TP / (TP + FP + FN)
        false_negative_rate : FN / (FN + TP)
        false_positive_rate : FP / (FP + TN)
        precision           : TP / (TP + FP)
        recall              : TP / (TP + FN)
        f1_score            : harmonic mean of precision + recall
    """
    pred_danger = pred_risks > threshold
    true_danger = true_risks > threshold

    tp = int(np.sum(pred_danger & true_danger))
    fp = int(np.sum(pred_danger & ~true_danger))
    fn = int(np.sum(~pred_danger & true_danger))
    tn = int(np.sum(~pred_danger & ~true_danger))

    total = tp + fp + fn + tn
    collision_accuracy = tp / max(tp + fp + fn, 1)
    fn_rate = fn / max(fn + tp, 1)
    fp_rate = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'total_samples': total,
        'collision_accuracy': round(collision_accuracy, 4),
        'false_negative_rate': round(fn_rate, 4),
        'false_positive_rate': round(fp_rate, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
    }


# ============================================================================
# Latency benchmarking
# ============================================================================

def benchmark_latency(model: FASSRiskNet, input_dim: int = 30,
                      n_iters: int = 1000, mc_samples: int = 30,
                      device: str = 'cpu') -> dict:
    """Benchmark inference latency.

    Returns p50, p95, p99 latency in milliseconds.
    """
    model.to(device)
    model.eval()

    x = torch.randn(1, input_dim).to(device)

    # Warmup
    for _ in range(50):
        with torch.no_grad():
            model(x)

    # Single forward pass latency
    single_latencies = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        single_latencies.append((time.perf_counter() - t0) * 1000)

    # MC-Dropout latency (full prediction)
    mc_latencies = []
    for _ in range(min(n_iters // 10, 100)):
        result = model.predict_with_uncertainty(x, n_samples=mc_samples)
        mc_latencies.append(result['latency_ms'])

    single_latencies = np.array(single_latencies)
    mc_latencies = np.array(mc_latencies) if mc_latencies else np.array([0])

    return {
        'single_pass': {
            'p50_ms': round(float(np.percentile(single_latencies, 50)), 3),
            'p95_ms': round(float(np.percentile(single_latencies, 95)), 3),
            'p99_ms': round(float(np.percentile(single_latencies, 99)), 3),
            'mean_ms': round(float(np.mean(single_latencies)), 3),
        },
        'mc_dropout': {
            'p50_ms': round(float(np.percentile(mc_latencies, 50)), 3),
            'p95_ms': round(float(np.percentile(mc_latencies, 95)), 3),
            'p99_ms': round(float(np.percentile(mc_latencies, 99)), 3),
            'mean_ms': round(float(np.mean(mc_latencies)), 3),
            'mc_samples': mc_samples,
        },
    }


# ============================================================================
# Scenario coverage
# ============================================================================

def compute_scenario_coverage(data_dir: str, pred_risks: np.ndarray = None,
                              threshold: float = 0.5) -> dict:
    """Quantitative scenario coverage analysis.

    Reports per-category frame counts, risk distribution statistics,
    and an overall coverage score.  This strengthens experimental rigor
    for publications and patent filings by demonstrating that the model
    has been evaluated across all relevant edge-case categories.

    Coverage Formula:
        coverage = |exercised categories| / |total categories|
        A coverage of 1.0 means every category has ≥1 recorded frame.

    Per-Category Statistics:
        - frame_count : number of .npz frames in the category
        - risk_mean   : mean risk label across frames (if labels available)
        - risk_max    : maximum risk observed
        - has_high_risk : True if any frame has risk > threshold
    """
    from ..data.scenario_generator import SCENARIO_CATEGORIES

    found_categories = set()
    scenario_dirs = []
    per_category_stats = {}

    if os.path.isdir(data_dir):
        for item in sorted(os.listdir(data_dir)):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                scenario_dirs.append(item)
                matched_cat = None
                for cat in SCENARIO_CATEGORIES:
                    if item.startswith(cat):
                        found_categories.add(cat)
                        matched_cat = cat
                        break

                # Count frames and extract risk stats
                npz_files = glob.glob(os.path.join(item_path, 'frame_*.npz'))
                frame_count = len(npz_files)

                risk_values = []
                for npz_path in npz_files[:50]:  # sample up to 50 for speed
                    try:
                        data = np.load(npz_path, allow_pickle=True)
                        if 'labels' in data:
                            import json as _json
                            labels = _json.loads(str(data['labels']))
                            risk_values.append(float(labels.get('risk', 0)))
                    except Exception:
                        pass

                cat_key = matched_cat or item
                stats = {
                    'frame_count': frame_count,
                    'scenario_id': item,
                }
                if risk_values:
                    stats['risk_mean'] = round(float(np.mean(risk_values)), 4)
                    stats['risk_std'] = round(float(np.std(risk_values)), 4)
                    stats['risk_max'] = round(float(np.max(risk_values)), 4)
                    stats['has_high_risk'] = bool(np.max(risk_values) > threshold)
                    stats['high_risk_fraction'] = round(
                        float(np.mean(np.array(risk_values) > threshold)), 4
                    )

                if cat_key not in per_category_stats:
                    per_category_stats[cat_key] = []
                per_category_stats[cat_key].append(stats)

    coverage = len(found_categories) / max(len(SCENARIO_CATEGORIES), 1)
    total_frames = sum(s['frame_count']
                       for cat_list in per_category_stats.values()
                       for s in cat_list)

    # ASIL-D assessment
    if coverage >= 1.0 and total_frames >= 1000:
        asil_readiness = 'HIGH'
    elif coverage >= 0.7 and total_frames >= 500:
        asil_readiness = 'MODERATE'
    else:
        asil_readiness = 'LOW'

    return {
        'coverage_score': round(coverage, 3),
        'coverage_pct': f"{coverage:.0%}",
        'found_categories': sorted(found_categories),
        'missing_categories': sorted(set(SCENARIO_CATEGORIES) - found_categories),
        'total_categories': len(SCENARIO_CATEGORIES),
        'total_frames': total_frames,
        'per_category': per_category_stats,
        'scenario_dirs': scenario_dirs,
        'asil_d_readiness': asil_readiness,
    }


# ============================================================================
# Main evaluation pipeline
# ============================================================================

def evaluate(
    checkpoint_path: str = None,
    config: FASSConfig = None,
    synthetic: bool = False,
    carla_recorded: str = None,
):
    """Run full evaluation pipeline.

    Parameters
    ----------
    checkpoint_path : str
        Path to saved model checkpoint.  If None, creates a fresh model.
    config : FASSConfig
    synthetic : bool
        Use synthetic data.
    carla_recorded : str
        Path to directory of CARLA-recorded .npz files.
    """
    if config is None:
        config = DEFAULT_CONFIG
    config.set_deterministic()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Load model ---
    model = FASSRiskNet(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        dropout_p=config.dropout_p,
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"[FASS Eval] Loaded checkpoint: {checkpoint_path}")
    else:
        print("[FASS Eval] No checkpoint loaded — using random weights (for testing)")

    model.to(device)
    model.eval()

    # --- Load data ---
    if synthetic:
        print("[FASS Eval] Using synthetic data")
        from .train import _create_synthetic_data
        features, true_risks = _create_synthetic_data(n_samples=500, seed=config.global_seed + 100)
    elif carla_recorded:
        print(f"[FASS Eval] Loading CARLA-recorded data from: {carla_recorded}")
        from ..data.feature_extractor import extract_from_npz
        npz_files = sorted(glob.glob(os.path.join(carla_recorded, '**', 'frame_*.npz'),
                                     recursive=True))
        if not npz_files:
            print(f"  ERROR: No .npz files found in {carla_recorded}")
            return

        features_list, risks_list = [], []
        for f in npz_files:
            try:
                feat, labels = extract_from_npz(f)
                features_list.append(feat)
                risks_list.append(labels['risk'])
            except Exception as e:
                print(f"  WARNING: Failed to load {f}: {e}")

        features = np.array(features_list, dtype=np.float32)
        true_risks = np.array(risks_list, dtype=np.float32)
        print(f"  Loaded {len(features)} frames")
    else:
        print("[FASS Eval] No data source specified.  Using synthetic fallback.")
        from .train import _create_synthetic_data
        features, true_risks = _create_synthetic_data(n_samples=500, seed=config.global_seed + 100)

    # --- Run evaluations ---
    print("\n" + "="*60)
    print("  FASS MODEL EVALUATION REPORT")
    print("="*60)

    report = {
        'model': model.summary(),
        'data_source': 'synthetic' if synthetic else (carla_recorded or 'synthetic_fallback'),
        'n_samples': len(features),
    }

    # 1. Uncertainty decomposition
    print("\n── Uncertainty Decomposition ──")
    unc_results = evaluate_uncertainty(model, features, n_samples=config.mc_samples,
                                       device=device)
    pred_risks = unc_results['risk_mean']
    report['uncertainty'] = {
        'avg_epistemic': round(unc_results['avg_epistemic'], 6),
        'avg_aleatoric': round(unc_results['avg_aleatoric'], 6),
        'avg_total': round(unc_results['avg_total'], 6),
    }
    print(f"  Avg epistemic uncertainty: {unc_results['avg_epistemic']:.6f}")
    print(f"  Avg aleatoric uncertainty: {unc_results['avg_aleatoric']:.6f}")
    print(f"  Avg total uncertainty:     {unc_results['avg_total']:.6f}")

    # 2. ECE + Reliability Diagram
    print("\n── Expected Calibration Error ──")
    ece_results = compute_ece(pred_risks, true_risks, n_bins=config.ece_n_bins,
                              threshold=config.risk_threshold)
    report['calibration'] = ece_results
    print(f"  ECE = {ece_results['ece']:.6f}  (MCE = {ece_results['mce']:.6f})")
    print(f"  Quality: {ece_results['calibration_quality']}")
    print(f"  Recommendation: {ece_results['recommendation']}")

    # ASCII Reliability Diagram
    print("\n  Reliability Diagram (confidence → accuracy):")
    print("  " + "-" * 52)
    for i, (acc, conf, gap, cnt) in enumerate(zip(
            ece_results['bin_accuracies'], ece_results['bin_confidences'],
            ece_results['bin_gaps'], ece_results['bin_counts'])):
        bar_len = int(acc * 40) if cnt > 0 else 0
        ideal_len = int(conf * 40) if cnt > 0 else 0
        bar = '█' * bar_len + '░' * max(0, ideal_len - bar_len)
        gap_str = f"gap={gap:.3f}" if cnt > 0 else "empty"
        print(f"  Bin {i:2d} [{conf:.2f}] |{bar:40s}| acc={acc:.3f} {gap_str} (n={cnt})")
    print(f"  " + "-" * 52)

    # 3. Safety KPIs
    print("\n── Safety / Performance KPIs ──")
    kpis = compute_safety_kpis(pred_risks, true_risks, threshold=config.risk_threshold)
    report['safety_kpis'] = kpis
    for k, v in kpis.items():
        print(f"  {k}: {v}")

    # 4. Latency
    print("\n── Latency Benchmark ──")
    latency = benchmark_latency(model, input_dim=config.input_dim,
                                 n_iters=config.latency_benchmark_iters,
                                 mc_samples=config.mc_samples, device=device)
    report['latency'] = latency
    print(f"  Single pass: p50={latency['single_pass']['p50_ms']:.2f}ms  "
          f"p95={latency['single_pass']['p95_ms']:.2f}ms  "
          f"p99={latency['single_pass']['p99_ms']:.2f}ms")
    print(f"  MC-Dropout ({config.mc_samples}x): "
          f"p50={latency['mc_dropout']['p50_ms']:.2f}ms  "
          f"p95={latency['mc_dropout']['p95_ms']:.2f}ms  "
          f"p99={latency['mc_dropout']['p99_ms']:.2f}ms")

    # 5. Scenario coverage (always report, with synthetic fallback)
    print("\n── Scenario Coverage ──")
    if carla_recorded:
        coverage = compute_scenario_coverage(carla_recorded,
                                             threshold=config.risk_threshold)
        report['scenario_coverage'] = coverage
        print(f"  Coverage: {coverage['coverage_pct']} "
              f"({len(coverage['found_categories'])}/{coverage['total_categories']} categories)")
        print(f"  Total frames: {coverage['total_frames']}")
        print(f"  ASIL-D readiness: {coverage['asil_d_readiness']}")
        if coverage['per_category']:
            print("  Per-category breakdown:")
            for cat, entries in coverage['per_category'].items():
                total_cat_frames = sum(e['frame_count'] for e in entries)
                risk_means = [e.get('risk_mean', 0) for e in entries if 'risk_mean' in e]
                avg_risk = np.mean(risk_means) if risk_means else 0
                print(f"    {cat:16s}: {total_cat_frames:5d} frames, "
                      f"avg_risk={avg_risk:.3f}")
        if coverage['missing_categories']:
            print(f"  ⚠ MISSING categories: {coverage['missing_categories']}")
            print(f"    → Run scenario_generator to fill gaps before publication")
    else:
        report['scenario_coverage'] = {
            'coverage_score': 0.0,
            'note': 'No CARLA data directory provided; use --carla-recorded '
                    'for quantitative coverage analysis.',
        }
        print("  No CARLA data directory provided.")
        print("  Use --carla-recorded <dir> for quantitative coverage analysis.")

    # 6. Latency compliance check
    max_ok = config.max_inference_latency_ms
    mc_p99 = latency['mc_dropout']['p99_ms']
    if mc_p99 > max_ok:
        print(f"\n  ⚠ WARNING: MC-Dropout p99 latency ({mc_p99:.1f}ms) "
              f"exceeds {max_ok}ms threshold!")
        report['latency_compliance'] = 'FAIL'
    else:
        print(f"\n  ✓ Latency compliance: p99={mc_p99:.1f}ms < {max_ok}ms")
        report['latency_compliance'] = 'PASS'

    # --- Save report ---
    config.ensure_dirs()
    report_path = config.eval_output
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[FASS Eval] Report saved to: {report_path}")

    return report


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FASS Model Evaluation")
    parser.add_argument('--checkpoint', default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--synthetic', action='store_true',
                        help='Evaluate on synthetic data')
    parser.add_argument('--carla-recorded', default=None,
                        help='Path to CARLA-recorded .npz data directory')
    parser.add_argument('--mc-samples', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = FASSConfig(
        mc_samples=args.mc_samples,
        global_seed=args.seed,
    )

    evaluate(
        checkpoint_path=args.checkpoint,
        config=config,
        synthetic=args.synthetic,
        carla_recorded=args.carla_recorded,
    )


if __name__ == '__main__':
    main()
