"""Render the new torch-uncertainty-style metric tables for docs/RESULTS.md's Acevedo
artifact-simulation section, computed from the already-existing artifact-axis-sweep
predictions.csv/metrics.json files (see docs/MASTER_INFER_RESULTS_PATH.md).

Only the artifact/corrupted stream is reported (or, for the two-stream OOD-detection
table, the same real-vs-artifact framing the existing AUROC(Entropy) column already
uses) -- the real/in-distribution numbers (including Brier, backfilled from each run's
`real_baseline/metrics.json`) live entirely in the "In-Distribution" section at the top
of docs/RESULTS.md and are intentionally not repeated here. Renders as a 2x2 grid per
axis -- Classification (top-left), Selective Classification (top-right), OOD Detection
(bottom-left), Calibration (bottom-right) -- the two widest tables (Classification,
Calibration) are placed on opposite rows so each row pairs a wide table with a narrower
one. See `.claude/skills/metrics/SKILL.md` and
`src/metrics/artifact_quantification.py::quantify_artifact_impact`
(which this wraps) for the underlying metric definitions.

Usage:
    uv run src/paper_helpers/ood_metrics/render_artifact_results_tables.py \\
        --config configs/paper_helpers/acevedo_artifact_axis_paths.yaml \\
        --output-dir csv/artifact_quantification/acevedo
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import rootutils
import yaml

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.metrics.artifact_quantification import quantify_artifact_impact
from src.visualization.style import order_models


def _fmt_scaled(x: float) -> str:
    """×10⁻² convention used throughout docs/RESULTS.md's artifact section (e.g. ECE
    "4.60" for a raw value of 0.0460). Renders NaN (a legitimately undefined metric,
    e.g. CovAt5Risk when no coverage level reaches <=5% risk) as "—"."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x * 100:.2f}"


def _fmt_unscaled(x: float) -> str:
    """Raw [0, 1]-range quantities (AUROC, AUPR, Cov@5Risk, mean entropy)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.4f}"


def _bold_best(values: list[str], raw: list[float], higher_is_better: bool | None) -> list[str]:
    """Wrap the best-in-column formatted string in **bold**, matching the existing
    tables' convention. `higher_is_better=None` (e.g. mean entropy) bolds nothing."""
    if higher_is_better is None:
        return values
    finite = [(i, v) for i, v in enumerate(raw) if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not finite:
        return values
    best_idx = max(finite, key=lambda iv: iv[1])[0] if higher_is_better else min(finite, key=lambda iv: iv[1])[0]
    out = list(values)
    out[best_idx] = f"**{values[best_idx]}**"
    return out


def _load_artifact_metrics_json(metrics_json_path: str | Path, keys: list[str]) -> dict[str, float]:
    """Pull `artifact.<key>` values already computed at inference time into `metrics.json`
    (see `src/inference/infer_artifact.py`) -- covers ECE (the anchor column) as well as
    Precision/Recall/F1, none of which `quantify_artifact_impact` computes itself. NaN for
    any key missing from the file (or if the file itself is missing)."""
    path = Path(metrics_json_path)
    data = json.loads(path.read_text()) if path.exists() else {}
    return {key: float(data.get(f"artifact.{key}", float("nan"))) for key in keys}


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---:" if i else "---" for i in range(len(headers))]) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _render_2x2_grid(top_left: str, top_right: str, bottom_left: str, bottom_right: str) -> str:
    """Wrap 4 markdown tables in an HTML <table> grid. GFM parses markdown inside a raw
    HTML block only when it's set off by blank lines, hence the blank lines around each
    table below -- dropping them silently degrades to unrendered pipe-and-dash text."""

    def cell(md: str) -> str:
        return f"<td valign=\"top\">\n\n{md}\n\n</td>"

    return (
        "<table>\n<tr>\n"
        f"{cell(top_left)}\n{cell(top_right)}\n"
        "</tr>\n<tr>\n"
        f"{cell(bottom_left)}\n{cell(bottom_right)}\n"
        "</tr>\n</table>"
    )


def render_artifact_axis_tables(
    model_csv_map: dict[str, str | Path],
    real_csv_map: dict[str, str | Path],
    metrics_json_map: dict[str, str | Path],
    axis_label: str,
    output_dir: str | Path,
) -> str:
    """Compute the new metrics for one axis and render its 4 markdown tables."""
    summary_df, _ = quantify_artifact_impact(
        model_csv_map=model_csv_map,
        output_dir=output_dir,
        real_csv_map=real_csv_map,
        metrics_json_map=metrics_json_map,
    )
    # Rows in the project's canonical model order, for whichever models this sidecar has.
    summary_df = summary_df.set_index("model").loc[order_models(summary_df["model"])].reset_index()

    calibration_rows = []
    ood_rows = []
    classification_rows = []
    selective_rows = []
    for _, r in summary_df.iterrows():
        from_json = _load_artifact_metrics_json(metrics_json_map[r["model"]], ["ece", "precision", "recall", "f1"])
        calibration_rows.append(
            [r["model"], from_json["ece"], r["artifact_ece_plus"], r["artifact_ece_minus"], r["artifact_mce"], r["artifact_smece"], r["artifact_aece"]]
        )
        ood_rows.append([r["model"], r["auroc_entropy"], r["aupr_entropy"], r["fpr95_entropy"]])
        classification_rows.append(
            [
                r["model"],
                r["acc_artifact"],
                from_json["precision"],
                from_json["recall"],
                from_json["f1"],
                r["artifact_brier"],
                r["artifact_nll"],
                r["mean_entropy_artifact"],
            ]
        )
        selective_rows.append([r["model"], r["artifact_aurc"], r["artifact_augrc"], r["artifact_cov_5risk"], r["artifact_risk_80cov"]])

    def build(headers, rows, fmts, directions):
        formatted_rows = []
        raw_cols = list(zip(*[r[1:] for r in rows])) if rows else []
        model_names = [r[0] for r in rows]
        formatted_cols = [[fmts[i](v) for v in raw_cols[i]] for i in range(len(raw_cols))]
        formatted_cols = [_bold_best(formatted_cols[i], list(raw_cols[i]), directions[i]) for i in range(len(raw_cols))]
        for row_i, name in enumerate(model_names):
            formatted_rows.append([name] + [formatted_cols[col_i][row_i] for col_i in range(len(raw_cols))])
        return _render_table(headers, formatted_rows)

    calibration_md = build(
        ["Model", "ECE (Artifact) (×10⁻²) ↓", "ECE+ (×10⁻²) ↓", "ECE− (×10⁻²) ↓", "MCE (×10⁻²) ↓", "SmECE (×10⁻²) ↓", "aECE (×10⁻²) ↓"],
        calibration_rows,
        [_fmt_scaled] * 6,
        [False] * 6,
    )
    ood_md = build(
        ["Model", "AUROC (Entropy) ↑", "AUPR (Entropy) ↑", "FPR95 (Entropy) ↓"],
        ood_rows,
        [_fmt_unscaled, _fmt_unscaled, _fmt_unscaled],
        [True, True, False],
    )
    classification_md = build(
        ["Model", "Accuracy (Artifact) ↑", "Precision ↑", "Recall ↑", "F1 ↑", "Brier (×10⁻²) ↓", "NLL (×10⁻²) ↓", "Mean Entropy"],
        classification_rows,
        [_fmt_unscaled, _fmt_unscaled, _fmt_unscaled, _fmt_unscaled, _fmt_scaled, _fmt_scaled, _fmt_unscaled],
        [True, True, True, True, False, False, None],
    )
    selective_md = build(
        ["Model", "AURC (×10⁻²) ↓", "AUGRC (×10⁻²) ↓", "Cov@5%Risk ↑", "Risk@80%Cov (×10⁻²) ↓"],
        selective_rows,
        [_fmt_scaled, _fmt_scaled, _fmt_unscaled, _fmt_scaled],
        [False, False, True, False],
    )

    # Classification (8 cols incl. Model) and Calibration (7 cols) are the two wide tables --
    # pairing each with a narrower one (Selective Classification: 5 cols, OOD Detection: 4
    # cols) balances row width better than putting the two wide ones on the same row.
    grid = _render_2x2_grid(
        top_left=f"**Classification (Artifact)**\n\n{classification_md}",
        top_right=f"**Selective Classification (Artifact)**\n\n{selective_md}",
        bottom_left=f"**OOD Detection — Real vs. Artifact**\n\n{ood_md}",
        bottom_right=f"**Calibration (Artifact)**\n\n{calibration_md}",
    )
    return f"**{axis_label}**\n\n{grid}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the axis-paths YAML sidecar.")
    parser.add_argument("--output-dir", required=True, help="Directory to write quantify_artifact_impact's CSV/plot outputs under.")
    parser.add_argument("--out", default=None, help="Optional path to also save the combined markdown to.")
    args = parser.parse_args()

    axes = yaml.safe_load(Path(args.config).read_text())
    output_dir = Path(args.output_dir)

    sections = []
    for axis_key, axis in axes.items():
        model_csv_map = {name: paths["artifact"] for name, paths in axis["models"].items()}
        real_csv_map = {name: paths["real_baseline"] for name, paths in axis["models"].items()}
        metrics_json_map = {name: paths["metrics_json"] for name, paths in axis["models"].items()}
        section_md = render_artifact_axis_tables(
            model_csv_map=model_csv_map,
            real_csv_map=real_csv_map,
            metrics_json_map=metrics_json_map,
            axis_label=axis["label"],
            output_dir=output_dir / axis_key,
        )
        sections.append(section_md)

    combined = "\n---\n\n".join(sections)
    print(combined)
    if args.out:
        Path(args.out).write_text(combined)


if __name__ == "__main__":
    main()
