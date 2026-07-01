from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence, TextIO

import pandas as pd


class EvaluationError(ValueError):
    """Raised when evaluation inputs are missing or structurally invalid."""


QUALITY_THRESHOLDS: dict[tuple[str, str], tuple[str, float]] = {
    ("frequent_tasks", "precision"): (">=", 1.0),
    ("frequent_tasks", "recall"): (">=", 1.0),
    ("frequent_tasks", "event_count_accuracy"): (">=", 1.0),
    ("frequent_tasks", "active_days_accuracy"): (">=", 1.0),
    ("frequent_tasks", "rank_accuracy"): (">=", 1.0),
    ("periodic_tasks", "precision"): (">=", 0.75),
    ("periodic_tasks", "recall"): (">=", 0.80),
    ("periodic_tasks", "negative_false_positive_rate"): ("<=", 0.02),
    ("recent_high_freq_tasks", "precision"): (">=", 0.80),
    ("recent_high_freq_tasks", "recall"): (">=", 0.85),
    ("recent_high_freq_tasks", "negative_false_positive_rate"): ("<=", 0.15),
    ("time_period_tasks", "recall"): (">=", 0.85),
    ("time_period_tasks", "negative_false_positive_rate"): ("<=", 0.15),
}


def evaluate_profile_outputs(
    ground_truth_dir: str | Path,
    profiles_dir: str | Path,
    *,
    encoding: str = "utf-8-sig",
    dirty_profiles_dir: str | Path | None = None,
) -> dict[str, Any]:
    truth_dir = Path(ground_truth_dir)
    output_dir = Path(profiles_dir)
    dataset_manifest = _read_dataset_manifest(truth_dir)
    time_period_scope = dataset_manifest.get(
        "time_period_ground_truth_scope", "injected_only"
    )

    recent_truth = _read_table(
        truth_dir,
        "sample_ground_truth_recent.csv",
        ["user_id", "action_name", "event_time", "expected_rank", "expected_activity_level"],
        ["user_id", "action_name", "event_time"],
        encoding,
    )
    frequent_truth = _read_table(
        truth_dir,
        "sample_ground_truth_frequent.csv",
        ["user_id", "action_name", "expected_event_count", "expected_active_days", "expected_rank"],
        ["user_id", "action_name"],
        encoding,
    )
    periodic_truth = _read_table(
        truth_dir,
        "sample_ground_truth_periodic.csv",
        ["user_id", "action_name", "period_type", "next_expected_time"],
        ["user_id", "action_name", "period_type"],
        encoding,
    )
    periodic_negative = _read_table(
        truth_dir,
        "sample_ground_truth_periodic_negative.csv",
        ["user_id", "action_name"],
        ["user_id", "action_name"],
        encoding,
    )
    high_freq_truth = _read_table(
        truth_dir,
        "sample_ground_truth_high_freq.csv",
        ["user_id", "action_name"],
        ["user_id", "action_name"],
        encoding,
    )
    high_freq_negative = _read_table(
        truth_dir,
        "sample_ground_truth_high_freq_negative.csv",
        ["user_id", "action_name"],
        ["user_id", "action_name"],
        encoding,
    )
    time_period_required = ["user_id", "action_name", "time_period"]
    if time_period_scope == "exhaustive":
        time_period_required.extend(
            [
                "period_event_count",
                "total_event_count",
                "period_share",
                "active_days",
                "first_time",
                "last_time",
                "time_period_score",
                "rank",
            ]
        )
    time_period_truth = _read_table(
        truth_dir,
        "sample_ground_truth_time_period.csv",
        time_period_required,
        ["user_id", "action_name", "time_period"],
        encoding,
    )
    time_period_negative = _read_table(
        truth_dir,
        "sample_ground_truth_time_period_negative.csv",
        ["user_id", "action_name"],
        ["user_id", "action_name"],
        encoding,
    )
    sequence_truth = _read_table(
        truth_dir,
        "sample_ground_truth_sequences.csv",
        ["user_id", "sequence", "sequence_length"],
        ["user_id", "sequence"],
        encoding,
    )
    injected_time_period = _read_optional_table(
        truth_dir,
        "sample_injected_time_period_patterns.csv",
        ["user_id", "action_name", "time_period"],
        ["user_id", "action_name", "time_period"],
        encoding,
    )
    sequence_negative = _read_optional_table(
        truth_dir,
        "sample_ground_truth_sequences_negative.csv",
        ["user_id", "sequence", "negative_type"],
        ["user_id", "sequence"],
        encoding,
    )

    activity = _read_table(
        output_dir,
        "activity_levels.csv",
        ["user_id", "user_activity_level"],
        ["user_id"],
        encoding,
        require_nonempty=False,
    )
    recent = _read_table(
        output_dir,
        "recent_tasks.csv",
        ["user_id", "action_name", "event_time", "rank"],
        ["user_id", "action_name", "event_time"],
        encoding,
        require_nonempty=False,
    )
    frequent = _read_table(
        output_dir,
        "frequent_tasks.csv",
        ["user_id", "action_name", "event_count", "active_days", "rank"],
        ["user_id", "action_name"],
        encoding,
        require_nonempty=False,
    )
    periodic = _read_table(
        output_dir,
        "periodic_tasks.csv",
        ["user_id", "action_name", "period_type", "next_expected_time"],
        ["user_id", "action_name", "period_type"],
        encoding,
        require_nonempty=False,
    )
    high_freq = _read_table(
        output_dir,
        "recent_high_freq_tasks.csv",
        ["user_id", "action_name"],
        ["user_id", "action_name"],
        encoding,
        require_nonempty=False,
    )
    time_period = _read_table(
        output_dir,
        "time_period_tasks.csv",
        ["user_id", "action_name", "time_period"],
        ["user_id", "action_name", "time_period"],
        encoding,
        require_nonempty=False,
    )
    sequences = _read_table(
        output_dir,
        "personal_behavior_sequences.csv",
        ["user_id", "sequence", "sequence_length"],
        ["user_id", "sequence"],
        encoding,
        require_nonempty=False,
    )

    recent_truth = _normalize_datetime(recent_truth, "event_time")
    recent = _normalize_datetime(recent, "event_time")
    periodic_truth = _normalize_datetime(periodic_truth, "next_expected_time")
    periodic = _normalize_datetime(periodic, "next_expected_time")
    if time_period_scope == "exhaustive":
        for column in ["first_time", "last_time"]:
            time_period_truth = _normalize_datetime(time_period_truth, column)
            time_period = _normalize_datetime(time_period, column)

    modules = {
        "activity_levels": _evaluate_activity(activity, recent_truth),
        "recent_tasks": _evaluate_recent(recent, recent_truth),
        "frequent_tasks": _evaluate_frequent(frequent, frequent_truth),
        "periodic_tasks": _evaluate_periodic(periodic, periodic_truth, periodic_negative),
        "recent_high_freq_tasks": _evaluate_binary_task(
            "recent_high_freq_tasks", high_freq, high_freq_truth, high_freq_negative
        ),
        "time_period_tasks": _evaluate_time_period(
            time_period,
            time_period_truth,
            time_period_negative,
            scope=time_period_scope,
            injected=injected_time_period,
        ),
        "personal_behavior_sequences": _evaluate_sequences(
            sequences, sequence_truth, sequence_negative
        ),
    }
    if dirty_profiles_dir is not None:
        modules["dirty_input_consistency"] = _evaluate_dirty_consistency(
            output_dir, Path(dirty_profiles_dir), encoding
        )
    gated_metrics = [
        metric
        for module in modules.values()
        for metric in module["metrics"].values()
        if metric["gated"]
    ]
    failed_metrics = [metric for metric in gated_metrics if metric["status"] == "fail"]

    return {
        "evaluation_schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ground_truth_dir": str(truth_dir.resolve()),
        "profiles_dir": str(output_dir.resolve()),
        "dirty_profiles_dir": (
            str(Path(dirty_profiles_dir).resolve())
            if dirty_profiles_dir is not None
            else None
        ),
        "dataset_manifest": dataset_manifest,
        "synthetic_regression_only": True,
        "modules": modules,
        "summary": {
            "module_count": len(modules),
            "gated_metric_count": len(gated_metrics),
            "failed_metric_count": len(failed_metrics),
        },
        "overall_pass": not failed_metrics,
    }


def write_evaluation_report(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def print_evaluation_report(
    report: dict[str, Any], *, stream: TextIO | None = None
) -> None:
    output = stream or sys.stdout
    rows: list[tuple[str, str, str, str, str]] = []
    for module_name, module in report["modules"].items():
        for metric_name, metric in module["metrics"].items():
            value = "N/A" if metric["value"] is None else f"{metric['value']:.6f}"
            threshold = "-"
            if metric["threshold"] is not None:
                threshold = f"{metric['comparison']} {metric['threshold']:.6f}"
            rows.append((module_name, metric_name, value, threshold, metric["status"].upper()))

    headers = ("Module", "Metric", "Value", "Threshold", "Status")
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print(
        "  ".join(headers[index].ljust(widths[index]) for index in range(len(headers))),
        file=output,
    )
    print("  ".join("-" * width for width in widths), file=output)
    for row in rows:
        print(
            "  ".join(row[index].ljust(widths[index]) for index in range(len(row))),
            file=output,
        )
    print(file=output)
    gate_status = "PASS" if report["overall_pass"] else "FAIL"
    print(f"Overall quality gate: {gate_status}", file=output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate generated portrait CSV files against synthetic ground truth."
    )
    parser.add_argument("ground_truth_dir", help="Directory containing sample ground-truth CSV files.")
    parser.add_argument("profiles_dir", help="Directory containing generated portrait CSV files.")
    parser.add_argument(
        "--output-json",
        default=None,
        help="JSON report path; defaults to <profiles_dir>/evaluation_metrics.json.",
    )
    parser.add_argument("--encoding", default="utf-8-sig", help="Input CSV encoding.")
    parser.add_argument(
        "--dirty-profiles-dir",
        default=None,
        help="Optional portraits generated from sample_events_dirty.csv for consistency checks.",
    )
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="Return exit code 1 when any gated metric misses its threshold.",
    )
    args = parser.parse_args(argv)

    try:
        report = evaluate_profile_outputs(
            args.ground_truth_dir,
            args.profiles_dir,
            encoding=args.encoding,
            dirty_profiles_dir=args.dirty_profiles_dir,
        )
        output_path = args.output_json or Path(args.profiles_dir) / "evaluation_metrics.json"
        written_path = write_evaluation_report(report, output_path)
    except (EvaluationError, OSError, UnicodeError) as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2

    print_evaluation_report(report)
    print(f"JSON report: {written_path}")
    if args.fail_on_threshold and not report["overall_pass"]:
        return 1
    return 0


def _read_table(
    directory: Path,
    filename: str,
    required_columns: list[str],
    unique_key: list[str],
    encoding: str,
    *,
    require_nonempty: bool = True,
) -> pd.DataFrame:
    path = directory / filename
    if not path.is_file():
        raise EvaluationError(f"missing required CSV: {path}")
    try:
        frame = pd.read_csv(path, encoding=encoding, dtype="string")
    except Exception as exc:
        raise EvaluationError(f"failed to read {path}: {exc}") from exc
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise EvaluationError(f"{path} is missing required columns: {', '.join(missing)}")
    if require_nonempty and frame.empty:
        raise EvaluationError(f"ground-truth CSV is empty: {path}")
    duplicate_rows = frame.duplicated(unique_key, keep=False)
    if duplicate_rows.any():
        examples = frame.loc[duplicate_rows, unique_key].head(3).to_dict("records")
        raise EvaluationError(f"{path} contains duplicate keys {unique_key}: {examples}")
    return frame


def _read_optional_table(
    directory: Path,
    filename: str,
    required_columns: list[str],
    unique_key: list[str],
    encoding: str,
) -> pd.DataFrame | None:
    if not (directory / filename).is_file():
        return None
    return _read_table(
        directory, filename, required_columns, unique_key, encoding
    )


def _read_dataset_manifest(directory: Path) -> dict[str, Any]:
    path = directory / "sample_dataset_manifest.json"
    if not path.is_file():
        return {"time_period_ground_truth_scope": "injected_only"}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"failed to read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"{path} must contain a JSON object")
    scope = value.get("time_period_ground_truth_scope", "injected_only")
    if scope not in {"injected_only", "exhaustive"}:
        raise EvaluationError(f"unsupported time-period ground-truth scope: {scope}")
    return value


def _normalize_datetime(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    result = frame.copy()
    parsed = pd.to_datetime(result[column], errors="coerce")
    invalid = parsed.isna() & result[column].notna()
    if invalid.any():
        examples = result.loc[invalid, column].head(3).tolist()
        raise EvaluationError(f"invalid datetime values in {column}: {examples}")
    result[column] = parsed
    return result


def _evaluate_activity(activity: pd.DataFrame, recent_truth: pd.DataFrame) -> dict[str, Any]:
    levels = recent_truth[["user_id", "expected_activity_level"]].drop_duplicates()
    conflicts = levels.groupby("user_id")["expected_activity_level"].nunique() > 1
    if conflicts.any():
        raise EvaluationError("sample_ground_truth_recent.csv has conflicting activity levels")
    expected = levels.drop_duplicates("user_id")
    matched = expected.merge(activity, on="user_id", how="left", validate="one_to_one")
    accuracy = float(
        (matched["expected_activity_level"] == matched["user_activity_level"]).fillna(False).mean()
    )
    return _module(
        {"expected_users": len(expected), "predicted_users": len(activity)},
        {"classification_accuracy": _metric("activity_levels", "classification_accuracy", accuracy)},
    )


def _evaluate_recent(recent: pd.DataFrame, truth: pd.DataFrame) -> dict[str, Any]:
    keys = ["user_id", "action_name", "event_time"]
    precision, recall, matched_count = _precision_recall(recent, truth, keys)
    rank_accuracy = _column_accuracy(recent, truth, keys, "rank", "expected_rank", numeric=True)
    return _module(
        _counts(len(recent), len(truth), matched_count),
        {
            "precision": _metric("recent_tasks", "precision", precision),
            "recall": _metric("recent_tasks", "recall", recall),
            "rank_accuracy": _metric("recent_tasks", "rank_accuracy", rank_accuracy),
        },
    )


def _evaluate_frequent(frequent: pd.DataFrame, truth: pd.DataFrame) -> dict[str, Any]:
    keys = ["user_id", "action_name"]
    precision, recall, matched_count = _precision_recall(frequent, truth, keys)
    metrics = {
        "precision": _metric("frequent_tasks", "precision", precision),
        "recall": _metric("frequent_tasks", "recall", recall),
        "event_count_accuracy": _metric(
            "frequent_tasks",
            "event_count_accuracy",
            _column_accuracy(frequent, truth, keys, "event_count", "expected_event_count", numeric=True),
        ),
        "active_days_accuracy": _metric(
            "frequent_tasks",
            "active_days_accuracy",
            _column_accuracy(frequent, truth, keys, "active_days", "expected_active_days", numeric=True),
        ),
        "rank_accuracy": _metric(
            "frequent_tasks",
            "rank_accuracy",
            _column_accuracy(frequent, truth, keys, "rank", "expected_rank", numeric=True),
        ),
    }
    return _module(_counts(len(frequent), len(truth), matched_count), metrics)


def _evaluate_periodic(
    periodic: pd.DataFrame, truth: pd.DataFrame, negative: pd.DataFrame
) -> dict[str, Any]:
    keys = ["user_id", "action_name", "period_type"]
    precision, recall, matched_count = _precision_recall(periodic, truth, keys)
    false_positive_rate, negative_hits = _negative_false_positive_rate(
        periodic, negative, ["user_id", "action_name"]
    )
    next_time_accuracy = _column_accuracy(
        periodic, truth, keys, "next_expected_time", "next_expected_time"
    )
    result = _module(
        _counts(len(periodic), len(truth), matched_count, len(negative), negative_hits),
        {
            "precision": _metric("periodic_tasks", "precision", precision),
            "recall": _metric("periodic_tasks", "recall", recall),
            "negative_false_positive_rate": _metric(
                "periodic_tasks", "negative_false_positive_rate", false_positive_rate
            ),
            "next_expected_time_accuracy": _metric(
                "periodic_tasks", "next_expected_time_accuracy", next_time_accuracy
            ),
        },
    )
    result["breakdowns"] = {
        "recall_by_period_type": _recall_breakdown(periodic, truth, keys, "period_type"),
        "recall_by_pattern_strength": _recall_breakdown(
            periodic, truth, keys, "pattern_strength"
        ),
        "negative_false_positive_rate_by_type": _negative_breakdown(
            periodic, negative, ["user_id", "action_name"], "negative_type"
        ),
    }
    return result


def _evaluate_binary_task(
    module_name: str,
    predicted: pd.DataFrame,
    truth: pd.DataFrame,
    negative: pd.DataFrame,
) -> dict[str, Any]:
    keys = ["user_id", "action_name"]
    precision, recall, matched_count = _precision_recall(predicted, truth, keys)
    false_positive_rate, negative_hits = _negative_false_positive_rate(predicted, negative, keys)
    result = _module(
        _counts(len(predicted), len(truth), matched_count, len(negative), negative_hits),
        {
            "precision": _metric(module_name, "precision", precision),
            "recall": _metric(module_name, "recall", recall),
            "negative_false_positive_rate": _metric(
                module_name, "negative_false_positive_rate", false_positive_rate
            ),
        },
    )
    if "negative_type" in negative.columns:
        result["breakdowns"] = {
            "negative_false_positive_rate_by_type": _negative_breakdown(
                predicted, negative, keys, "negative_type"
            )
        }
    return result


def _evaluate_time_period(
    predicted: pd.DataFrame,
    truth: pd.DataFrame,
    negative: pd.DataFrame,
    *,
    scope: str,
    injected: pd.DataFrame | None,
) -> dict[str, Any]:
    keys = ["user_id", "action_name", "time_period"]
    precision, recall, matched_count = _precision_recall(predicted, truth, keys)
    false_positive_rate, negative_hits = _negative_false_positive_rate(
        predicted, negative, ["user_id", "action_name"]
    )
    if scope == "exhaustive":
        metrics = {
            "precision": _metric_with_threshold(precision, ">=", 1.0),
            "recall": _metric_with_threshold(recall, ">=", 1.0),
            "negative_false_positive_rate": _metric(
                "time_period_tasks", "negative_false_positive_rate", false_positive_rate
            ),
        }
        for column in [
            "period_event_count",
            "total_event_count",
            "period_share",
            "active_days",
            "first_time",
            "last_time",
            "time_period_score",
            "rank",
        ]:
            metrics[f"{column}_accuracy"] = _metric_with_threshold(
                _column_accuracy(
                    predicted,
                    truth,
                    keys,
                    column,
                    column,
                    numeric=column
                    in {
                        "period_event_count",
                        "total_event_count",
                        "period_share",
                        "active_days",
                        "time_period_score",
                        "rank",
                    },
                ),
                ">=",
                1.0,
            )
    else:
        precision_note = (
            "Positive ground truth is not exhaustive, so precision is not meaningful."
        )
        metrics = {
            "precision": _metric(
                "time_period_tasks", "precision", None, note=precision_note
            ),
            "recall": _metric("time_period_tasks", "recall", recall),
            "negative_false_positive_rate": _metric(
                "time_period_tasks", "negative_false_positive_rate", false_positive_rate
            ),
        }
    if injected is not None:
        _, injected_recall, _ = _precision_recall(predicted, injected, keys)
        metrics["injected_pattern_recall"] = _metric_with_threshold(
            injected_recall, ">=", 0.85
        )
    result = _module(
        _counts(len(predicted), len(truth), matched_count, len(negative), negative_hits),
        metrics,
    )
    result["ground_truth_scope"] = scope
    result["breakdowns"] = {
        "recall_by_time_period": _recall_breakdown(predicted, truth, keys, "time_period"),
        "negative_false_positive_rate_by_type": _negative_breakdown(
            predicted, negative, ["user_id", "action_name"], "negative_type"
        ),
    }
    return result


def _evaluate_sequences(
    predicted: pd.DataFrame, truth: pd.DataFrame, negative: pd.DataFrame | None
) -> dict[str, Any]:
    keys = ["user_id", "sequence"]
    precision, recall, matched_count = _precision_recall(predicted, truth, keys)
    lengths = pd.to_numeric(predicted["sequence_length"], errors="coerce")
    if lengths.isna().any():
        raise EvaluationError("personal_behavior_sequences.csv contains invalid sequence_length values")
    distribution = {
        str(int(length)): int(count)
        for length, count in lengths.value_counts().sort_index().items()
    }
    metrics = {
            "precision": _metric("personal_behavior_sequences", "precision", precision),
            "recall": _metric("personal_behavior_sequences", "recall", recall),
    }
    negative_hits = None
    if negative is not None:
        negative_rate, negative_hits = _negative_false_positive_rate(
            predicted, negative, keys
        )
        metrics["negative_false_positive_rate"] = _metric_with_threshold(
            negative_rate, "<=", 0.05
        )
    result = _module(
        _counts(
            len(predicted),
            len(truth),
            matched_count,
            len(negative) if negative is not None else None,
            negative_hits,
        ),
        metrics,
    )
    result["sequence_length_distribution"] = distribution
    result["breakdowns"] = {
        "recall_by_sequence_length": _recall_breakdown(
            predicted, truth, keys, "sequence_length"
        )
    }
    if negative is not None:
        result["breakdowns"]["negative_false_positive_rate_by_type"] = _negative_breakdown(
            predicted, negative, keys, "negative_type"
        )
    return result


def _precision_recall(
    predicted: pd.DataFrame, truth: pd.DataFrame, keys: list[str]
) -> tuple[float, float, int]:
    predicted_keys = _key_set(predicted, keys)
    truth_keys = _key_set(truth, keys)
    matched_count = len(predicted_keys & truth_keys)
    precision = matched_count / len(predicted_keys) if predicted_keys else 0.0
    recall = matched_count / len(truth_keys) if truth_keys else 0.0
    return precision, recall, matched_count


def _recall_breakdown(
    predicted: pd.DataFrame,
    truth: pd.DataFrame,
    keys: list[str],
    group_column: str,
) -> dict[str, float]:
    if group_column not in truth.columns:
        return {}
    result: dict[str, float] = {}
    for value, group in truth.groupby(group_column, dropna=False):
        _, recall, _ = _precision_recall(predicted, group, keys)
        result[str(value)] = recall
    return result


def _negative_breakdown(
    predicted: pd.DataFrame,
    negative: pd.DataFrame,
    keys: list[str],
    group_column: str,
) -> dict[str, float]:
    if group_column not in negative.columns:
        return {}
    result: dict[str, float] = {}
    for value, group in negative.groupby(group_column, dropna=False):
        rate, _ = _negative_false_positive_rate(predicted, group, keys)
        result[str(value)] = rate
    return result


def _evaluate_dirty_consistency(
    clean_dir: Path, dirty_dir: Path, encoding: str
) -> dict[str, Any]:
    filenames = [
        "activity_levels.csv",
        "recent_tasks.csv",
        "periodic_tasks.csv",
        "recent_high_freq_tasks.csv",
        "frequent_tasks.csv",
        "time_period_tasks.csv",
        "personal_behavior_sequences.csv",
    ]
    matches: dict[str, bool] = {}
    for filename in filenames:
        clean_path = clean_dir / filename
        dirty_path = dirty_dir / filename
        if not clean_path.is_file() or not dirty_path.is_file():
            raise EvaluationError(
                f"dirty consistency requires both clean and dirty {filename}"
            )
        clean = pd.read_csv(clean_path, encoding=encoding, dtype="string")
        dirty = pd.read_csv(dirty_path, encoding=encoding, dtype="string")
        if list(clean.columns) != list(dirty.columns):
            matches[filename] = False
            continue
        columns = list(clean.columns)
        clean = clean.sort_values(columns).reset_index(drop=True)
        dirty = dirty.sort_values(columns).reset_index(drop=True)
        matches[filename] = clean.equals(dirty)
    matched = sum(matches.values())
    result = _module(
        {"compared_tables": len(filenames), "matched_tables": matched},
        {
            "table_match_rate": _metric_with_threshold(
                matched / len(filenames), ">=", 1.0
            )
        },
    )
    result["table_matches"] = matches
    return result


def _negative_false_positive_rate(
    predicted: pd.DataFrame, negative: pd.DataFrame, keys: list[str]
) -> tuple[float, int]:
    predicted_keys = _key_set(predicted, keys)
    negative_keys = _key_set(negative, keys)
    negative_hits = len(predicted_keys & negative_keys)
    return negative_hits / len(negative_keys), negative_hits


def _column_accuracy(
    predicted: pd.DataFrame,
    truth: pd.DataFrame,
    keys: list[str],
    predicted_column: str,
    truth_column: str,
    *,
    numeric: bool = False,
) -> float:
    matched = truth[keys + [truth_column]].merge(
        predicted[keys + [predicted_column]],
        on=keys,
        how="inner",
        suffixes=("_truth", "_predicted"),
        validate="one_to_one",
    )
    if matched.empty:
        return 0.0
    left_name = truth_column
    right_name = predicted_column
    if truth_column == predicted_column:
        left_name = f"{truth_column}_truth"
        right_name = f"{predicted_column}_predicted"
    left = matched[left_name]
    right = matched[right_name]
    if numeric:
        left = pd.to_numeric(left, errors="coerce")
        right = pd.to_numeric(right, errors="coerce")
        if left.isna().any() or right.isna().any():
            raise EvaluationError(
                f"invalid numeric values while comparing {truth_column} and {predicted_column}"
            )
    return float((left == right).mean())


def _key_set(frame: pd.DataFrame, keys: list[str]) -> set[tuple[Any, ...]]:
    return set(frame[keys].itertuples(index=False, name=None))


def _metric(
    module_name: str,
    metric_name: str,
    value: float | None,
    *,
    note: str | None = None,
) -> dict[str, Any]:
    threshold_spec = QUALITY_THRESHOLDS.get((module_name, metric_name))
    threshold = threshold_spec[1] if threshold_spec else None
    comparison = threshold_spec[0] if threshold_spec else None
    if value is None:
        status = "not_applicable"
    elif threshold_spec is None:
        status = "reported"
    else:
        passed = value >= threshold if comparison == ">=" else value <= threshold
        status = "pass" if passed else "fail"
    result: dict[str, Any] = {
        "value": value,
        "threshold": threshold,
        "comparison": comparison,
        "gated": threshold_spec is not None,
        "status": status,
    }
    if note is not None:
        result["note"] = note
    return result


def _metric_with_threshold(
    value: float | None, comparison: str, threshold: float
) -> dict[str, Any]:
    if value is None:
        status = "not_applicable"
    else:
        passed = value >= threshold if comparison == ">=" else value <= threshold
        status = "pass" if passed else "fail"
    return {
        "value": value,
        "threshold": threshold,
        "comparison": comparison,
        "gated": True,
        "status": status,
    }


def _module(counts: dict[str, int], metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"sample_counts": counts, "metrics": metrics}


def _counts(
    predicted: int,
    positive: int,
    matched_positive: int,
    negative: int | None = None,
    matched_negative: int | None = None,
) -> dict[str, int]:
    result = {
        "predicted": int(predicted),
        "positive_ground_truth": int(positive),
        "matched_positive": int(matched_positive),
    }
    if negative is not None:
        result["negative_ground_truth"] = int(negative)
    if matched_negative is not None:
        result["matched_negative"] = int(matched_negative)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
