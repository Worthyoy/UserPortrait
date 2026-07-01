from __future__ import annotations

from io import StringIO
import json
from pathlib import Path

import pandas as pd
import pytest

from user_portrait.evaluation import (
    EvaluationError,
    evaluate_profile_outputs,
    main,
    print_evaluation_report,
    write_evaluation_report,
)


def test_current_sample_outputs_have_expected_time_period_metrics() -> None:
    root = Path(__file__).parents[1]

    report = evaluate_profile_outputs(root / "sample_data", root / "portrait_output")

    time_period = report["modules"]["time_period_tasks"]
    assert time_period["metrics"]["precision"]["value"] is None
    assert time_period["metrics"]["precision"]["status"] == "not_applicable"
    assert time_period["metrics"]["recall"]["value"] == pytest.approx(0.95)
    assert time_period["metrics"]["negative_false_positive_rate"]["value"] == 0.0
    assert report["overall_pass"] is True


def test_report_can_be_printed_and_serialized(tmp_path: Path) -> None:
    truth_dir, profiles_dir = _write_minimal_evaluation_data(tmp_path)
    report = evaluate_profile_outputs(truth_dir, profiles_dir)
    stream = StringIO()

    print_evaluation_report(report, stream=stream)
    output_path = write_evaluation_report(report, tmp_path / "metrics.json")
    stored = json.loads(output_path.read_text(encoding="utf-8"))

    assert "time_period_tasks" in stream.getvalue()
    assert "0.950000" in stream.getvalue()
    assert "NOT_APPLICABLE" in stream.getvalue()
    assert stored["modules"]["time_period_tasks"]["metrics"]["precision"]["value"] is None
    assert stored["overall_pass"] is True


def test_missing_file_and_column_raise_clear_errors(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="missing required CSV"):
        evaluate_profile_outputs(tmp_path / "missing-truth", tmp_path / "missing-output")

    truth_dir, profiles_dir = _write_minimal_evaluation_data(tmp_path / "invalid-column")
    pd.DataFrame([{"user_id": "u1"}]).to_csv(
        profiles_dir / "activity_levels.csv", index=False, encoding="utf-8-sig"
    )
    with pytest.raises(EvaluationError, match="missing required columns: user_activity_level"):
        evaluate_profile_outputs(truth_dir, profiles_dir)


def test_duplicate_keys_raise_clear_error(tmp_path: Path) -> None:
    truth_dir, profiles_dir = _write_minimal_evaluation_data(tmp_path)
    frequent_path = truth_dir / "sample_ground_truth_frequent.csv"
    frequent = pd.read_csv(frequent_path, encoding="utf-8-sig")
    pd.concat([frequent, frequent], ignore_index=True).to_csv(
        frequent_path, index=False, encoding="utf-8-sig"
    )

    with pytest.raises(EvaluationError, match="contains duplicate keys"):
        evaluate_profile_outputs(truth_dir, profiles_dir)


def test_cli_only_fails_threshold_gate_when_requested(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    truth_dir, profiles_dir = _write_minimal_evaluation_data(tmp_path)
    pd.DataFrame(columns=["user_id", "action_name"]).to_csv(
        profiles_dir / "recent_high_freq_tasks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    output_path = tmp_path / "evaluation.json"
    args = [str(truth_dir), str(profiles_dir), "--output-json", str(output_path)]

    assert main(args) == 0
    assert main([*args, "--fail-on-threshold"]) == 1
    stored = json.loads(output_path.read_text(encoding="utf-8"))

    assert stored["overall_pass"] is False
    assert stored["modules"]["recent_high_freq_tasks"]["metrics"]["recall"]["status"] == "fail"
    assert "Overall quality gate: FAIL" in capsys.readouterr().out


def _write_minimal_evaluation_data(root: Path) -> tuple[Path, Path]:
    truth_dir = root / "truth"
    profiles_dir = root / "profiles"
    truth_dir.mkdir(parents=True)
    profiles_dir.mkdir(parents=True)

    truth_tables = {
        "sample_ground_truth_recent.csv": pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "action_name": "recent",
                    "event_time": "2026-03-31 10:00:00",
                    "expected_rank": 1,
                    "expected_activity_level": "high",
                }
            ]
        ),
        "sample_ground_truth_frequent.csv": pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "action_name": "frequent",
                    "expected_event_count": 5,
                    "expected_active_days": 4,
                    "expected_rank": 1,
                }
            ]
        ),
        "sample_ground_truth_periodic.csv": pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "action_name": "periodic",
                    "period_type": "weekly",
                    "next_expected_time": "2026-04-07 09:00:00",
                }
            ]
        ),
        "sample_ground_truth_periodic_negative.csv": pd.DataFrame(
            [{"user_id": "u2", "action_name": "not_periodic"}]
        ),
        "sample_ground_truth_high_freq.csv": pd.DataFrame(
            [{"user_id": "u1", "action_name": "high_freq"}]
        ),
        "sample_ground_truth_high_freq_negative.csv": pd.DataFrame(
            [{"user_id": "u2", "action_name": "not_high_freq"}]
        ),
        "sample_ground_truth_time_period.csv": pd.DataFrame(
            [
                {
                    "user_id": f"u{index:02d}",
                    "action_name": f"time_{index:02d}",
                    "time_period": "morning",
                }
                for index in range(20)
            ]
        ),
        "sample_ground_truth_time_period_negative.csv": pd.DataFrame(
            [{"user_id": "u99", "action_name": "not_time_period"}]
        ),
        "sample_ground_truth_sequences.csv": pd.DataFrame(
            [{"user_id": "u1", "sequence": "A -> B", "sequence_length": 2}]
        ),
    }
    profile_tables = {
        "activity_levels.csv": pd.DataFrame(
            [{"user_id": "u1", "user_activity_level": "high"}]
        ),
        "recent_tasks.csv": pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "action_name": "recent",
                    "event_time": "2026-03-31 10:00:00",
                    "rank": 1,
                }
            ]
        ),
        "frequent_tasks.csv": pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "action_name": "frequent",
                    "event_count": 5,
                    "active_days": 4,
                    "rank": 1,
                }
            ]
        ),
        "periodic_tasks.csv": pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "action_name": "periodic",
                    "period_type": "weekly",
                    "next_expected_time": "2026-04-07 09:00:00",
                }
            ]
        ),
        "recent_high_freq_tasks.csv": pd.DataFrame(
            [{"user_id": "u1", "action_name": "high_freq"}]
        ),
        "time_period_tasks.csv": pd.DataFrame(
            [
                {
                    "user_id": f"u{index:02d}",
                    "action_name": f"time_{index:02d}",
                    "time_period": "morning",
                }
                for index in range(19)
            ]
        ),
        "personal_behavior_sequences.csv": pd.DataFrame(
            [{"user_id": "u1", "sequence": "A -> B", "sequence_length": 2}]
        ),
    }

    for filename, frame in truth_tables.items():
        frame.to_csv(truth_dir / filename, index=False, encoding="utf-8-sig")
    for filename, frame in profile_tables.items():
        frame.to_csv(profiles_dir / filename, index=False, encoding="utf-8-sig")
    return truth_dir, profiles_dir
