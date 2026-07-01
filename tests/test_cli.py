from __future__ import annotations

import sys

import pandas as pd

from user_portrait import cli


def test_cli_reads_string_ids_and_builds_manifest(monkeypatch) -> None:
    events = pd.DataFrame(
        [
            {"user_id": "001", "action_name": "查余额", "event_time": "2026-03-30 10:00:00"},
            {"user_id": "001", "action_name": "转账", "event_time": "invalid"},
            {"user_id": "001", "action_name": "转账", "event_time": "2026-04-01 10:00:00"},
        ]
    )
    captured: dict[str, object] = {}

    def fake_read_csv(input_csv, *, encoding, dtype):
        assert input_csv == "events.csv"
        assert encoding == "utf-8-sig"
        assert dtype == {"user_id": "string", "action_name": "string"}
        return events

    def fake_build_user_profiles(input_events, *, now, config):
        assert input_events is events
        assert now == "2026-03-31 12:00:00"
        return {
            "activity_levels": pd.DataFrame([{"user_id": "001"}]),
            "time_period_tasks": pd.DataFrame(
                [{"user_id": "001", "action_name": "balance"}]
            ),
        }

    def fake_write_snapshot(output_dir, profiles, manifest, *, encoding):
        captured["output_dir"] = output_dir
        captured["profiles"] = profiles
        captured["manifest"] = manifest
        captured["encoding"] = encoding

    monkeypatch.setattr(cli.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(cli, "build_user_profiles", fake_build_user_profiles)
    monkeypatch.setattr(cli, "_write_snapshot", fake_write_snapshot)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "user_portrait.cli",
            "events.csv",
            "profiles",
            "--now",
            "2026-03-31 12:00:00",
        ],
    )

    cli.main()

    manifest = captured["manifest"]
    assert manifest["profile_schema_version"] == 2
    assert manifest["reference_time"] == "2026-03-31T12:00:00"
    assert manifest["config"]["daily_max_avg_interval_days"] == 1.5
    assert manifest["config"]["high_freq_min_count"] == 4
    assert manifest["config"]["sequence_recency_decay_days"] == 30.0
    assert "high_freq_recent_count_7d" not in manifest["config"]
    assert manifest["input_quality"] == {
        "input_row_count": 3,
        "invalid_row_count": 1,
        "future_event_count": 1,
        "valid_event_count": 1,
        "filtered_row_count": 2,
    }
    assert manifest["output_row_counts"] == {
        "activity_levels": 1,
        "time_period_tasks": 1,
    }
    assert captured["encoding"] == "utf-8-sig"


def test_input_quality_normalizes_timezone_aware_events() -> None:
    events = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "action_name": "past",
                "event_time": "2026-03-31T00:00:00Z",
            },
            {
                "user_id": "u1",
                "action_name": "future",
                "event_time": "2026-03-31T05:00:00Z",
            },
        ]
    )
    timezone = cli._resolve_business_timezone("Asia/Shanghai")
    reference_time = cli._normalize_timestamp("2026-03-31 12:00:00", timezone)

    quality = cli._input_quality(events, reference_time, timezone)

    assert quality == {
        "input_row_count": 2,
        "invalid_row_count": 0,
        "future_event_count": 1,
        "valid_event_count": 1,
        "filtered_row_count": 1,
    }
