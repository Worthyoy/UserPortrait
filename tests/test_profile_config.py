from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
from io import StringIO
import json

import pandas as pd
import pytest

from user_portrait import ProfileConfig, build_user_profiles
from user_portrait import cli


NOW = datetime(2026, 3, 31, 12, 0, 0)


def _row(user_id: str, action_name: str, when: datetime) -> dict[str, object]:
    return {"user_id": user_id, "action_name": action_name, "event_time": when}


@pytest.mark.parametrize(
    ("overrides", "field_name"),
    [
        ({"high_active_percentile": 1.1}, "high_active_percentile"),
        ({"daily_max_avg_interval_days": 0}, "daily_max_avg_interval_days"),
        ({"high_freq_recent_window_days": 0}, "high_freq_recent_window_days"),
        (
            {
                "high_freq_recent_window_days": 7,
                "high_freq_baseline_window_days": 7,
            },
            "high_freq_baseline_window_days",
        ),
        ({"sequence_min_len": 6, "sequence_max_len": 5}, "sequence_min_len"),
        (
            {"period_cv_weight": 0.7, "period_concentration_weight": 0.4},
            "period_cv_weight",
        ),
        ({"sequence_recency_decay_days": "30"}, "sequence_recency_decay_days"),
    ],
)
def test_profile_config_rejects_invalid_values(
    overrides: dict[str, object], field_name: str
) -> None:
    with pytest.raises(ValueError, match=field_name):
        ProfileConfig(**overrides)


def test_profile_config_contains_all_centralized_thresholds() -> None:
    values = asdict(ProfileConfig())

    expected = {
        "daily_max_avg_interval_days",
        "weekly_anchor_tolerance_days",
        "monthly_anchor_tolerance_days",
        "monthly_min_span_days",
        "interval_min_occurrences",
        "interval_min_avg_days",
        "interval_min_span_cycles",
        "long_interval_priority_days",
        "period_cv_weight",
        "period_concentration_weight",
        "daily_concentration_prior",
        "interval_concentration_prior",
        "high_freq_min_count",
        "high_freq_recent_window_days",
        "high_freq_low_activity_window_days",
        "high_freq_baseline_window_days",
        "high_freq_low_activity_baseline_window_days",
        "high_freq_baseline_floor",
        "high_freq_trend_lift_cap",
        "sequence_global_user_ratio_threshold",
        "sequence_global_min_confidence",
        "sequence_recency_decay_days",
        "sequence_pair_length_weight",
        "sequence_triple_length_weight",
        "sequence_long_length_weight",
    }
    removed = {
        "daily_min_occurrences",
        "daily_min_span_days",
        "high_recent_window_days",
        "low_recent_fallback_events",
        "low_recent_window_days",
        "mid_recent_window_days",
        "min_sequence_confidence",
        "min_sequence_support",
        "recent_top_k",
    }

    assert expected <= values.keys()
    assert removed.isdisjoint(values)


def test_cli_migrates_legacy_high_frequency_count(monkeypatch) -> None:
    payload = json.dumps({"high_freq_recent_count_7d": 6})
    monkeypatch.setattr(cli.Path, "open", lambda *args, **kwargs: StringIO(payload))

    with pytest.warns(FutureWarning, match="high_freq_min_count"):
        config = cli._load_config("config.json")

    assert config.high_freq_min_count == 6


def test_cli_rejects_conflicting_legacy_high_frequency_count(monkeypatch) -> None:
    payload = json.dumps(
        {"high_freq_recent_count_7d": 6, "high_freq_min_count": 5}
    )
    monkeypatch.setattr(cli.Path, "open", lambda *args, **kwargs: StringIO(payload))

    with pytest.raises(ValueError, match="conflicting config fields"):
        cli._load_config("config.json")


def test_daily_average_interval_threshold_is_configurable() -> None:
    rows = [
        _row("u1", "daily_action", datetime(2026, 3, day, 9))
        for day in [28, 29, 31]
    ]

    default_periods = build_user_profiles(pd.DataFrame(rows), now=NOW)["periodic_tasks"]
    strict_periods = build_user_profiles(
        pd.DataFrame(rows),
        now=NOW,
        config=ProfileConfig(daily_max_avg_interval_days=1.49),
    )["periodic_tasks"]

    assert "daily_action" in default_periods["action_name"].tolist()
    assert "daily_action" not in strict_periods["action_name"].tolist()


def test_high_frequency_min_count_is_configurable() -> None:
    rows = [
        _row("u1", "burst", NOW - timedelta(hours=offset))
        for offset in [1, 2, 3, 4]
    ]
    rows.append(_row("u1", "burst", NOW - timedelta(days=60)))

    default_tasks = build_user_profiles(pd.DataFrame(rows), now=NOW)[
        "recent_high_freq_tasks"
    ]
    strict_tasks = build_user_profiles(
        pd.DataFrame(rows),
        now=NOW,
        config=ProfileConfig(high_freq_min_count=5),
    )["recent_high_freq_tasks"]

    assert "burst" in default_tasks["action_name"].tolist()
    assert "burst" not in strict_tasks["action_name"].tolist()


def test_sequence_decay_and_length_weight_are_configurable() -> None:
    rows: list[dict[str, object]] = []
    for day in [1, 2, 3]:
        start = datetime(2026, 3, day, 10)
        rows.extend(
            [_row("u1", "A", start), _row("u1", "B", start + timedelta(minutes=2))]
        )

    default_sequence = build_user_profiles(pd.DataFrame(rows), now=NOW)[
        "personal_behavior_sequences"
    ].iloc[0]
    custom_sequence = build_user_profiles(
        pd.DataFrame(rows),
        now=NOW,
        config=ProfileConfig(
            sequence_recency_decay_days=60,
            sequence_pair_length_weight=2,
        ),
    )["personal_behavior_sequences"].iloc[0]

    assert custom_sequence["sequence_score"] > default_sequence["sequence_score"] * 2
