from __future__ import annotations

import hashlib

from user_portrait import build_user_profiles
from user_portrait.sample_data import (
    FILLER_ACTIONS,
    HIGH_FREQ_ACTIONS,
    NOW,
    SampleDataConfig,
    TIME_PERIOD_ACTIONS,
    generate_sample_dataset,
)


def _precision_recall(pred, truth, keys):
    pred_set = set(pred[keys].itertuples(index=False, name=None))
    truth_set = set(truth[keys].itertuples(index=False, name=None))
    true_positive = len(pred_set & truth_set)
    precision = true_positive / len(pred_set) if pred_set else 0.0
    recall = true_positive / len(truth_set) if truth_set else 0.0
    return precision, recall


def _pairs(frame, columns):
    return set(frame[columns].itertuples(index=False, name=None))


def test_sample_dataset_shape_and_ground_truth_coverage() -> None:
    frames = generate_sample_dataset(SampleDataConfig())
    events = frames["sample_events"]
    periodic = frames["sample_ground_truth_periodic"]
    periodic_negative = frames["sample_ground_truth_periodic_negative"]
    high_freq = frames["sample_ground_truth_high_freq"]
    high_freq_negative = frames["sample_ground_truth_high_freq_negative"]
    time_period = frames["sample_ground_truth_time_period"]
    time_period_negative = frames["sample_ground_truth_time_period_negative"]
    sequences = frames["sample_ground_truth_sequences"]
    recent = frames["sample_ground_truth_recent"]
    frequent = frames["sample_ground_truth_frequent"]

    assert events["user_id"].nunique() == 240
    assert 18000 <= len(events) <= 23000
    assert (events["event_time"] <= NOW).all()
    assert set(events.columns) == {"user_id", "action_name", "event_time"}
    assert set(periodic["period_type"].unique()) == {"daily", "weekly", "monthly", "interval"}
    assert (periodic["period_type"].value_counts() >= 20).all()
    assert 130 <= len(periodic) <= 150
    assert set(periodic["pattern_strength"].unique()) == {"strong", "weak"}
    assert 120 <= len(periodic_negative) <= 180
    assert 90 <= len(high_freq) <= 110
    assert 80 <= len(high_freq_negative) <= 120
    assert 70 <= len(time_period) <= 90
    assert 45 <= len(time_period_negative) <= 70
    assert set(time_period["time_period"].unique()) == {
        "early_morning",
        "morning",
        "afternoon",
        "evening",
    }
    assert len(sequences) >= 150
    assert set(sequences["sequence_length"].unique()) == {2, 3, 4, 5}
    assert set(recent.columns) == {
        "user_id",
        "action_name",
        "event_time",
        "expected_rank",
        "expected_activity_level",
    }
    assert recent.groupby("user_id").size().max() == 3
    assert set(recent["expected_rank"].unique()) == {1, 2, 3}
    assert set(recent["expected_activity_level"].unique()) == {"high", "middle", "low"}
    assert frequent["user_id"].nunique() == 240
    assert frequent.groupby("user_id").size().max() == 5
    assert frequent["expected_rank"].between(1, 5).all()
    assert set(frequent.columns) == {
        "user_id",
        "action_name",
        "expected_event_count",
        "expected_active_days",
        "expected_rank",
    }
    assert set(FILLER_ACTIONS).isdisjoint(HIGH_FREQ_ACTIONS)
    assert set(TIME_PERIOD_ACTIONS).isdisjoint(FILLER_ACTIONS)
    assert set(TIME_PERIOD_ACTIONS).isdisjoint(HIGH_FREQ_ACTIONS)


def test_sample_dataset_contains_mixed_users() -> None:
    frames = generate_sample_dataset(SampleDataConfig())
    periodic = frames["sample_ground_truth_periodic"]
    high_freq = frames["sample_ground_truth_high_freq"]
    sequences = frames["sample_ground_truth_sequences"]

    periodic_users = set(periodic["user_id"])
    high_freq_users = set(high_freq["user_id"])
    sequence_users = set(sequences["user_id"])

    assert periodic_users & high_freq_users
    assert periodic_users & sequence_users
    assert high_freq_users & sequence_users
    assert len(periodic_users | high_freq_users | sequence_users) < (
        len(periodic_users) + len(high_freq_users) + len(sequence_users)
    )


def test_sample_dataset_excludes_page_or_system_actions() -> None:
    frames = generate_sample_dataset(SampleDataConfig())
    events = frames["sample_events"]
    sequences = frames["sample_ground_truth_sequences"]

    forbidden_actions = {
        "登录",
        "首页",
        "搜索",
        "消息中心",
        "设置",
        "安全中心",
        "扫码结果",
        "电子回单",
        "搜索理财",
        "理财首页",
        "信用卡首页",
        "保险首页",
    }
    event_actions = set(events["action_name"])
    sequence_text = "\n".join(sequences["sequence"].astype(str))

    assert event_actions.isdisjoint(forbidden_actions)
    assert not any(action in sequence_text for action in forbidden_actions)
    assert {"查交易结果", "查还款结果"}.issubset(event_actions)
    assert "查交易结果" in sequence_text
    assert "查还款结果" in sequence_text


def test_sample_dataset_runs_through_portrait_builder() -> None:
    frames = generate_sample_dataset(SampleDataConfig())
    profiles = build_user_profiles(frames["sample_events"], now=NOW)

    activity = profiles["activity_levels"]
    recent = profiles["recent_tasks"]
    periodic = profiles["periodic_tasks"]
    high_freq = profiles["recent_high_freq_tasks"]
    frequent = profiles["frequent_tasks"]
    time_period = profiles["time_period_tasks"]
    sequences = profiles["personal_behavior_sequences"]

    assert set(activity["user_activity_level"].unique()) == {"high", "middle", "low"}
    assert recent["user_id"].nunique() == 240
    assert set(recent.columns) == {
        "user_id",
        "action_name",
        "event_time",
        "rank",
        "user_activity_level",
    }
    assert recent.groupby("user_id").size().max() == 3
    assert set(recent["rank"].unique()) == {1, 2, 3}
    assert {"daily", "weekly", "monthly", "interval"}.issubset(
        set(periodic["period_type"].unique())
    )
    assert 90 <= len(high_freq) <= 130
    high_freq_digest = hashlib.sha256(
        high_freq.to_csv(index=False).encode("utf-8")
    ).hexdigest()
    assert high_freq_digest == "229ff92f856f7bf6f697fba2ed918a029c8c12030dc728fe65ff7852e0319332"
    assert frequent["user_id"].nunique() == 240
    assert frequent.groupby("user_id").size().max() == 5
    assert set(frequent.columns) == {
        "user_id",
        "action_name",
        "event_count",
        "active_days",
        "first_time",
        "last_time",
        "user_total_event_count",
        "frequency_share",
        "observation_days",
        "avg_monthly_count",
        "rank",
    }
    assert set(time_period.columns) == {
        "user_id",
        "action_name",
        "time_period",
        "period_event_count",
        "total_event_count",
        "period_share",
        "active_days",
        "first_time",
        "last_time",
        "time_period_score",
        "rank",
    }
    assert {"early_morning", "morning", "afternoon", "evening"}.issubset(
        set(time_period["time_period"].unique())
    )
    assert len(sequences) >= 150

    expected_frequent = set(
        frames["sample_ground_truth_frequent"][
            [
                "user_id",
                "action_name",
                "expected_event_count",
                "expected_active_days",
                "expected_rank",
            ]
        ].itertuples(index=False, name=None)
    )
    actual_frequent = set(
        frequent[
            ["user_id", "action_name", "event_count", "active_days", "rank"]
        ].itertuples(index=False, name=None)
    )
    assert actual_frequent == expected_frequent

    periodic_precision, periodic_recall = _precision_recall(
        periodic,
        frames["sample_ground_truth_periodic"],
        ["user_id", "action_name", "period_type"],
    )
    high_freq_precision, high_freq_recall = _precision_recall(
        high_freq,
        frames["sample_ground_truth_high_freq"],
        ["user_id", "action_name"],
    )
    _, time_period_recall = _precision_recall(
        time_period,
        frames["sample_ground_truth_time_period"],
        ["user_id", "action_name", "time_period"],
    )

    assert periodic_precision >= 0.70
    assert periodic_recall >= 0.75
    assert high_freq_precision >= 0.80
    assert high_freq_recall >= 0.85
    assert time_period_recall >= 0.85

    matched_monthly = frames["sample_ground_truth_periodic"].merge(
        periodic,
        on=["user_id", "action_name", "period_type"],
        suffixes=("_truth", "_actual"),
    )
    matched_monthly = matched_monthly[matched_monthly["period_type"] == "monthly"]
    assert (
        matched_monthly["next_expected_time_truth"]
        == matched_monthly["next_expected_time_actual"]
    ).all()

    period_negative_pairs = set(
        frames["sample_ground_truth_periodic_negative"][
            ["user_id", "action_name"]
        ].itertuples(index=False, name=None)
    )
    period_pred_pairs = set(
        periodic[["user_id", "action_name"]].itertuples(index=False, name=None)
    )
    high_freq_negative_pairs = set(
        frames["sample_ground_truth_high_freq_negative"][
            ["user_id", "action_name"]
        ].itertuples(index=False, name=None)
    )
    high_freq_pred_pairs = set(
        high_freq[["user_id", "action_name"]].itertuples(index=False, name=None)
    )
    time_period_negative_pairs = set(
        frames["sample_ground_truth_time_period_negative"][
            ["user_id", "action_name"]
        ].itertuples(index=False, name=None)
    )
    time_period_pred_pairs = set(
        time_period[["user_id", "action_name"]].itertuples(index=False, name=None)
    )

    assert len(period_negative_pairs & period_pred_pairs) / len(period_negative_pairs) <= 0.20
    assert (
        len(high_freq_negative_pairs & high_freq_pred_pairs) / len(high_freq_negative_pairs)
        <= 0.15
    )
    assert (
        len(time_period_negative_pairs & time_period_pred_pairs)
        / len(time_period_negative_pairs)
        <= 0.15
    )


def test_high_frequency_ground_truth_is_not_labeled_periodic_by_profile_builder() -> None:
    frames = generate_sample_dataset(SampleDataConfig())
    profiles = build_user_profiles(frames["sample_events"], now=NOW)

    high_freq_pairs = set(
        frames["sample_ground_truth_high_freq"][["user_id", "action_name"]].itertuples(
            index=False, name=None
        )
    )
    periodic_pairs = set(
        profiles["periodic_tasks"][["user_id", "action_name"]].itertuples(
            index=False, name=None
        )
    )

    assert len(high_freq_pairs & periodic_pairs) == 0


def test_periodic_detection_is_stable_across_sample_seeds() -> None:
    seeds = [7, 17, 42, 99, 123, 2026, 20260624]

    for seed in seeds:
        frames = generate_sample_dataset(SampleDataConfig(seed=seed))
        profiles = build_user_profiles(frames["sample_events"], now=NOW)
        periodic = profiles["periodic_tasks"]
        high_freq = profiles["recent_high_freq_tasks"]

        periodic_precision, periodic_recall = _precision_recall(
            periodic,
            frames["sample_ground_truth_periodic"],
            ["user_id", "action_name", "period_type"],
        )
        high_freq_precision, high_freq_recall = _precision_recall(
            high_freq,
            frames["sample_ground_truth_high_freq"],
            ["user_id", "action_name"],
        )

        periodic_negative_pairs = _pairs(
            frames["sample_ground_truth_periodic_negative"],
            ["user_id", "action_name"],
        )
        periodic_pairs = _pairs(periodic, ["user_id", "action_name"])
        high_freq_pairs = _pairs(high_freq, ["user_id", "action_name"])
        periodic_false_positive_rate = len(
            periodic_negative_pairs & periodic_pairs
        ) / len(periodic_negative_pairs)
        periodic_type_counts = periodic["period_type"].value_counts().to_dict()
        diagnostics = (
            f"seed={seed}, periodic_precision={periodic_precision:.3f}, "
            f"periodic_recall={periodic_recall:.3f}, "
            f"periodic_negative_fp_rate={periodic_false_positive_rate:.3f}, "
            f"high_freq_precision={high_freq_precision:.3f}, "
            f"high_freq_recall={high_freq_recall:.3f}, "
            f"periodic_rows={len(periodic)}, high_freq_rows={len(high_freq)}, "
            f"periodic_type_counts={periodic_type_counts}"
        )

        assert periodic_precision >= 0.75, diagnostics
        assert periodic_recall >= 0.80, diagnostics
        assert periodic_false_positive_rate <= 0.02, diagnostics
        assert set(periodic_type_counts) == {"daily", "weekly", "monthly", "interval"}
        assert high_freq_recall == 1.0, diagnostics
        assert len(periodic_pairs & high_freq_pairs) == 0, diagnostics
