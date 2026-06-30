from __future__ import annotations

from datetime import datetime, timedelta
from math import exp, log1p

import pandas as pd
import pytest

from user_portrait import ProfileConfig, build_user_profiles


NOW = datetime(2026, 3, 31, 12, 0, 0)


def _row(user_id: str, action_name: str, when: datetime) -> dict[str, object]:
    return {"user_id": user_id, "action_name": action_name, "event_time": when}


def test_recent_tasks_keep_latest_three_events_in_order() -> None:
    rows = [
        _row("u1", "older", NOW - timedelta(days=3)),
        _row("u1", "latest", NOW - timedelta(minutes=1)),
        _row("u1", "second", NOW - timedelta(minutes=2)),
        _row("u1", "third", NOW - timedelta(minutes=3)),
        _row("u1", "fourth", NOW - timedelta(minutes=4)),
    ]

    profiles = build_user_profiles(pd.DataFrame(rows), now=NOW)
    recent = profiles["recent_tasks"]

    assert recent["action_name"].tolist() == ["latest", "second", "third"]
    assert recent["rank"].tolist() == [1, 2, 3]
    assert set(recent.columns) == {
        "user_id",
        "action_name",
        "event_time",
        "rank",
        "user_activity_level",
    }


def test_recent_tasks_keep_repeated_actions() -> None:
    rows = [
        _row("u1", "查余额", NOW - timedelta(minutes=1)),
        _row("u1", "查余额", NOW - timedelta(minutes=2)),
        _row("u1", "查余额", NOW - timedelta(minutes=3)),
        _row("u1", "转账", NOW - timedelta(minutes=4)),
    ]

    profiles = build_user_profiles(pd.DataFrame(rows), now=NOW)
    recent = profiles["recent_tasks"]

    assert recent["action_name"].tolist() == ["查余额", "查余额", "查余额"]


def test_frequent_tasks_rank_all_history_and_calculate_metrics() -> None:
    rows = [
        _row("u1", "transfer", NOW - timedelta(days=30)),
        _row("u1", "transfer", NOW - timedelta(days=20)),
        _row("u1", "transfer", NOW - timedelta(days=10)),
        _row("u1", "balance", NOW - timedelta(days=5, hours=2)),
        _row("u1", "balance", NOW - timedelta(days=5, hours=1)),
        _row("u1", "balance", NOW - timedelta(days=5)),
        _row("u1", "gold", NOW - timedelta(days=2)),
        _row("u1", "gold", NOW - timedelta(days=1)),
        _row("u1", "bill", NOW - timedelta(hours=1)),
    ]

    profiles = build_user_profiles(
        pd.DataFrame(rows), now=NOW, config=ProfileConfig(frequent_task_top_k=3)
    )
    frequent = profiles["frequent_tasks"]

    assert frequent["action_name"].tolist() == ["transfer", "balance", "gold"]
    assert frequent["rank"].tolist() == [1, 2, 3]
    assert frequent["event_count"].tolist() == [3, 3, 2]
    assert frequent["active_days"].tolist() == [3, 1, 2]
    transfer = frequent.iloc[0]
    assert transfer["user_total_event_count"] == 9
    assert transfer["frequency_share"] == pytest.approx(3 / 9, abs=1e-6)
    assert transfer["observation_days"] == pytest.approx(30.0)
    assert transfer["avg_monthly_count"] == pytest.approx(3.0)


def test_frequent_tasks_allow_periodic_and_recent_high_frequency_overlap() -> None:
    rows = [
        *[
            _row("periodic", "weekly_transfer", NOW - timedelta(days=day))
            for day in [0, 7, 14, 21, 28]
        ],
        *[
            _row("burst", "buy_gold", NOW - timedelta(hours=hour))
            for hour in [1, 2, 3, 4]
        ],
    ]

    profiles = build_user_profiles(pd.DataFrame(rows), now=NOW)
    frequent_pairs = set(
        profiles["frequent_tasks"][["user_id", "action_name"]].itertuples(
            index=False, name=None
        )
    )

    assert ("periodic", "weekly_transfer") in frequent_pairs
    assert ("burst", "buy_gold") in frequent_pairs
    assert "weekly_transfer" in profiles["periodic_tasks"]["action_name"].tolist()
    assert "buy_gold" in profiles["recent_high_freq_tasks"]["action_name"].tolist()


def test_periodic_tasks_detect_weekly_and_monthly_behaviors() -> None:
    rows = []
    for day in [3, 10, 17, 24, 31]:
        rows.append(_row("u1", "给朋友转账", datetime(2026, 3, day, 18, 0, 0)))
    for month in [1, 2, 3]:
        rows.append(_row("u2", "还信用卡", datetime(2026, month, 25, 9, 0, 0)))

    profiles = build_user_profiles(pd.DataFrame(rows), now=NOW)
    periodic = profiles["periodic_tasks"]

    weekly = periodic[
        (periodic["user_id"] == "u1") & (periodic["action_name"] == "给朋友转账")
    ].iloc[0]
    monthly = periodic[
        (periodic["user_id"] == "u2") & (periodic["action_name"] == "还信用卡")
    ].iloc[0]

    assert weekly["period_type"] == "weekly"
    assert monthly["period_type"] == "monthly"
    assert monthly["period_value"] == "day≈25"
    assert monthly["next_expected_time"] == pd.Timestamp("2026-04-25 09:00:00")
    assert monthly["preferred_time_window"] == "morning"


def test_short_term_burst_is_high_frequency_not_periodic() -> None:
    rows = []
    for day in range(1, 8):
        rows.append(_row("u1", "查账单", NOW - timedelta(days=day * 10)))
    for hour in [9, 10, 11, 12]:
        rows.append(_row("u1", "买黄金", datetime(2026, 3, 30, hour, 0, 0)))

    profiles = build_user_profiles(pd.DataFrame(rows), now=NOW)
    periodic = profiles["periodic_tasks"]
    high_freq = profiles["recent_high_freq_tasks"]

    assert "买黄金" not in periodic["action_name"].tolist()
    assert "买黄金" in high_freq["action_name"].tolist()


def test_time_period_tasks_detect_irregular_morning_preference() -> None:
    rows = [
        _row("u1", "check_market", datetime(2026, 1, 3, 9, 0, 0)),
        _row("u1", "check_market", datetime(2026, 1, 11, 10, 0, 0)),
        _row("u1", "check_market", datetime(2026, 2, 2, 8, 30, 0)),
        _row("u1", "check_market", datetime(2026, 2, 21, 11, 0, 0)),
        _row("u1", "check_market", datetime(2026, 3, 18, 9, 15, 0)),
        _row("u1", "check_market", datetime(2026, 3, 20, 20, 0, 0)),
    ]

    profiles = build_user_profiles(pd.DataFrame(rows), now=NOW)
    time_period = profiles["time_period_tasks"]
    task = time_period[
        (time_period["user_id"] == "u1")
        & (time_period["action_name"] == "check_market")
    ].iloc[0]

    assert task["time_period"] == "morning"
    assert task["period_event_count"] == 5
    assert task["total_event_count"] == 6
    assert task["period_share"] == pytest.approx(5 / 6, abs=1e-6)
    assert task["active_days"] == 5
    assert task["rank"] == 1


def test_time_period_tasks_require_count_days_and_share() -> None:
    rows = [
        *[
            _row("u1", "low_count", datetime(2026, 3, day, 9, 0, 0))
            for day in [1, 9, 20]
        ],
        *[
            _row("u1", "low_days", datetime(2026, 3, 1, hour, 0, 0))
            for hour in [8, 9]
        ],
        *[
            _row("u1", "low_days", datetime(2026, 3, 2, hour, 0, 0))
            for hour in [10, 11]
        ],
        *[
            _row("u1", "low_share", datetime(2026, 2, day, 9, 0, 0))
            for day in [1, 5, 9, 13]
        ],
        *[
            _row("u1", "low_share", datetime(2026, 2, day, 20, 0, 0))
            for day in [17, 21, 25]
        ],
    ]

    profiles = build_user_profiles(pd.DataFrame(rows), now=NOW)
    time_period = profiles["time_period_tasks"]

    assert "low_count" not in time_period["action_name"].tolist()
    assert "low_days" not in time_period["action_name"].tolist()
    assert "low_share" not in time_period["action_name"].tolist()


def test_time_period_tasks_can_overlap_other_task_types() -> None:
    rows = [
        *[
            _row("periodic", "weekly_transfer", NOW - timedelta(days=day))
            for day in [0, 7, 14, 21, 28]
        ],
        *[
            _row("burst", "buy_gold", datetime(2026, 3, day, 9, 0, 0))
            for day in [26, 27, 28, 29, 30]
        ],
    ]

    profiles = build_user_profiles(pd.DataFrame(rows), now=NOW)
    time_period_pairs = set(
        profiles["time_period_tasks"][["user_id", "action_name"]].itertuples(
            index=False, name=None
        )
    )

    assert ("periodic", "weekly_transfer") in time_period_pairs
    assert ("burst", "buy_gold") in time_period_pairs
    assert "weekly_transfer" in profiles["periodic_tasks"]["action_name"].tolist()
    assert "buy_gold" in profiles["recent_high_freq_tasks"]["action_name"].tolist()
    assert {"weekly_transfer", "buy_gold"}.issubset(
        set(profiles["frequent_tasks"]["action_name"])
    )


def test_personal_behavior_sequences_detect_repeated_paths() -> None:
    rows = []
    for day in [20, 21, 22]:
        base = datetime(2026, 3, day, 10, 0, 0)
        rows.extend(
            [
                _row("u1", "查余额", base),
                _row("u1", "转账", base + timedelta(minutes=2)),
                _row("u1", "查交易结果", base + timedelta(minutes=4)),
            ]
        )

    profiles = build_user_profiles(pd.DataFrame(rows), now=NOW)
    sequences = profiles["personal_behavior_sequences"]

    assert "查余额 -> 转账 -> 查交易结果" in sequences["sequence"].tolist()
    assert "查余额 -> 转账" not in sequences["sequence"].tolist()
    assert {"sequence_length", "sequence_score"}.issubset(sequences.columns)
    assert sequences["sequence_length"].tolist() == [3]
    assert sequences["next_action_candidate"].tolist() == ["查交易结果"]


def test_two_step_sequences_need_stronger_evidence() -> None:
    rows = []
    for day in [20, 21]:
        base = datetime(2026, 3, day, 10, 0, 0)
        rows.extend([_row("u1", "A", base), _row("u1", "B", base + timedelta(minutes=2))])

    profiles = build_user_profiles(pd.DataFrame(rows), now=NOW)
    sequences = profiles["personal_behavior_sequences"]

    assert sequences.empty


def test_short_sequence_is_kept_when_support_is_much_higher_than_long_sequence() -> None:
    rows = []
    for day in [19, 20, 21]:
        base = datetime(2026, 3, day, 10, 0, 0)
        rows.extend(
            [
                _row("u1", "A", base),
                _row("u1", "B", base + timedelta(minutes=2)),
                _row("u1", "C", base + timedelta(minutes=4)),
            ]
        )
    for day in [22, 23]:
        base = datetime(2026, 3, day, 10, 0, 0)
        rows.extend([_row("u1", "A", base), _row("u1", "B", base + timedelta(minutes=2))])
    rows.extend(
        [
            _row("noise1", "X", datetime(2026, 3, 1, 8, 0, 0)),
            _row("noise2", "Y", datetime(2026, 3, 1, 9, 0, 0)),
        ]
    )

    profiles = build_user_profiles(pd.DataFrame(rows), now=NOW)
    sequences = profiles["personal_behavior_sequences"]

    assert "A -> B -> C" in sequences["sequence"].tolist()
    assert "A -> B" in sequences["sequence"].tolist()


def test_sequence_confidence_counts_prefixes_that_end_the_session() -> None:
    rows = []
    for day in [20, 21]:
        base = datetime(2026, 3, day, 10, 0, 0)
        rows.extend(
            [
                _row("u1", "A", base),
                _row("u1", "B", base + timedelta(minutes=2)),
                _row("u1", "C", base + timedelta(minutes=4)),
            ]
        )
    for day in [22, 23, 24]:
        base = datetime(2026, 3, day, 10, 0, 0)
        rows.extend(
            [_row("u1", "A", base), _row("u1", "B", base + timedelta(minutes=2))]
        )
    rows.extend(
        [
            _row("noise1", "X", datetime(2026, 3, 1, 8, 0, 0)),
            _row("noise2", "Y", datetime(2026, 3, 1, 9, 0, 0)),
        ]
    )

    profiles = build_user_profiles(
        pd.DataFrame(rows),
        now=NOW,
        config=ProfileConfig(min_long_sequence_confidence=0.0),
    )
    sequences = profiles["personal_behavior_sequences"]
    chain = sequences[sequences["sequence"] == "A -> B -> C"].iloc[0]

    assert chain["support_count"] == 2
    assert chain["transition_confidence"] == 0.4


def test_sequence_recency_uses_profile_reference_time() -> None:
    rows = []
    for day in [1, 2, 3]:
        base = datetime(2026, 1, day, 10, 0, 0)
        rows.extend(
            [_row("u1", "A", base), _row("u1", "B", base + timedelta(minutes=2))]
        )

    profiles = build_user_profiles(pd.DataFrame(rows), now=NOW)
    sequence = profiles["personal_behavior_sequences"].iloc[0]
    age_days = (pd.Timestamp(NOW) - pd.Timestamp("2026-01-03 10:02:00")).total_seconds() / 86400
    expected_score = round(log1p(3) * exp(-age_days / 30), 6)

    assert sequence["sequence"] == "A -> B"
    assert sequence["sequence_score"] == pytest.approx(expected_score, abs=1e-6)


def test_behavior_sequences_are_limited_to_top_k_per_user() -> None:
    rows = []
    chains = [
        ("A1", "B1", "C1"),
        ("A2", "B2", "C2"),
        ("A3", "B3", "C3"),
        ("A4", "B4", "C4"),
        ("A5", "B5", "C5"),
        ("A6", "B6", "C6"),
    ]
    for chain_index, chain in enumerate(chains):
        for repeat in range(3):
            base = datetime(2026, 3, 1 + chain_index * 4 + repeat, 10, 0, 0)
            rows.extend(
                [
                    _row("u1", chain[0], base),
                    _row("u1", chain[1], base + timedelta(minutes=2)),
                    _row("u1", chain[2], base + timedelta(minutes=4)),
                ]
            )

    profiles = build_user_profiles(
        pd.DataFrame(rows), now=NOW, config=ProfileConfig(sequence_top_k=3)
    )
    sequences = profiles["personal_behavior_sequences"]

    assert len(sequences[sequences["user_id"] == "u1"]) == 3
