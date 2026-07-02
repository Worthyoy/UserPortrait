from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from math import exp, isclose, log1p, sqrt
from statistics import median
from typing import Any
import warnings
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd


REQUIRED_COLUMNS = {"user_id", "action_name", "event_time"}


@dataclass(frozen=True)
class ProfileConfig:
    business_timezone: str = "Asia/Shanghai"
    activity_window_days: int = 30
    high_active_days: int = 15
    high_active_percentile: float = 0.80
    mid_active_days: int = 4
    recent_event_top_k: int = 3
    min_period_occurrences: int = 3
    weekly_min_occurrences: int = 4
    weekly_min_span_days: int = 21
    monthly_min_months: int = 3
    daily_detection_window_days: int = 7
    weekly_detection_window_days: int = 56
    monthly_detection_window_days: int = 186
    interval_detection_recent_occurrences: int = 6
    strong_period_cv: float = 0.20
    calendar_concentration: float = 0.75
    interval_max_min_ratio: float = 1.25
    min_period_confidence: float = 0.70
    daily_max_avg_interval_days: float = 1.5
    weekly_anchor_tolerance_days: int = 1
    monthly_anchor_tolerance_days: int = 2
    monthly_min_span_days: int = 45
    interval_min_occurrences: int = 4
    interval_min_avg_days: float = 2.0
    interval_min_span_cycles: float = 3.0
    long_interval_priority_days: float = 7.5
    period_cv_weight: float = 0.65
    period_concentration_weight: float = 0.35
    daily_concentration_prior: float = 0.95
    interval_concentration_prior: float = 0.90
    session_gap_minutes: int = 30
    sequence_min_len: int = 2
    sequence_max_len: int = 5
    min_pair_sequence_support: int = 3
    min_pair_sequence_confidence: float = 0.75
    min_long_sequence_support: int = 2
    min_long_sequence_confidence: float = 0.60
    subsequence_support_margin: int = 2
    sequence_top_k: int = 5
    sequence_global_user_ratio_threshold: float = 0.80
    sequence_global_min_confidence: float = 0.75
    sequence_recency_decay_days: float = 30.0
    sequence_pair_length_weight: float = 1.0
    sequence_triple_length_weight: float = 1.2
    sequence_long_length_weight: float = 1.3
    high_freq_min_count: int = 4
    high_freq_lift: float = 3.0
    high_freq_recent_concentration: float = 0.60
    high_freq_recent_window_days: int = 7
    high_freq_low_activity_window_days: int = 30
    high_freq_baseline_window_days: int = 90
    high_freq_low_activity_baseline_window_days: int = 180
    high_freq_baseline_floor: float = 0.5
    high_freq_trend_lift_cap: float = 10.0
    frequent_task_top_k: int = 5
    time_period_min_count: int = 4
    time_period_min_active_days: int = 3
    time_period_min_share: float = 0.70
    time_period_top_k: int = 5

    def __post_init__(self) -> None:
        try:
            ZoneInfo(self.business_timezone)
        except (ZoneInfoNotFoundError, TypeError) as exc:
            raise ValueError(
                f"invalid business timezone: {self.business_timezone}"
            ) from exc

        positive_int_fields = (
            "activity_window_days",
            "recent_event_top_k",
            "min_period_occurrences",
            "weekly_min_occurrences",
            "monthly_min_months",
            "interval_min_occurrences",
            "session_gap_minutes",
            "sequence_min_len",
            "sequence_max_len",
            "min_pair_sequence_support",
            "min_long_sequence_support",
            "sequence_top_k",
            "high_freq_min_count",
            "high_freq_recent_window_days",
            "high_freq_low_activity_window_days",
            "high_freq_baseline_window_days",
            "high_freq_low_activity_baseline_window_days",
            "frequent_task_top_k",
            "time_period_min_count",
            "time_period_min_active_days",
            "time_period_top_k",
        )
        for name in positive_int_fields:
            self._validate_integer(name, minimum=1)

        nonnegative_int_fields = (
            "high_active_days",
            "mid_active_days",
            "weekly_min_span_days",
            "daily_detection_window_days",
            "weekly_detection_window_days",
            "monthly_detection_window_days",
            "interval_detection_recent_occurrences",
            "weekly_anchor_tolerance_days",
            "monthly_anchor_tolerance_days",
            "monthly_min_span_days",
            "subsequence_support_margin",
        )
        for name in nonnegative_int_fields:
            self._validate_integer(name, minimum=0)

        if self.sequence_min_len < 2:
            self._invalid("sequence_min_len", "must be at least 2")
        if self.sequence_min_len > self.sequence_max_len:
            self._invalid(
                "sequence_min_len",
                f"must not exceed sequence_max_len={self.sequence_max_len}",
            )
        if self.weekly_anchor_tolerance_days > 3:
            self._invalid("weekly_anchor_tolerance_days", "must not exceed 3")
        if self.high_freq_low_activity_window_days < self.high_freq_recent_window_days:
            self._invalid(
                "high_freq_low_activity_window_days",
                "must be at least high_freq_recent_window_days",
            )
        if self.high_freq_baseline_window_days <= self.high_freq_recent_window_days:
            self._invalid(
                "high_freq_baseline_window_days",
                "must exceed high_freq_recent_window_days",
            )
        if (
            self.high_freq_low_activity_baseline_window_days
            <= self.high_freq_low_activity_window_days
        ):
            self._invalid(
                "high_freq_low_activity_baseline_window_days",
                "must exceed high_freq_low_activity_window_days",
            )

        proportion_fields = (
            "high_active_percentile",
            "calendar_concentration",
            "min_period_confidence",
            "period_cv_weight",
            "period_concentration_weight",
            "daily_concentration_prior",
            "interval_concentration_prior",
            "min_pair_sequence_confidence",
            "min_long_sequence_confidence",
            "sequence_global_user_ratio_threshold",
            "sequence_global_min_confidence",
            "high_freq_recent_concentration",
            "time_period_min_share",
        )
        for name in proportion_fields:
            self._validate_number(name, minimum=0.0, maximum=1.0)

        positive_number_fields = (
            "daily_max_avg_interval_days",
            "interval_min_avg_days",
            "interval_min_span_cycles",
            "interval_max_min_ratio",
            "sequence_recency_decay_days",
            "sequence_pair_length_weight",
            "sequence_triple_length_weight",
            "sequence_long_length_weight",
            "high_freq_lift",
            "high_freq_baseline_floor",
            "high_freq_trend_lift_cap",
        )
        for name in positive_number_fields:
            self._validate_number(name, minimum=0.0, minimum_inclusive=False)

        self._validate_number("strong_period_cv", minimum=0.0)
        self._validate_number("long_interval_priority_days", minimum=0.0)
        if self.interval_max_min_ratio < 1.0:
            self._invalid("interval_max_min_ratio", "must be at least 1.0")
        if not isclose(
            self.period_cv_weight + self.period_concentration_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            self._invalid(
                "period_cv_weight",
                "period_cv_weight + period_concentration_weight must equal 1.0",
            )

    def _validate_integer(self, name: str, *, minimum: int) -> None:
        value = getattr(self, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            self._invalid(name, f"must be an integer >= {minimum}")

    def _validate_number(
        self,
        name: str,
        *,
        minimum: float,
        maximum: float | None = None,
        minimum_inclusive: bool = True,
    ) -> None:
        value = getattr(self, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self._invalid(name, "must be numeric")
        below_minimum = value < minimum if minimum_inclusive else value <= minimum
        if below_minimum or (maximum is not None and value > maximum):
            if maximum is not None:
                requirement = f"within [{minimum}, {maximum}]"
            elif minimum_inclusive:
                requirement = f">= {minimum}"
            else:
                requirement = f"> {minimum}"
            self._invalid(name, f"must be {requirement}")

    def _invalid(self, name: str, message: str) -> None:
        value = getattr(self, name)
        raise ValueError(f"invalid ProfileConfig.{name}={value!r}: {message}")


def build_user_profiles(
    events: pd.DataFrame,
    *,
    now: str | datetime | pd.Timestamp | None = None,
    config: ProfileConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build behavior portrait tables from user_id/action_name/event_time events."""

    cfg = config or ProfileConfig()
    business_timezone = _resolve_business_timezone(cfg.business_timezone)
    clean_events = _prepare_events(events, business_timezone)
    reference_time = _resolve_now(clean_events, now, business_timezone)
    clean_events = clean_events[clean_events["event_time"] <= reference_time].reset_index(
        drop=True
    )
    if clean_events.empty:
        raise ValueError("events is empty after filtering rows later than now")

    activity = _build_activity_levels(clean_events, reference_time, cfg)
    recent_tasks = _build_recent_tasks(clean_events, activity, reference_time, cfg)
    periodic_tasks = _build_periodic_tasks(clean_events, reference_time, cfg)
    recent_high_freq_tasks = _build_recent_high_freq_tasks(
        clean_events, activity, periodic_tasks, reference_time, cfg
    )
    frequent_tasks = _build_frequent_tasks(clean_events, reference_time, cfg)
    time_period_tasks = _build_time_period_tasks(clean_events, reference_time, cfg)
    personal_behavior_sequences = _build_behavior_sequences(
        clean_events, reference_time, cfg
    )

    return {
        "activity_levels": activity,
        "recent_tasks": recent_tasks,
        "periodic_tasks": periodic_tasks,
        "recent_high_freq_tasks": recent_high_freq_tasks,
        "frequent_tasks": frequent_tasks,
        "time_period_tasks": time_period_tasks,
        "personal_behavior_sequences": personal_behavior_sequences,
    }


def _prepare_events(events: pd.DataFrame, business_timezone: ZoneInfo) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(events.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"missing required columns: {missing_text}")

    df = events.loc[:, ["user_id", "action_name", "event_time"]].copy()
    df["event_time"] = _normalize_datetime_series(
        df["event_time"], business_timezone
    )
    df = df.dropna(subset=["user_id", "action_name", "event_time"])
    df["user_id"] = df["user_id"].astype(str)
    df["action_name"] = df["action_name"].astype(str)
    df = df.sort_values(["user_id", "event_time", "action_name"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("events is empty after dropping invalid rows")
    return df


def _resolve_now(
    events: pd.DataFrame,
    now: str | datetime | pd.Timestamp | None,
    business_timezone: ZoneInfo,
) -> pd.Timestamp:
    if now is None:
        return pd.Timestamp(events["event_time"].max())
    normalized = _normalize_timestamp(now, business_timezone)
    if pd.isna(normalized):
        raise ValueError(f"invalid profile reference time: {now!r}")
    return normalized


def _resolve_business_timezone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"invalid business timezone: {name!r}") from exc


def _normalize_datetime_series(
    values: pd.Series, business_timezone: ZoneInfo
) -> pd.Series:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    except (FutureWarning, ValueError):
        return values.map(
            lambda value: _normalize_timestamp(value, business_timezone)
        )
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return parsed.dt.tz_convert(business_timezone).dt.tz_localize(None)
    if pd.api.types.is_datetime64_dtype(parsed.dtype):
        return (
            parsed.dt.tz_localize(
                business_timezone, ambiguous="NaT", nonexistent="NaT"
            )
            .dt.tz_localize(None)
        )
    return values.map(
        lambda value: _normalize_timestamp(value, business_timezone)
    )


def _normalize_timestamp(value: Any, business_timezone: ZoneInfo) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return pd.NaT
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is None:
        try:
            localized = timestamp.tz_localize(
                business_timezone, ambiguous="raise", nonexistent="raise"
            )
        except (TypeError, ValueError):
            return pd.NaT
    else:
        localized = timestamp.tz_convert(business_timezone)
    return localized.tz_localize(None)


def _build_activity_levels(
    events: pd.DataFrame, now: pd.Timestamp, cfg: ProfileConfig
) -> pd.DataFrame:
    window_start = now - pd.Timedelta(days=cfg.activity_window_days)
    window_events = events[
        (events["event_time"] > window_start) & (events["event_time"] <= now)
    ].copy()
    window_events["event_date"] = window_events["event_time"].dt.date

    all_users = pd.DataFrame({"user_id": sorted(events["user_id"].unique())})
    stats = (
        window_events.groupby("user_id")
        .agg(event_count_30d=("event_time", "size"), active_days_30d=("event_date", "nunique"))
        .reset_index()
    )
    stats = all_users.merge(stats, on="user_id", how="left").fillna(0)
    stats["event_count_30d"] = stats["event_count_30d"].astype(int)
    stats["active_days_30d"] = stats["active_days_30d"].astype(int)
    active_event_counts = stats.loc[
        stats["event_count_30d"] > 0, "event_count_30d"
    ]
    p80 = (
        float(active_event_counts.quantile(cfg.high_active_percentile))
        if not active_event_counts.empty
        else float("inf")
    )

    def classify(row: pd.Series) -> str:
        if row["event_count_30d"] == 0:
            return "low"
        if row["active_days_30d"] >= cfg.high_active_days or row["event_count_30d"] >= p80:
            return "high"
        if row["active_days_30d"] >= cfg.mid_active_days:
            return "middle"
        return "low"

    stats["user_activity_level"] = stats.apply(classify, axis=1)
    return stats.sort_values("user_id").reset_index(drop=True)


def _build_recent_tasks(
    events: pd.DataFrame,
    activity: pd.DataFrame,
    now: pd.Timestamp,
    cfg: ProfileConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    activity_by_user = activity.set_index("user_id")["user_activity_level"].to_dict()

    for user_id, user_events in events.groupby("user_id", sort=False):
        level = activity_by_user[user_id]
        selected = user_events[user_events["event_time"] <= now].sort_values(
            ["event_time", "action_name"], ascending=[False, True]
        )
        selected = selected.head(cfg.recent_event_top_k)
        if selected.empty:
            continue

        for rank, item in enumerate(selected.itertuples(index=False), start=1):
            rows.append(
                {
                    "user_id": user_id,
                    "action_name": item.action_name,
                    "event_time": item.event_time,
                    "rank": rank,
                    "user_activity_level": level,
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return _empty_recent_tasks()
    return result.sort_values(["user_id", "rank"]).reset_index(drop=True)


def _build_periodic_tasks(
    events: pd.DataFrame, now: pd.Timestamp, cfg: ProfileConfig
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = events.groupby(["user_id", "action_name"], sort=False)
    eligible_pairs = grouped.size()
    eligible_pairs = eligible_pairs[eligible_pairs >= cfg.min_period_occurrences].index
    for user_id, action_name in eligible_pairs:
        group = grouped.get_group((user_id, action_name))
        candidate = _detect_period(group["event_time"].sort_values().tolist(), now, cfg)
        if candidate is None:
            continue
        rows.append({"user_id": user_id, "action_name": action_name, **candidate})

    if not rows:
        return _empty_periodic_tasks()
    return pd.DataFrame(rows).sort_values(
        ["user_id", "confidence", "action_name"], ascending=[True, False, True]
    ).reset_index(drop=True)


def _detect_period(
    times: list[pd.Timestamp], now: pd.Timestamp, cfg: ProfileConfig
) -> dict[str, Any] | None:
    if len(times) < cfg.min_period_occurrences:
        return None

    timestamps = sorted(pd.Timestamp(value) for value in times)
    dates = sorted({value.normalize() for value in timestamps})
    if len(dates) < cfg.min_period_occurrences:
        return None

    candidates: list[dict[str, Any]] = []
    daily_timestamps = _first_timestamp_per_active_date(
        _timestamps_in_recent_days(timestamps, now, cfg.daily_detection_window_days)
    )
    daily_dates = _unique_dates(daily_timestamps)
    daily_stats = _date_interval_stats(daily_dates)
    if (
        daily_stats is not None
        and len(daily_dates) >= cfg.min_period_occurrences
        and daily_dates[-1] == now.normalize()
        and daily_stats["avg_interval"] <= cfg.daily_max_avg_interval_days
    ):
        candidate = _period_candidate(
            "daily",
            "every 1 day",
            daily_dates[-1] + pd.Timedelta(days=1),
            daily_stats["cv"],
            cfg.daily_concentration_prior,
            (
                f"最近 {cfg.daily_detection_window_days} 天内 {len(daily_dates)} "
                f"个活跃日平均间隔 {daily_stats['avg_interval']:.1f} 天"
            ),
            cfg,
        )
        candidate["_timestamps"] = daily_timestamps
        candidates.append(candidate)

    weekly_timestamps = _first_timestamp_per_active_date(
        _timestamps_in_recent_days(timestamps, now, cfg.weekly_detection_window_days)
    )
    weekly_dates = _unique_dates(weekly_timestamps)
    weekly_stats = _date_interval_stats(weekly_dates)
    weekday_counts = Counter(value.weekday() for value in weekly_timestamps)
    if weekly_stats is not None and weekday_counts:
        weekday = max(weekday_counts, key=weekday_counts.get)
        concentration = weekday_counts[weekday] / len(weekly_timestamps)
        weekly_exact_timestamps = [
            value for value in weekly_timestamps if value.weekday() == weekday
        ]
        weekly_near_timestamps = [
            value
            for value in weekly_timestamps
            if _weekday_distance(value.weekday(), weekday)
            <= cfg.weekly_anchor_tolerance_days
        ]
        weekly_anchor_timestamps, weekly_anchor_dates, weekly_anchor_stats = (
            _best_period_dates_for_cv(
                [weekly_exact_timestamps, weekly_near_timestamps],
                min_occurrences=cfg.weekly_min_occurrences,
                min_span_days=cfg.weekly_min_span_days,
            )
        )
        if (
            weekly_anchor_stats is not None
            and concentration >= cfg.calendar_concentration
        ):
            candidate = _period_candidate(
                "weekly",
                f"weekday={weekday}",
                _next_weekday(weekly_anchor_dates[-1], weekday),
                weekly_anchor_stats["cv"],
                concentration,
                (
                    f"最近 {cfg.weekly_detection_window_days} 天内 "
                    f"{concentration:.0%} 的活跃日发生在星期 {weekday + 1}"
                ),
                cfg,
            )
            candidate["_timestamps"] = weekly_anchor_timestamps
            candidates.append(candidate)

    monthly_timestamps = _first_timestamp_per_active_date(
        _timestamps_in_recent_days(timestamps, now, cfg.monthly_detection_window_days)
    )
    monthly_dates = _unique_dates(monthly_timestamps)
    monthly_stats = _date_interval_stats(monthly_dates)
    month_anchor, month_concentration = (
        _month_day_anchor_values(
            [value.day for value in monthly_timestamps],
            cfg.monthly_anchor_tolerance_days,
        )
        if monthly_timestamps
        else (0, 0.0)
    )
    monthly_anchor_timestamps = [
        value
        for value in monthly_timestamps
        if abs(value.day - month_anchor) <= cfg.monthly_anchor_tolerance_days
    ]
    monthly_anchor_dates = _unique_dates(monthly_anchor_timestamps)
    monthly_anchor_stats = _date_interval_stats(monthly_anchor_dates)
    month_count = len({(value.year, value.month) for value in monthly_anchor_timestamps})
    if (
        monthly_anchor_stats is not None
        and month_count >= cfg.monthly_min_months
        and monthly_anchor_stats["span_days"] >= cfg.monthly_min_span_days
        and month_concentration >= cfg.calendar_concentration
    ):
        candidate = _period_candidate(
            "monthly",
            f"day≈{month_anchor}",
            _next_month_day(monthly_anchor_dates[-1], month_anchor),
            monthly_anchor_stats["cv"],
            month_concentration,
            (
                f"最近 {cfg.monthly_detection_window_days} 天内 "
                f"{month_concentration:.0%} 的活跃日集中在每月 "
                f"{month_anchor} 日前后"
            ),
            cfg,
        )
        candidate["_timestamps"] = monthly_anchor_timestamps
        candidates.append(candidate)

    interval_dates = (
        dates[-cfg.interval_detection_recent_occurrences :]
        if cfg.interval_detection_recent_occurrences > 0
        else []
    )
    interval_timestamps = _first_timestamp_per_active_date(
        _timestamps_for_dates(timestamps, interval_dates)
    )
    interval_stats = _date_interval_stats(interval_dates)
    if interval_stats is not None:
        min_interval = min(interval_stats["intervals"])
        max_interval = max(interval_stats["intervals"])
        interval_ratio = max_interval / min_interval if min_interval > 0 else 999.0
        if (
            len(interval_dates) >= cfg.interval_min_occurrences
            and len(interval_stats["intervals"]) >= cfg.interval_min_occurrences - 1
            and interval_stats["avg_interval"] >= cfg.interval_min_avg_days
            and interval_stats["span_days"]
            >= interval_stats["avg_interval"] * cfg.interval_min_span_cycles
            and interval_stats["cv"] <= cfg.strong_period_cv
            and interval_ratio <= cfg.interval_max_min_ratio
        ):
            interval_days = max(int(round(interval_stats["avg_interval"])), 1)
            candidate = _period_candidate(
                "interval",
                f"every {interval_days} days",
                interval_dates[-1] + pd.Timedelta(days=interval_days),
                interval_stats["cv"],
                cfg.interval_concentration_prior,
                (
                    f"最近 {len(interval_dates)} 个活跃日相邻行为间隔约 "
                    f"{interval_days} 天，CV={interval_stats['cv']:.2f}"
                ),
                cfg,
            )
            candidate["_timestamps"] = interval_timestamps
            candidates.append(candidate)

    if not candidates:
        return None

    viable_candidates = [
        item for item in candidates if item["confidence"] >= cfg.min_period_confidence
    ]
    if not viable_candidates:
        return None

    active_candidates: list[dict[str, Any]] = []
    for item in viable_candidates:
        finalized = dict(item)
        time_series = pd.Series(finalized.pop("_timestamps"))
        finalized["preferred_time_window"] = _preferred_time_window(time_series)
        finalized["next_expected_time"] = _apply_preferred_hour(
            finalized["next_expected_time"], time_series
        )
        if finalized["next_expected_time"] > now:
            active_candidates.append(finalized)
    if not active_candidates:
        return None

    monthly_candidates = [
        item for item in active_candidates if item["period_type"] == "monthly"
    ]
    interval_candidates = [
        item
        for item in active_candidates
        if item["period_type"] == "interval"
        and _interval_days_from_candidate(item) > cfg.long_interval_priority_days
    ]
    selected = (
        max(monthly_candidates, key=lambda item: item["confidence"])
        if monthly_candidates
        else max(interval_candidates, key=lambda item: item["confidence"])
        if interval_candidates
        else max(active_candidates, key=lambda item: item["confidence"])
    )
    return selected


def _timestamps_in_recent_days(
    timestamps: list[pd.Timestamp], now: pd.Timestamp, window_days: int
) -> list[pd.Timestamp]:
    if window_days <= 0:
        return []
    window_start = now - pd.Timedelta(days=window_days)
    return [value for value in timestamps if window_start < value <= now]


def _unique_dates(timestamps: list[pd.Timestamp]) -> list[pd.Timestamp]:
    return sorted({value.normalize() for value in timestamps})


def _first_timestamp_per_active_date(
    timestamps: list[pd.Timestamp],
) -> list[pd.Timestamp]:
    first_by_date: dict[pd.Timestamp, pd.Timestamp] = {}
    for value in sorted(pd.Timestamp(timestamp) for timestamp in timestamps):
        first_by_date.setdefault(value.normalize(), value)
    return list(first_by_date.values())


def _best_period_dates_for_cv(
    timestamp_options: list[list[pd.Timestamp]],
    *,
    min_occurrences: int,
    min_span_days: int,
) -> tuple[list[pd.Timestamp], list[pd.Timestamp], dict[str, Any] | None]:
    choices: list[tuple[list[pd.Timestamp], list[pd.Timestamp], dict[str, Any]]] = []
    for option in timestamp_options:
        dates = _unique_dates(option)
        stats = _date_interval_stats(dates)
        if (
            stats is not None
            and len(dates) >= min_occurrences
            and stats["span_days"] >= min_span_days
        ):
            choices.append((option, dates, stats))
    if not choices:
        return [], [], None
    return min(choices, key=lambda item: item[2]["cv"])


def _weekday_distance(value: int, target: int) -> int:
    distance = abs(value - target)
    return min(distance, 7 - distance)


def _timestamps_for_dates(
    timestamps: list[pd.Timestamp], dates: list[pd.Timestamp]
) -> list[pd.Timestamp]:
    date_set = set(dates)
    return [value for value in timestamps if value.normalize() in date_set]


def _date_interval_stats(dates: list[pd.Timestamp]) -> dict[str, Any] | None:
    if len(dates) < 2:
        return None
    intervals = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
    avg_interval = float(sum(intervals) / len(intervals))
    if avg_interval <= 0:
        return None
    variance = sum((value - avg_interval) ** 2 for value in intervals) / len(intervals)
    return {
        "span_days": max((dates[-1] - dates[0]).days, 0),
        "intervals": intervals,
        "avg_interval": avg_interval,
        "cv": float(sqrt(variance) / avg_interval),
    }


def _interval_days_from_candidate(candidate: dict[str, Any]) -> float:
    if candidate["period_type"] != "interval":
        return 0.0
    parts = str(candidate["period_value"]).split()
    if len(parts) < 2:
        return 0.0
    try:
        return float(parts[1])
    except ValueError:
        return 0.0


def _period_candidate(
    period_type: str,
    period_value: str,
    next_expected_time: pd.Timestamp,
    cv: float,
    concentration: float,
    evidence: str,
    cfg: ProfileConfig,
) -> dict[str, Any]:
    cv_score = max(0.0, min(1.0, 1.0 - cv))
    confidence = round(
        cfg.period_cv_weight * cv_score
        + cfg.period_concentration_weight * concentration,
        6,
    )
    return {
        "period_type": period_type,
        "period_value": period_value,
        "next_expected_time": next_expected_time,
        "confidence": confidence,
        "evidence": evidence,
    }


def _preferred_time_window(times: pd.Series) -> str:
    def bucket(hour: int) -> str:
        if 0 <= hour < 6:
            return "night"
        if 6 <= hour < 11:
            return "morning"
        if 11 <= hour < 14:
            return "noon"
        if 14 <= hour < 18:
            return "afternoon"
        return "evening"

    return str(times.dt.hour.map(bucket).value_counts().idxmax())


def _apply_preferred_hour(date_value: pd.Timestamp, times: pd.Series) -> pd.Timestamp:
    preferred_hour = int(times.dt.hour.mode().iloc[0])
    return pd.Timestamp(date_value).replace(hour=preferred_hour)


def _next_weekday(last_date: pd.Timestamp, weekday: int) -> pd.Timestamp:
    delta = (weekday - last_date.weekday()) % 7
    if delta == 0:
        delta = 7
    return last_date + pd.Timedelta(days=delta)


def _month_day_anchor_values(
    days: list[int], tolerance_days: int
) -> tuple[int, float]:
    median_day = float(median(days))
    concentrations = {
        day: sum(abs(value - day) <= tolerance_days for value in days) / len(days)
        for day in range(1, 32)
    }
    best_concentration = max(concentrations.values())
    candidate_days = [
        day
        for day, concentration in concentrations.items()
        if concentration == best_concentration
    ]
    best_day = min(candidate_days, key=lambda day: (abs(day - median_day), day))
    return best_day, float(best_concentration)


def _next_month_day(last_date: pd.Timestamp, target_day: int) -> pd.Timestamp:
    year = last_date.year
    month = last_date.month + 1
    if month > 12:
        year += 1
        month = 1
    days_in_month = pd.Timestamp(year=year, month=month, day=1).days_in_month
    return pd.Timestamp(year=year, month=month, day=min(target_day, days_in_month))


def _build_recent_high_freq_tasks(
    events: pd.DataFrame,
    activity: pd.DataFrame,
    periodic_tasks: pd.DataFrame,
    now: pd.Timestamp,
    cfg: ProfileConfig,
) -> pd.DataFrame:
    periodic_pairs = set()
    if not periodic_tasks.empty:
        periodic_pairs = set(
            periodic_tasks[["user_id", "action_name"]].itertuples(index=False, name=None)
        )

    activity_by_user = activity.set_index("user_id")["user_activity_level"].to_dict()
    rows: list[dict[str, Any]] = []
    for (user_id, action_name), group in events.groupby(["user_id", "action_name"]):
        if (user_id, action_name) in periodic_pairs:
            continue

        level = activity_by_user[user_id]
        recent_window = cfg.high_freq_recent_window_days
        low_activity_window = cfg.high_freq_low_activity_window_days
        recent_window_count = _count_between(
            group, now - pd.Timedelta(days=recent_window), now
        )
        low_activity_window_count = _count_between(
            group, now - pd.Timedelta(days=low_activity_window), now
        )
        recent_concentration = recent_window_count / max(low_activity_window_count, 1)
        recent_baseline = _baseline_average(
            group,
            now,
            recent_days=recent_window,
            baseline_days=cfg.high_freq_baseline_window_days,
        )
        recent_lift = recent_window_count / max(
            recent_baseline, cfg.high_freq_baseline_floor
        )

        selected_window = recent_window
        recent_count = recent_window_count
        baseline = recent_baseline
        lift = recent_lift

        if level == "low":
            low_activity_baseline = _baseline_average(
                group,
                now,
                recent_days=low_activity_window,
                baseline_days=cfg.high_freq_low_activity_baseline_window_days,
            )
            low_activity_lift = low_activity_window_count / max(
                low_activity_baseline, cfg.high_freq_baseline_floor
            )
            if (
                low_activity_window_count >= cfg.high_freq_min_count
                and low_activity_lift >= cfg.high_freq_lift
            ):
                selected_window = low_activity_window
                recent_count = low_activity_window_count
                baseline = low_activity_baseline
                lift = low_activity_lift

        if recent_count < cfg.high_freq_min_count or lift < cfg.high_freq_lift:
            continue
        if recent_concentration < cfg.high_freq_recent_concentration:
            continue

        rows.append(
            {
                "user_id": user_id,
                "action_name": action_name,
                "recent_count": int(recent_count),
                "baseline_count": round(float(baseline), 6),
                "lift": round(float(lift), 6),
                "trend_score": round(
                    log1p(recent_count) * min(lift, cfg.high_freq_trend_lift_cap),
                    6,
                ),
                "time_window": f"{selected_window}d",
            }
        )

    if not rows:
        return _empty_high_freq_tasks()
    return pd.DataFrame(rows).sort_values(
        ["user_id", "trend_score", "recent_count"], ascending=[True, False, False]
    ).reset_index(drop=True)


def _count_between(group: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> int:
    return int(((group["event_time"] > start) & (group["event_time"] <= end)).sum())


def _baseline_average(
    group: pd.DataFrame, now: pd.Timestamp, *, recent_days: int, baseline_days: int
) -> float:
    baseline_start = now - pd.Timedelta(days=baseline_days)
    baseline_end = now - pd.Timedelta(days=recent_days)
    count = _count_between(group, baseline_start, baseline_end)
    window_count = max((baseline_days - recent_days) / recent_days, 1.0)
    return count / window_count


def _build_frequent_tasks(
    events: pd.DataFrame, now: pd.Timestamp, cfg: ProfileConfig
) -> pd.DataFrame:
    user_stats = (
        events.groupby("user_id")
        .agg(
            user_total_event_count=("event_time", "size"),
            user_first_time=("event_time", "min"),
        )
        .reset_index()
    )
    task_stats = (
        events.assign(event_date=events["event_time"].dt.date)
        .groupby(["user_id", "action_name"])
        .agg(
            event_count=("event_time", "size"),
            active_days=("event_date", "nunique"),
            first_time=("event_time", "min"),
            last_time=("event_time", "max"),
        )
        .reset_index()
        .merge(user_stats, on="user_id", how="left", validate="many_to_one")
    )

    elapsed_days = (now - task_stats["user_first_time"]).dt.total_seconds() / 86400
    task_stats["observation_days"] = elapsed_days.clip(lower=1.0)
    task_stats["frequency_share"] = (
        task_stats["event_count"] / task_stats["user_total_event_count"]
    )
    task_stats["avg_monthly_count"] = (
        task_stats["event_count"] / task_stats["observation_days"] * 30
    )

    task_stats = task_stats.sort_values(
        ["user_id", "event_count", "active_days", "last_time", "action_name"],
        ascending=[True, False, False, False, True],
    )
    task_stats["rank"] = task_stats.groupby("user_id").cumcount() + 1
    task_stats = task_stats[task_stats["rank"] <= cfg.frequent_task_top_k].copy()
    task_stats["frequency_share"] = task_stats["frequency_share"].round(6)
    task_stats["observation_days"] = task_stats["observation_days"].round(6)
    task_stats["avg_monthly_count"] = task_stats["avg_monthly_count"].round(6)

    return task_stats[
        [
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
        ]
    ].reset_index(drop=True)


TIME_PERIOD_ORDER = {
    "early_morning": 0,
    "morning": 1,
    "afternoon": 2,
    "evening": 3,
}


def _build_time_period_tasks(
    events: pd.DataFrame, now: pd.Timestamp, cfg: ProfileConfig
) -> pd.DataFrame:
    if (
        cfg.time_period_min_count <= 0
        or cfg.time_period_min_active_days <= 0
        or cfg.time_period_min_share <= 0
        or cfg.time_period_top_k <= 0
    ):
        return _empty_time_period_tasks()

    period_events = events.copy()
    period_events["time_period"] = period_events["event_time"].dt.hour.map(
        _time_period_label
    )
    period_events["event_date"] = period_events["event_time"].dt.date

    totals = (
        period_events.groupby(["user_id", "action_name"])
        .agg(total_event_count=("event_time", "size"))
        .reset_index()
    )
    period_stats = (
        period_events.groupby(["user_id", "action_name", "time_period"])
        .agg(
            period_event_count=("event_time", "size"),
            active_days=("event_date", "nunique"),
            first_time=("event_time", "min"),
            last_time=("event_time", "max"),
        )
        .reset_index()
        .merge(totals, on=["user_id", "action_name"], how="left", validate="many_to_one")
    )
    period_stats["time_period_order"] = period_stats["time_period"].map(
        TIME_PERIOD_ORDER
    )
    period_stats = period_stats.sort_values(
        [
            "user_id",
            "action_name",
            "period_event_count",
            "active_days",
            "time_period_order",
        ],
        ascending=[True, True, False, False, True],
    )
    selected = period_stats.groupby(["user_id", "action_name"], sort=False).head(1).copy()
    selected["period_share"] = (
        selected["period_event_count"] / selected["total_event_count"]
    )
    selected = selected[
        (selected["period_event_count"] >= cfg.time_period_min_count)
        & (selected["active_days"] >= cfg.time_period_min_active_days)
        & (selected["period_share"] >= cfg.time_period_min_share)
    ].copy()
    if selected.empty:
        return _empty_time_period_tasks()

    selected["time_period_score"] = (
        selected["period_event_count"].map(log1p) * selected["period_share"]
    )
    selected["period_share"] = selected["period_share"].round(6)
    selected["time_period_score"] = selected["time_period_score"].round(6)
    selected = selected.sort_values(
        [
            "user_id",
            "time_period_score",
            "period_event_count",
            "active_days",
            "last_time",
            "action_name",
        ],
        ascending=[True, False, False, False, False, True],
    )
    selected["rank"] = selected.groupby("user_id").cumcount() + 1
    selected = selected[selected["rank"] <= cfg.time_period_top_k]

    return selected[
        [
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
        ]
    ].reset_index(drop=True)


def _time_period_label(hour: int) -> str:
    if 0 <= hour < 6:
        return "early_morning"
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "afternoon"
    return "evening"


def _build_behavior_sequences(
    events: pd.DataFrame, now: pd.Timestamp, cfg: ProfileConfig
) -> pd.DataFrame:
    occurrences: list[dict[str, Any]] = []
    prefix_counts: dict[tuple[str, tuple[str, ...]], int] = {}
    global_users_by_sequence: dict[tuple[str, ...], set[str]] = {}

    for user_id, user_events in events.groupby("user_id", sort=False):
        for session in _split_sessions(user_events, cfg.session_gap_minutes):
            actions = session["action_name"].tolist()
            times = session["event_time"].tolist()
            min_prefix_len = max(cfg.sequence_min_len - 1, 1)
            max_prefix_len = max(cfg.sequence_max_len - 1, min_prefix_len)
            for start in range(len(actions)):
                for length in range(min_prefix_len, max_prefix_len + 1):
                    end = start + length
                    if end > len(actions):
                        break
                    prefix = tuple(actions[start:end])
                    key = (user_id, prefix)
                    prefix_counts[key] = prefix_counts.get(key, 0) + 1
            for start in range(len(actions)):
                for length in range(cfg.sequence_min_len, cfg.sequence_max_len + 1):
                    end = start + length
                    if end > len(actions):
                        break
                    sequence = tuple(actions[start:end])
                    occurrences.append(
                        {
                            "user_id": user_id,
                            "sequence_tuple": sequence,
                            "last_time": times[end - 1],
                        }
                    )
                    global_users_by_sequence.setdefault(sequence, set()).add(user_id)

    if not occurrences:
        return _empty_sequences()

    total_users = max(events["user_id"].nunique(), 1)
    grouped = (
        pd.DataFrame(occurrences)
        .groupby(["user_id", "sequence_tuple"])
        .agg(support_count=("last_time", "size"), last_time=("last_time", "max"))
        .reset_index()
    )

    rows: list[dict[str, Any]] = []
    for item in grouped.itertuples(index=False):
        prefix = item.sequence_tuple[:-1]
        prefix_total = prefix_counts.get((item.user_id, prefix), item.support_count)
        confidence = item.support_count / max(prefix_total, 1)
        global_user_ratio = len(global_users_by_sequence[item.sequence_tuple]) / total_users
        sequence_length = len(item.sequence_tuple)
        min_support = (
            cfg.min_pair_sequence_support
            if sequence_length == 2
            else cfg.min_long_sequence_support
        )
        min_confidence = (
            cfg.min_pair_sequence_confidence
            if sequence_length == 2
            else cfg.min_long_sequence_confidence
        )
        if item.support_count < min_support:
            continue
        if confidence < min_confidence:
            continue
        if (
            global_user_ratio > cfg.sequence_global_user_ratio_threshold
            and confidence < cfg.sequence_global_min_confidence
        ):
            continue
        age_days = max(
            (pd.Timestamp(now) - item.last_time).total_seconds()
            / 86400.0,
            0.0,
        )
        length_weight = _sequence_length_weight(sequence_length, cfg)
        recency_weight = exp(-age_days / cfg.sequence_recency_decay_days)
        sequence_score = (
            log1p(item.support_count)
            * confidence
            * length_weight
            * recency_weight
        )
        rows.append(
            {
                "user_id": item.user_id,
                "sequence": " -> ".join(item.sequence_tuple),
                "sequence_tuple": item.sequence_tuple,
                "sequence_length": sequence_length,
                "support_count": int(item.support_count),
                "last_time": item.last_time,
                "transition_confidence": round(float(confidence), 6),
                "sequence_score": round(float(sequence_score), 6),
                "next_action_candidate": item.sequence_tuple[-1],
            }
        )

    if not rows:
        return _empty_sequences()
    result = pd.DataFrame(rows)
    result = _filter_subsequences(result, cfg)
    if result.empty:
        return _empty_sequences()
    result = result.sort_values(
        ["user_id", "sequence_score", "support_count", "transition_confidence", "last_time"],
        ascending=[True, False, False, False, False],
    )
    result["rank"] = result.groupby("user_id").cumcount() + 1
    return (
        result[result["rank"] <= cfg.sequence_top_k]
        .drop(columns=["sequence_tuple", "rank"])
        .reset_index(drop=True)
    )


def _sequence_length_weight(sequence_length: int, cfg: ProfileConfig) -> float:
    if sequence_length == 2:
        return cfg.sequence_pair_length_weight
    if sequence_length == 3:
        return cfg.sequence_triple_length_weight
    return cfg.sequence_long_length_weight


def _filter_subsequences(sequences: pd.DataFrame, cfg: ProfileConfig) -> pd.DataFrame:
    keep_indexes: set[int] = set(sequences.index)
    for user_id, user_rows in sequences.groupby("user_id", sort=False):
        records = list(user_rows.itertuples())
        for short in records:
            short_tuple = short.sequence_tuple
            for long in records:
                long_tuple = long.sequence_tuple
                if len(long_tuple) <= len(short_tuple):
                    continue
                if not _is_contiguous_subsequence(short_tuple, long_tuple):
                    continue
                if short.support_count >= long.support_count + cfg.subsequence_support_margin:
                    continue
                keep_indexes.discard(short.Index)
                break
    return sequences.loc[sorted(keep_indexes)].copy()


def _is_contiguous_subsequence(short: tuple[str, ...], long: tuple[str, ...]) -> bool:
    if len(short) >= len(long):
        return False
    limit = len(long) - len(short) + 1
    return any(long[start : start + len(short)] == short for start in range(limit))


def _split_sessions(user_events: pd.DataFrame, gap_minutes: int) -> list[pd.DataFrame]:
    sorted_events = user_events.sort_values("event_time").reset_index(drop=True)
    gaps = sorted_events["event_time"].diff() > pd.Timedelta(minutes=gap_minutes)
    session_id = gaps.cumsum()
    sessions: list[pd.DataFrame] = []
    for _, session in sorted_events.groupby(session_id, sort=False):
        ambiguous = session["event_time"].duplicated(keep=False)
        if not ambiguous.any():
            sessions.append(session)
            continue
        segment_id = ambiguous.cumsum()
        unambiguous = session.loc[~ambiguous].copy()
        if unambiguous.empty:
            continue
        unambiguous["_segment_id"] = segment_id.loc[~ambiguous]
        sessions.extend(
            segment.drop(columns="_segment_id")
            for _, segment in unambiguous.groupby("_segment_id", sort=False)
        )
    return sessions


def _empty_recent_tasks() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "user_id",
            "action_name",
            "event_time",
            "rank",
            "user_activity_level",
        ]
    )


def _empty_periodic_tasks() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "user_id",
            "action_name",
            "period_type",
            "period_value",
            "preferred_time_window",
            "next_expected_time",
            "confidence",
            "evidence",
        ]
    )


def _empty_high_freq_tasks() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "user_id",
            "action_name",
            "recent_count",
            "baseline_count",
            "lift",
            "trend_score",
            "time_window",
        ]
    )


def _empty_time_period_tasks() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
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
        ]
    )


def _empty_sequences() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "user_id",
            "sequence",
            "sequence_length",
            "support_count",
            "last_time",
            "transition_confidence",
            "sequence_score",
            "next_action_candidate",
        ]
    )
