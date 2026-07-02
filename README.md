# Bank App User Portrait

This project builds user behavior portraits from normalized bank app click events.

Required input columns:

- `user_id`
- `action_name`
- `event_time`

Generated portrait tables:

- `activity_levels`
- `recent_tasks` with each user's latest 3 behavior events
- `periodic_tasks`
- `recent_high_freq_tasks`
- `frequent_tasks` with each user's all-history Top 5 actions and frequency metrics
- `time_period_tasks` with fixed time-of-day high-frequency behavior
- `personal_behavior_sequences` with `sequence_length` and `sequence_score`

Key rules:

- Naive timestamps are interpreted in `Asia/Shanghai`; timezone-aware timestamps are
  converted to that business timezone before all date and time-period calculations.
- Users with no events in the latest 30 days are always classified as low activity.
- Monthly anchors use the date nearest the observed median when concentration ties.
- Period detection counts each active business date once per action and uses that
  date's first visit when deriving the preferred time and next expected hour.
- Sequence confidence is `full sequence count / all prefix count`, including prefixes
  that end a session and single-event sessions containing only the prefix action.
- Multiple actions for the same user at the same timestamp remain available to all
  aggregate portraits but form an ambiguity boundary in sequence mining.
- Recent high-frequency behavior keeps its existing 7/30-day rules unchanged.
- Time-period behavior uses fixed buckets only: `early_morning` 00:00-06:00,
  `morning` 06:00-12:00, `afternoon` 12:00-18:00, and `evening` 18:00-24:00.

## Python Usage

```python
import pandas as pd
from user_portrait import build_user_profiles

events = pd.DataFrame(
    [
        {"user_id": "u1", "action_name": "给朋友转账", "event_time": "2026-03-23 18:00:00"},
        {"user_id": "u1", "action_name": "买黄金", "event_time": "2026-03-24 11:00:00"},
    ]
)

profiles = build_user_profiles(events, now="2026-03-31 12:00:00")
print(profiles["recent_tasks"])
```

## CLI Usage

```powershell
python -m user_portrait.cli events.csv output_profiles --now "2026-03-31 12:00:00"
```

The CLI reads `user_id` and `action_name` as strings, writes one CSV per portrait
table, and writes `run_manifest.json` last to mark a complete snapshot. The manifest
contains input quality counts, the reference time, configuration, output row counts,
and elapsed time.

Use a JSON configuration file when non-default `ProfileConfig` values are required:

```powershell
python -m user_portrait.cli events.csv output_profiles --now "2026-03-31 12:00:00" --config profile_config.json
```

```json
{
  "business_timezone": "Asia/Shanghai",
  "activity_window_days": 30,
  "daily_max_avg_interval_days": 1.5,
  "monthly_min_span_days": 45,
  "high_freq_min_count": 4,
  "high_freq_recent_window_days": 7,
  "time_period_min_count": 4,
  "time_period_min_active_days": 3,
  "time_period_min_share": 0.7,
  "sequence_recency_decay_days": 30.0,
  "sequence_top_k": 5
}
```

All tunable detection windows, thresholds, score weights, and Top K limits are
defined by `ProfileConfig` and validated when the configuration is created. The
former `high_freq_recent_count_7d` JSON field is accepted as a deprecated alias
for `high_freq_min_count`; conflicting old and new values are rejected.

## Synthetic Validation Data

Generate a deterministic validation dataset with 240 users, event labels, positive
ground-truth tables, and hard negative ground-truth tables:

```powershell
python generate_sample_events.py sample_data
```

The command writes:

- `sample_events.csv`: input events with only `user_id`, `action_name`, and `event_time`
- `sample_ground_truth_recent.csv`: expected recent tasks
- `sample_ground_truth_frequent.csv`: expected all-history Top 5 frequent tasks
- `sample_ground_truth_periodic.csv`: positive periodic task labels
- `sample_ground_truth_periodic_negative.csv`: hard negative periodic task labels
- `sample_ground_truth_high_freq.csv`: positive recent high-frequency task labels
- `sample_ground_truth_high_freq_negative.csv`: hard negative recent high-frequency task labels
- `sample_ground_truth_time_period.csv`: positive fixed time-period task labels
- `sample_ground_truth_time_period_negative.csv`: hard negative fixed time-period task labels
- `sample_ground_truth_sequences.csv`: expected personal behavior sequences

Build portraits from the generated sample data:

```powershell
python -m user_portrait.cli sample_data\sample_events.csv portrait_output --now "2026-03-31 12:00:00"
```

Evaluate the generated CSV snapshot against the synthetic ground truth:

```powershell
python evaluate_profiles.py sample_data portrait_output
```

The evaluator prints a metric table and writes
`portrait_output\evaluation_metrics.json`. Enable the optional CI quality gate with:

```powershell
python evaluate_profiles.py sample_data portrait_output --fail-on-threshold
```

The gate returns exit code `1` when a metric with an established regression threshold
fails. Invalid or missing inputs return exit code `2`. Use `--output-json` to override
the report path. Fixed time-period precision is intentionally reported as `null`
because its positive labels are not an exhaustive list of all valid candidates.

Run the regression tests:

```powershell
python -m pytest -q
```

Synthetic precision and recall are regression indicators for the generated patterns;
they are not estimates of production accuracy.

## Large Comprehensive Validation

Keep the baseline dataset for fast regression and generate the separate 2,000-user,
one-year validation dataset with:

```powershell
python generate_sample_events.py sample_data_large --profile large --seed 20260701
python -m user_portrait.cli sample_data_large\sample_events.csv portrait_output_large --now "2026-06-30 12:00:00"
python -m user_portrait.cli sample_data_large\sample_events_dirty.csv portrait_output_large_dirty --now "2026-06-30 12:00:00"
python evaluate_profiles.py sample_data_large portrait_output_large --dirty-profiles-dir portrait_output_large_dirty --fail-on-threshold
```

The large profile writes `sample_dataset_manifest.json`, an exhaustive
`sample_ground_truth_time_period.csv`, and the separately named injected-pattern file
`sample_injected_time_period_patterns.csv`. It covers 175,902 clean events plus 100
future or invalid rows in the dirty input. The exhaustive time-period labels allow
Precision, Recall, all statistic fields, and rank to be evaluated directly.
