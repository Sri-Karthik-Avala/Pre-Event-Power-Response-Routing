# Pre-Event Power Response Routing

| | |
| --- | --- |
| Final rank | #10 |
| Domain | Sequence To Sequence |
| Difficulty | Medium |
| Scoring | ↑ Higher is better |
| Compute | CPU |
| Challenge status | Accepted / closed |
| Solutions submitted | 3 |
| Last submission | 2026-09-07 |

## Problem statement

### Overview

Power-monitoring operators do not only need an anomaly flag. Before an event fully develops, they need to choose the first response route: inspect the equipment or escalate a compound condition, investigate supply quality, or check telemetry. They also need an urgency estimate.

Each solver-facing row is a six-channel, 60-second sequence ending 60–150 seconds before a real controlled event. Predict the response route and urgency as one compact sequence label. The source station, calendar day, event identifier, and source labels are withheld.

This is a sequence to sequence task over real measured signals. The evaluation is event-balanced: every hidden event contributes total weight one across its available lead-time windows, so an event with more valid windows cannot dominate the score.

### Dataset

### File descriptions

- `train.csv` — labeled lead-time sequences from five calendar-day segments of each monitoring recording.
- `test.csv` — unlabeled lead-time sequences from two complete calendar-day segments held out from each recording.
- `sample_submission.csv` — valid submission format with deterministic placeholder labels.

### Column descriptions

- `id` — unique 16-character hexadecimal sequence identifier.
- `voltage_V_sequence` — JSON array of 60 measured voltage values.
- `current_A_sequence` — JSON array of 60 measured current values.
- `power_W_sequence` — JSON array of 60 measured active-power values.
- `frequency_Hz_sequence` — JSON array of 60 measured frequency values.
- `power_factor_sequence` — JSON array of 60 measured power-factor values.
- `wifi_rssi_dBm_sequence` — JSON array of 60 measured telemetry-strength values.
- `target` — training-only label in the form `route|urgency`.

The three route labels are:

- `equipment_escalation` — inspect the connected load or operating condition; compound events use this route because they require escalation at the first decision point.
- `supply_quality` — investigate voltage, frequency, power-quality, or outage behavior.
- `sensor_telemetry` — check the measurement or communications path.

The four urgency labels are `incipient`, `low`, `medium`, and `urgent`. The source’s high and critical states are intentionally combined into `urgent` because both require immediate operator escalation before finer-grained diagnosis.

### Evaluation

Split every predicted `target` at the first `|`. Give each row the inverse of its hidden event group’s row count, so each event contributes total weight one. Then compute:

```
event_weight = 1.0 / hidden_event_group_row_counts

route_f1 = macro_f1(true_route, predicted_route,

                    labels=["equipment_escalation", "supply_quality",

                            "sensor_telemetry"], sample_weight=event_weight)

urgency_f1 = macro_f1(true_urgency, predicted_urgency,

                      labels=["incipient", "low", "medium", "urgent"],

                      sample_weight=event_weight)

score = 0.65  *route_f1 + 0.35*  urgency_f1
```

The score is bounded in `[0, 1]` and higher is better. The route term receives more weight because a wrong first response can send a technician to the wrong subsystem; urgency remains important for dispatch order.

### Submission

Submit exactly one row per `id` in `test.csv`.

- `id` — the exact test sequence identifier.
- `target` — one route and one urgency joined by a single pipe, for example `supply_quality|urgent`.

Example:

```
id,target

00052a6150a995b0,supply_quality|incipient

0030c306300ab36a,supply_quality|urgent
```

### Requirements

- Include every test `id` exactly once.
- Use only the three route labels and four urgency labels defined above.
- Use exactly one `|` separator and lowercase labels.
- Keep all six sequence columns out of the submission.
- The score is calculated on hidden event groups; do not assume row-level class balance.

### What Not To Use

- Do not use the source archive, source station files, source event IDs, source calendar-day identifiers, or the source’s label columns.
- Do not recover target labels by matching sequence values or hashed IDs to the source release or its released analysis code.
- Do not use the source detector flag, threshold, precomputed entropy features, or any source-derived label proxy; they are intentionally absent from `train.csv` and `test.csv`.
- Do not generate synthetic signal windows or inject artificial faults. The intended data are the supplied measured windows.
- Do not replace the learned model with a threshold-only or hand-written physical rule; a genuine model trained on the labeled sequences is required.
- Do not use a fixed lookup table keyed by sequence order, row ID, or source identity instead of learning from the labeled sequences.
