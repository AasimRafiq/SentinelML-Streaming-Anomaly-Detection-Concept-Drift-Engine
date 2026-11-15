# SentinelML — Streaming Anomaly Detection & Concept Drift Engine

A lightweight, zero-dependency Python machine learning & statistical telemetry engine designed for real-time microservice monitoring, root-cause attribution, and Population Stability Index (PSI) data drift detection.

## Mathematical & Algorithmic Architecture
1. **Hybrid Z-Score & Modified Z-Score (MAD):** Combines rolling window standard Z-scores with Median Absolute Deviation (MAD) scoring to detect univariate spikes without outlier baseline poisoning.
2. **Multivariate Standardized Euclidean Distance:** Aggregates normalized feature deviations across latency, error rate, CPU load, and transaction throughput into a unified anomaly severity score (`NORMAL`, `WARNING`, `CRITICAL`).
3. **Population Stability Index (PSI) Drift Monitor:** Compares baseline probability distributions against live rolling windows to flag gradual concept/data drift (`PSI >= 0.20`).
4. **Automated Incident Triage & Export:** Outputs structured JSON incident audit reports and scored CSV telemetry logs.

## Quick Start
```bash
python3 sentinel_engine.py
```
