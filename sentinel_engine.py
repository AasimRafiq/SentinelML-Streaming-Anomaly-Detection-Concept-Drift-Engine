#!/usr/bin/env python3
"""
SentinelML — Real-Time Anomaly Detection, Drift Monitoring & Predictive Triage CLI/Engine
Author: Aasim Rafique
Stack: Python 3.9+ (Zero external dependencies — pure standard library math, statistics, json, csv)
"""

import csv
import json
import math
import random
import statistics
from datetime import datetime, timedelta
from pathlib import Path


class StreamingAnomalyDetector:
    """
    Hybrid statistical & geometric anomaly detector for multi-dimensional operational telemetry:
    1. Rolling Z-Score & Modified Z-Score (MAD) for univariate spike detection.
    2. Mahalanobis-like Standardized Euclidean Distance for multivariate feature correlation.
    3. Population Stability Index (PSI) for Concept / Data Drift detection between baseline and live windows.
    """

    def __init__(self, window_size: int = 30, z_threshold: float = 2.8):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.history = {
            "latency_ms": [],
            "error_rate": [],
            "cpu_load": [],
            "tx_volume": []
        }
        self.baseline = {}

    def fit_baseline(self, records: list[dict]) -> None:
        for key in self.history:
            vals = [r[key] for r in records]
            mean = statistics.mean(vals)
            stdev = statistics.stdev(vals) if len(vals) > 1 else 1.0
            median = statistics.median(vals)
            mad = statistics.median([abs(v - median) for v in vals]) or 1e-6
            self.baseline[key] = {
                "mean": mean,
                "stdev": max(stdev, 1e-6),
                "median": median,
                "mad": mad,
                "sample": vals
            }
            self.history[key] = vals[-self.window_size:]

    def score_event(self, event: dict) -> dict:
        feature_scores = {}
        squared_z_sum = 0.0

        for key, stats in self.baseline.items():
            val = event[key]
            rolling_vals = self.history[key]
            r_mean = statistics.mean(rolling_vals) if rolling_vals else stats["mean"]
            r_std = statistics.stdev(rolling_vals) if len(rolling_vals) > 1 else stats["stdev"]
            r_std = max(r_std, 1e-6)

            z_score = (val - r_mean) / r_std
            mod_z = 0.6745 * (val - stats["median"]) / stats["mad"]
            composite_z = max(abs(z_score), abs(mod_z) * 0.85)

            feature_scores[key] = round(composite_z, 3)
            squared_z_sum += composite_z ** 2

            # Update rolling window only with non-extreme points to prevent baseline poisoning
            if composite_z < self.z_threshold * 1.3:
                self.history[key].append(val)
                if len(self.history[key]) > self.window_size:
                    self.history[key].pop(0)

        multivariate_dist = round(math.sqrt(squared_z_sum / len(self.baseline)), 3)
        max_feature = max(feature_scores.items(), key=lambda x: x[1])
        is_anomaly = multivariate_dist >= 2.1 or max_feature[1] >= self.z_threshold

        severity = "NORMAL"
        if multivariate_dist >= 3.2 or max_feature[1] >= 4.0:
            severity = "CRITICAL"
        elif is_anomaly:
            severity = "WARNING"

        return {
            "timestamp": event["timestamp"],
            "service": event["service"],
            "is_anomaly": is_anomaly,
            "severity": severity,
            "multivariate_score": multivariate_dist,
            "root_cause_feature": max_feature[0],
            "root_cause_z": max_feature[1],
            "feature_z_scores": feature_scores
        }

    def compute_psi_drift(self, feature: str, bins: int = 5) -> float:
        """Calculates Population Stability Index (PSI) between baseline and current rolling window."""
        base_vals = self.baseline[feature]["sample"]
        curr_vals = self.history[feature]
        if not base_vals or not curr_vals:
            return 0.0

        low = min(min(base_vals), min(curr_vals))
        high = max(max(base_vals), max(curr_vals))
        if high == low:
            return 0.0

        step = (high - low) / bins
        psi = 0.0
        for b in range(bins):
            b_min = low + b * step
            b_max = high + 1e-9 if b == bins - 1 else low + (b + 1) * step
            b_pct = max(sum(1 for v in base_vals if b_min <= v < b_max) / len(base_vals), 1e-4)
            c_pct = max(sum(1 for v in curr_vals if b_min <= v < b_max) / len(curr_vals), 1e-4)
            psi += (c_pct - b_pct) * math.log(c_pct / b_pct)
        return round(psi, 4)


def generate_synthetic_telemetry(n_baseline: int = 80, n_stream: int = 60) -> tuple[list[dict], list[dict]]:
    random.seed(42)
    start_time = datetime(2026, 10, 1, 9, 0, 0)
    services = ["auth-gateway", "checkout-api", "payment-worker", "inventory-sync"]

    def make_point(idx: int, inject_fault: str = None) -> dict:
        ts = (start_time + timedelta(minutes=idx)).isoformat()
        svc = services[idx % len(services)]
        lat = random.gauss(120, 14)
        err = max(0.01, random.gauss(0.45, 0.12))
        cpu = random.gauss(42, 5)
        tx = random.gauss(310, 28)

        if inject_fault == "LATENCY_SPIKE":
            lat += random.uniform(110, 180)
            cpu += random.uniform(25, 38)
        elif inject_fault == "ERROR_SURGE":
            err += random.uniform(3.2, 5.5)
            tx -= random.uniform(90, 140)
        elif inject_fault == "GRADUAL_DRIFT":
            lat += 38
            cpu += 19

        return {
            "timestamp": ts,
            "service": svc,
            "latency_ms": round(max(10, lat), 2),
            "error_rate": round(max(0, err), 3),
            "cpu_load": round(min(100, max(5, cpu)), 2),
            "tx_volume": round(max(10, tx), 1)
        }

    baseline = [make_point(i) for i in range(n_baseline)]
    stream = []
    for j in range(n_stream):
        idx = n_baseline + j
        fault = None
        if j in (14, 15, 37):
            fault = "LATENCY_SPIKE"
        elif j in (26, 49):
            fault = "ERROR_SURGE"
        elif j >= 40:
            fault = "GRADUAL_DRIFT"
        stream.append(make_point(idx, fault))

    return baseline, stream


def run_pipeline(output_dir: str = "sentinel_output") -> None:
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)

    baseline, stream = generate_synthetic_telemetry()
    detector = StreamingAnomalyDetector(window_size=30, z_threshold=2.8)
    detector.fit_baseline(baseline)

    results = [detector.score_event(ev) for ev in stream]
    anomalies = [r for r in results if r["is_anomaly"]]
    drift_report = {
        feat: {
            "psi": detector.compute_psi_drift(feat),
            "status": "DRIFT_DETECTED" if detector.compute_psi_drift(feat) >= 0.2 else "STABLE"
        }
        for feat in detector.baseline
    }

    # Export JSON Audit Report
    report_payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_events_analyzed": len(results),
        "anomalies_detected": len(anomalies),
        "anomaly_rate_pct": round((len(anomalies) / len(results)) * 100, 2),
        "feature_drift_psi": drift_report,
        "incident_log": anomalies
    }
    (out_path / "incident_report.json").write_text(json.dumps(report_payload, indent=2))

    # Export CSV Telemetry Scores
    with open(out_path / "scored_stream.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "service", "severity", "multivariate_score", "root_cause_feature", "root_cause_z"])
        for r in results:
            writer.writerow([r["timestamp"], r["service"], r["severity"], r["multivariate_score"], r["root_cause_feature"], r["root_cause_z"]])

    print("=" * 64)
    print("SENTINEL-ML :: STREAMING ANOMALY & CONCEPT DRIFT ENGINE")
    print("=" * 64)
    print(f"Events Processed   : {len(results)}")
    print(f"Anomalies Flagged  : {len(anomalies)} ({report_payload['anomaly_rate_pct']}%)")
    print("\nFeature Concept Drift (PSI):")
    for k, v in drift_report.items():
        print(f"  - {k:<14}: PSI={v['psi']:<7} [{v['status']}]")
    print("\nTop Detected Incidents:")
    for inc in anomalies[:5]:
        print(f"  [{inc['severity']}] {inc['timestamp']} | {inc['service']} | Score={inc['multivariate_score']} | Cause={inc['root_cause_feature']} (z={inc['root_cause_z']})")
    print(f"\nArtifacts saved to: {out_path.resolve()}/")


if __name__ == "__main__":
    run_pipeline()
