# ML Observability Stack Blueprint (Metrics + Alerts)

## 1) Why this matters

Training a model is not the hard part in production.
The hard part is detecting silent degradation early, routing the right alerts, and restoring service before user/business impact.

For ML Engineer/MLOps roles (Grab-like environments), observability quality is a major hiring signal.

---

## 2) Observability objectives

A production ML observability stack should answer:

1. Is the service healthy? (latency, error, throughput)
2. Is model quality degrading? (proxy metrics + delayed ground truth)
3. Is input data shifting? (feature drift, schema drift, missingness)
4. Is alerting actionable? (severity, ownership, runbook linkage)

---

## 3) Reference architecture

Flow:

1. **Instrumentation layer**
- model server exports metrics/traces/logs
2. **Telemetry pipeline**
- Prometheus/OpenTelemetry collectors
3. **Storage/Query**
- time-series DB for metrics, log backend for events
4. **Alerting layer**
- Alertmanager / pager routing by severity
5. **Dashboards + Runbooks**
- service, model, and data health views with owner actions

Key principle:
- keep one shared observability plane for platform + model teams to reduce blind spots.

---

## 4) Metric taxonomy (what to measure)

## A) Service-level metrics (golden signals)

- request rate (RPS)
- latency (p50/p95/p99)
- error rate by code/path
- saturation (CPU, GPU, memory, queue depth)

Purpose:
- detect serving incidents and infra bottlenecks quickly.

## B) Model-level metrics

Online proxies:

- prediction score distribution
- class/label distribution
- confidence entropy trends
- business proxy metrics (CTR, conversion, fraud flag rate)

Offline/delayed truth:

- rolling AUC/F1/MAE where labels arrive later
- calibration drift

Purpose:
- detect model quality issues not visible in infra metrics.

## C) Data/feature-level metrics

- missing/null ratio per feature
- out-of-range violations
- schema version mismatch
- drift scores vs reference window (PSI/KS/JSD)

Purpose:
- catch upstream data breakage before model output fails visibly.

---

## 5) Suggested SLOs (example)

Serving SLOs:

- p95 latency < 120 ms over 5-minute windows
- error rate < 1% for valid requests
- availability >= 99.9%

Model/data guardrails:

- drift score (PSI) for critical features < 0.2
- missing ratio for critical features < 2%
- prediction positive-rate deviation within defined control band

These are not universal values; tune by product criticality and risk.

---

## 6) Alert design principles

1. **Page only on user-impacting issues**
Informational anomalies should create tickets, not wake people.

2. **Multi-window + threshold**
Use fast + slow burn detection to reduce false alarms.

3. **Attach context**
Every alert must include dashboard link, owning team, and runbook.

4. **Deduplicate and group**
Avoid alert storms during one root-cause incident.

5. **Use severity tiers**
- P1: immediate user/business risk
- P2: degraded quality, short-term tolerable
- P3: investigation backlog

---

## 7) Example alert classes

P1 (pager):

- p99 latency spike + sustained error increase
- model endpoint unavailable
- critical feature missingness spikes

P2 (urgent ticket/Slack):

- drift threshold exceeded for high-impact features
- prediction distribution shift without infra issue
- queue lag threatens SLA

P3 (daily triage):

- low-severity schema warnings
- minor calibration movement
- non-critical feature drift

---

## 8) Incident response playbook (first 15 minutes)

1. Verify blast radius:
- affected model versions, regions, tenants
2. Check service health:
- latency/error/saturation before model rollback
3. Check data health:
- feature freshness, missingness, schema compatibility
4. Mitigate:
- rollback model version OR enable safe fallback path
5. Stabilize:
- rate-limit, autoscale, disable non-critical features
6. Communicate:
- incident channel updates with ETA and owner

---

## 9) Fallback strategies

Operationally safe options:

- last-known-good model version
- rules-based fallback for critical flows
- stale-feature tolerance with bounded TTL
- graceful degradation (reduced personalization, preserved core function)

Design fallback before launch; do not invent during incident.

---

## 10) Anti-noise controls (to prevent alert fatigue)

- require minimum sample size before firing drift alerts
- suppress low-volume periods and scheduled deploy windows
- anomaly confirmation over consecutive intervals
- route non-urgent anomalies to ticket queue with SLA

Success metric:
- high signal-to-noise alert stream where most pages are actionable.

---

## 11) Ownership and governance

Each production model should have:

- explicit owner rotation
- SLO document
- dashboard URL set
- alert rule file in version control
- runbook with rollback/fallback procedures
- postmortem template for P1/P2 incidents

Without ownership, metrics are decoration.

---

## 12) Portfolio-ready implementation scope

This repo artifact pairs with `alert-rules.yaml` to show:

- concrete metrics-to-alert mapping
- practical severity design
- production response thinking

That combination is stronger than "we monitor latency" statements in interviews.