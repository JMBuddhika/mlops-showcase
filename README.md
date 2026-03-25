# Batch vs Real-Time Inference Patterns (Distributed ML Systems)

## Why this decision matters

Inference architecture is a product and systems decision, not just an ML decision.
Choosing the wrong pattern causes either unnecessary latency/cost or stale predictions that hurt business outcomes.

For Grab-like environments, you usually need to optimize for both user-facing latency and fleet-scale compute efficiency.

---

## 1) Pattern definitions

## A) Batch inference

- score large datasets on a schedule (hourly/daily/etc.)
- write predictions to storage/cache for downstream consumption

Typical use cases:
- risk pre-screening
- demand forecasting snapshots
- offline ranking feature generation

## B) Real-time inference

- request-time prediction with strict latency budgets (e.g., p95 < 100 ms)
- fetch fresh features and run model online per request

Typical use cases:
- ETA personalization
- dynamic pricing adjustments
- checkout fraud decisions

## C) Hybrid inference (common production pattern)

- precompute heavy/stable components in batch
- combine with fresh request-time signals online

Typical use case:
- ranking: precomputed user/item embeddings + real-time context features

---

## 2) Decision framework

Evaluate each use case along these axes:

1. **Latency SLO**
- if strict sub-second user path -> real-time or hybrid
2. **Feature freshness requirement**
- if stale features degrade value quickly -> real-time component required
3. **Traffic shape**
- high sustained QPS can favor batch precompute + cache
4. **Cost envelope**
- real-time often costs more due to overprovisioning and low tail-latency constraints
5. **Failure tolerance**
- batch degrades gracefully (older outputs), real-time requires strong fallback paths
6. **Business impact of staleness**
- if wrong/stale predictions are expensive, invest in online serving and drift monitoring

---

## 3) Tradeoffs at a glance

### Batch strengths
- cheaper per prediction at scale
- operationally simpler serving path
- easier reproducibility and audit trails

### Batch weaknesses
- stale outputs between runs
- limited personalization to latest context
- poor fit for tight user-interaction loops

### Real-time strengths
- freshest context and personalization
- immediate adaptation to changing state
- better for interactive decisioning

### Real-time weaknesses
- higher infra and on-call complexity
- strict latency/reliability engineering required
- expensive autoscaling and tail-latency mitigation

---

## 4) Failure modes and mitigation

## Batch failure modes
- delayed pipelines -> stale predictions
- partial recompute -> inconsistent snapshots

Mitigations:
- data freshness SLO alerts
- atomic snapshot publish
- fallback to previous known-good snapshot

## Real-time failure modes
- model server timeout/overload
- feature store latency spikes
- cascading dependency failures

Mitigations:
- request timeout budgets + circuit breakers
- cached fallback features/predictions
- graceful degradation policy (heuristics/default model)
- shadow canary before full rollout

---

## 5) Cost and capacity considerations

Batch:
- optimize for throughput (large jobs, spot/preemptible where safe)
- run off-peak when possible

Real-time:
- optimize for p95/p99 latency
- account for peak burst headroom and multi-AZ redundancy
- cost grows with strict SLO + underutilized reserve capacity

Hybrid often wins:
- heavy feature transforms done offline
- smaller online model footprint reduces latency and cost

---

## 6) Recommended pattern choices (practical rules)

Use **batch** when:
- decision tolerance is minutes/hours
- prediction consumes large cohorts, not individual requests

Use **real-time** when:
- user or transaction context changes quickly
- stale prediction materially harms UX/revenue/risk

Use **hybrid** when:
- you need near-real-time quality but cannot afford full online recomputation
- model has expensive stable features + lightweight dynamic features

---

## 7) MLOps operating checklist

Before launching any inference pattern, define:

- latency SLO (p50/p95/p99)
- freshness SLO (max acceptable feature/prediction age)
- error budget and fallback behavior
- drift monitoring and retraining trigger
- rollout strategy (shadow -> canary -> gradual ramp)
- rollback playbook with explicit owner and alert thresholds

---

## 8) What to prioritize for portfolio impact

For ML Engineer/MLOps interviews, show that you can:

1. justify pattern choice with business/SLO constraints
2. design fallback and degradation behavior
3. quantify latency-freshness-cost tradeoffs
4. instrument reliability and model quality in production

That signals production readiness far better than model metrics alone.