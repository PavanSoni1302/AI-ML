# Drift Monitoring & Retraining

## Objective

Detect recommendation drift and trigger retraining.

## Inputs

- validated_recommendations.csv

## Outputs

- drift_report.csv
- retraining_log.csv

## Drift Rules

- |Drift| ≤ 5 → Stable
- |Drift| ≤ 15 → Monitor
- |Drift| > 15 → Retrain

## Retraining Actions

- Stable → No Action
- Monitor → Monitoring
- Retrain → Model Retrained