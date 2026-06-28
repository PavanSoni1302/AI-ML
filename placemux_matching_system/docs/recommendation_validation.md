# Recommendation Validation

## Objective

Validate recommendation quality before publishing to stakeholder portals.

## Inputs

- recommendations.csv

## Outputs

- validated_recommendations.csv
- validation_metrics.csv

## Validation Rules

- APPROVED trust + score >= 85 → VALID
- APPROVED trust + score >= 70 → REVIEW
- Otherwise → REJECTED

## Metrics

- Total Recommendations
- Valid
- Review
- Rejected