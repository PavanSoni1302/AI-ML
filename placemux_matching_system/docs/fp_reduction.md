# False Positive Reduction

## Objective

Reduce unnecessary manual reviews while maintaining candidate verification quality.

## Input

- proctoring_report.csv

## Output

- fp_reduction_report.csv

## Metrics

- Baseline False Positives
- Reduced False Positives
- Reduction Percentage

## Decision Rules

- Confidence >= 90 and no violations → Verified
- Confidence >= 80 and no violations → Verified
- Confidence >= 75 with <=1 violation → Manual Review
- Otherwise → Rejected