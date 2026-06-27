# AI Trust Sign-off

## Objective

Validate that candidate recommendations are trustworthy before final matching.

## Inputs

- fp_reduction_report.csv
- ontology_mapping.csv

## Outputs

- trust_signoff.csv

## Trust Rules

- Verified + Parsed Ontology → APPROVED
- Manual Review → REVIEW
- Failed Verification → REJECTED

## Metrics

- Trust Score
- Trust Status
- Decision Reason

## Purpose

Ensure recommendations are explainable, verified, and ready for production.