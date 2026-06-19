# Matching API Contract

POST /match

Request

{
  "student_id": 1,
  "job_id": 101
}

Response

{
  "match_score": 92,
  "decision": "MATCH",
  "reasons": [
    "Python requirement satisfied",
    "Machine Learning requirement satisfied",
    "Project requirement satisfied",
    "Experience requirement satisfied"
  ]
}