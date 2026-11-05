"""Validate an operator's recorded semantic review against immutable trace bytes."""
import hashlib
import json

def valid_review(folder):
    path=folder/"semantic-review.json"
    if not path.exists():return None
    report=json.loads(path.read_text())
    traces={str(p.relative_to(folder)) for pattern in
        ("turn-*/answer.md","turn-*/sandbox.json","turn-*/python.json","turn-*/pass.json",
         "turn-*/forward.json","turn-*/forward-request.json")
        for p in folder.glob(pattern)}
    if not traces or traces!=set(report["checked_files"]):return None
    for name,digest in report["checked_files"].items():
        p=folder/name
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:return None
    return report


def review_status(folder):
    """A hash-valid review may reject a run; its existence does not mean passed."""
    report=valid_review(folder)
    return report["verdict"] if report else "pending"
