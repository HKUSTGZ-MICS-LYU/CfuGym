"""Reconstruct known provider cost from the append-only request audit.

A request may have both received and completed events. They are updates to one
request's usage, not two billable calls. Missing provider usage is not zero.
"""
from __future__ import annotations

import json
from pathlib import Path

FIELDS = ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens")


def load_request_usage(path: Path, warnings: list[str] | None = None, *, started_at: str = "") -> dict:
    warnings = warnings if warnings is not None else []
    if not path.is_file():
        warnings.append("request_audit.jsonl missing; token usage falls back to CLI/report")
        return {}
    requests: dict[str, dict] = {}
    malformed = 0
    try:
        with path.open(errors="replace") as handle:
            for line_no, line in enumerate(handle, 1):
                try:
                    event = json.loads(line)
                except (ValueError, TypeError):
                    malformed += 1
                    warnings.append(f"request_audit.jsonl line {line_no} unparsed")
                    continue
                if not isinstance(event, dict):
                    malformed += 1
                    continue
                request_id = event.get("request_id")
                if not isinstance(request_id, str) or not request_id:
                    continue
                if event.get("event") == "request":
                    metadata = event.get("metadata") or {}
                    if not isinstance(metadata, dict):
                        metadata = {}
                    endpoint = metadata.get("endpoint")
                    if endpoint and endpoint not in {"chat", "responses"}:
                        continue  # capacity discovery is not a completion
                    if started_at and event.get("timestamp"):
                        from datetime import datetime
                        try:
                            stamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
                            start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                            if stamp < start:
                                continue
                        except (TypeError, ValueError):
                            pass
                    requests.setdefault(request_id, {"stage": metadata.get("stage") or "unknown",
                                                     "usage": {}, "status": "pending"})
                elif request_id in requests and event.get("event") in {"result", "error"}:
                    item = requests[request_id]
                    if event["event"] == "error":
                        item["status"] = "failed"
                        item["error"] = event.get("error")
                        continue
                    result = event.get("result") or {}
                    if not isinstance(result, dict):
                        continue
                    item["status"] = result.get("status") or "received"
                    usage = result.get("usage") or result
                    if not isinstance(usage, dict):
                        continue
                    for key in FIELDS:
                        value = usage.get(key)
                        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                            item["usage"][key] = value
                    if "total_tokens" not in usage and any(key in usage for key in FIELDS[:2]) and all(key in item["usage"] for key in FIELDS[:2]):
                        # A later partial event must not retain a stale total
                        # from an earlier received response.
                        item["usage"]["total_tokens"] = item["usage"]["input_tokens"] + item["usage"]["output_tokens"]
    except OSError as exc:
        warnings.append(f"request_audit.jsonl unreadable: {exc}")
        return {}
    known = {key: 0 for key in FIELDS}
    missing = {key: [] for key in FIELDS}
    by_stage: dict[str, dict] = {}
    for request_id, item in requests.items():
        usage = item["usage"]
        if "total_tokens" not in usage and all(key in usage for key in FIELDS[:2]):
            usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        bucket = by_stage.setdefault(item["stage"], {"calls": 0, "known_usage": {key: 0 for key in FIELDS},
                                                   "missing_usage_requests": []})
        bucket["calls"] += 1
        for key in FIELDS:
            if key in usage:
                known[key] += usage[key]
                bucket["known_usage"][key] += usage[key]
            else:
                missing[key].append(request_id)
        if any(key not in usage for key in FIELDS[:3]):
            bucket["missing_usage_requests"].append(request_id)
    missing_requests = sorted(set().union(*(set(missing[key]) for key in FIELDS[:3])))
    if missing_requests:
        warnings.append(f"request audit usage missing for {len(missing_requests)} request(s): "
                        + ", ".join(missing_requests))
    if missing["cached_input_tokens"]:
        warnings.append(f"request audit cached-input usage missing for {len(missing['cached_input_tokens'])} request(s)")
    if not requests:
        warnings.append("request_audit.jsonl has no completion requests; usage unavailable")
    return {"source": "request_audit", "calls": len(requests),
            **{key: known[key] if requests and not missing[key] and not malformed else None for key in FIELDS},
            "known_usage": known, "missing_usage_requests": missing_requests,
            "missing_by_field": missing, "usage_complete": bool(requests) and not missing_requests and not malformed,
            "malformed_events": malformed, "by_stage": by_stage}
