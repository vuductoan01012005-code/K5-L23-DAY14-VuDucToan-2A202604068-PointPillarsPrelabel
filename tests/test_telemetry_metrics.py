import json

from bonus.telemetry_metrics import main, session_metrics


def _ev(t, name, val=None, **payload):
    return {"timestamp": f"2026-11-24T10:00:{t:02d}Z", "obj_name": name, "obj_val": val,
            "job_id": 7, "payload": json.dumps(payload)}


EVENTS = [
    _ev(0, "canvas-click", dt_ms=None, dist_px=None),
    _ev(2, "canvas-click", dt_ms=2000, dist_px=100),
    _ev(3, "keydown", "f", dt_ms=None),
    _ev(5, "canvas-click", dt_ms=3000, dist_px=300),
    _ev(9, "keydown", "Ctrl+s", dt_ms=6000),
]


def test_session_metrics_counts_and_gaps():
    m = session_metrics(EVENTS)
    assert (m["canvas_clicks"], m["shortcut_keys"]) == (3, 2)
    assert m["median_click_gap_ms"] == 2500 and m["mean_click_dist_px"] == 200
    assert m["keyboard_ratio"] == 0.4 and m["active_span_s"] == 9 and m["job_id"] == 7


def test_cli_writes_csv(tmp_path):
    log = tmp_path / "s.json"
    log.write_text(json.dumps(EVENTS[::-1]))  # thứ tự lộn vẫn phải sắp lại theo thời gian
    out = tmp_path / "m.csv"
    assert main([str(log), "--csv", str(out)]) == 0
    assert "canvas_clicks" in out.read_text()
