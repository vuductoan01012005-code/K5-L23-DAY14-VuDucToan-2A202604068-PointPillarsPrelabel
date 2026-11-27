"""JSON box files saved by Windows Notepad carry a UTF-8 BOM and must still load."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import vf_d14  # noqa: E402


class TestJsonBom(unittest.TestCase):
    def test_load_reference_accepts_utf8_bom(self) -> None:
        box = {"x": 1.0, "y": 2.0, "z": 0.0, "length": 4.0, "width": 1.8, "height": 1.5, "yaw": 0.0, "label": "Car"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ref_A.json"
            path.write_text(json.dumps({"frames": {"s02f07": [box]}}), encoding="utf-8-sig")
            self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))
            got = vf_d14.load_reference(str(path), None, ["s02f07", "s03f01"])
        self.assertEqual(got, {"s02f07": [box]})


if __name__ == "__main__":
    unittest.main()
