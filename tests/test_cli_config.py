"""CLI config validation, --set override parsing, and zero-temp-leak tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import pp_prelabel  # noqa: E402
from scripts import pp_sweep  # noqa: E402
from vf_pp.core import Config  # noqa: E402


class TestCliConfigValidation(unittest.TestCase):
    def test_set_missing_equals_raises_system_exit(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            pp_prelabel.load_config(None, ["score_thr"])
        self.assertIn("=", str(ctx.exception))

    def test_set_none_on_required_numeric_field_raises_system_exit(self) -> None:
        for override in ("score_thr=none", "score_thr=null", "score_thr=", "passes=none", "intensity_mode=none"):
            with self.subTest(override=override):
                with self.assertRaises(SystemExit):
                    pp_prelabel.load_config(None, [override])

        cfg = pp_prelabel.load_config(None, ["thr_car=none", "thr_cyc=null", "thr_ped="])
        self.assertIsNone(cfg.thr_car)
        self.assertIsNone(cfg.thr_cyc)
        self.assertIsNone(cfg.thr_ped)

        cfg_floats = pp_prelabel.load_config(None, ["thr_car=0.35", "thr_cyc=0.25", "thr_ped=0.2"])
        self.assertEqual(cfg_floats.thr_car, 0.35)
        self.assertEqual(cfg_floats.thr_cyc, 0.25)
        self.assertEqual(cfg_floats.thr_ped, 0.2)

        for bad_opt in ("thr_car=abc", "thr_cyc=nan", "thr_ped=inf"):
            with self.subTest(bad_opt=bad_opt):
                with self.assertRaises(SystemExit):
                    pp_prelabel.load_config(None, [bad_opt])

    def test_set_invalid_passes_or_voxel_size_or_enum_raises_system_exit(self) -> None:
        bad_overrides = [
            "passes=0",
            "passes=-1",
            "voxel_size=0",
            "voxel_size=-0.1",
            "range_m=0",
            "range_m=-5",
            "min_points=-1",
            "intensity_mode=typo",
            "ground_mode=typo",
            "min_points=5.0",
            "passes=True",
            "score_thr=abc",
            "score_thr=nan",
            "voxel_size=inf",
            "extra=anything",
        ]
        for override in bad_overrides:
            with self.subTest(override=override):
                with self.assertRaises(SystemExit):
                    pp_prelabel.load_config(None, [override])

    def test_pp_sweep_load_config_from_creates_zero_temp_files(self) -> None:
        tmp_dir = Path(tempfile.gettempdir())
        before = set(tmp_dir.glob("tmp*.json"))
        after = before
        try:
            base = Config()
            cfg = pp_sweep.load_config_from(base, {"score_thr": "0.4", "min_points": "5", "thr_car": "0.3"})
            after = set(tmp_dir.glob("tmp*.json"))
            self.assertEqual(after, before)
            self.assertEqual(cfg.score_thr, 0.4)
            self.assertEqual(cfg.min_points, 5)
            self.assertEqual(cfg.thr_car, 0.3)
            self.assertEqual(base.score_thr, 0.1)

            cfg_none = pp_sweep.load_config_from(cfg, {"thr_car": None})
            self.assertIsNone(cfg_none.thr_car)
            self.assertEqual(set(tmp_dir.glob("tmp*.json")), before)

            for bad_vals in ({"passes": "0"}, {"min_points": "5.0"}, {"intensity_mode": "typo"}):
                with self.subTest(bad_vals=bad_vals):
                    with self.assertRaises(SystemExit):
                        pp_sweep.load_config_from(base, bad_vals)
                    self.assertEqual(set(tmp_dir.glob("tmp*.json")), before)
        finally:
            for leaked in after - before:
                leaked.unlink(missing_ok=True)

    def test_load_config_json_validates_types_choices_and_bounds(self) -> None:
        invalid_payloads = [
            {"passes": True},
            {"min_points": False},
            {"score_thr": True},
            {"thr_car": False},
            {"min_points": 5.0},
            {"passes": 4.0},
            {"passes": 0},
            {"voxel_size": 0},
            {"range_m": -1},
            {"min_points": -1},
            {"intensity_mode": "typo"},
            {"ground_mode": "typo"},
            {"score_thr": None},
            {"extra": "not_a_dict"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.json"
            for payload in invalid_payloads:
                with self.subTest(payload=payload):
                    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(SystemExit):
                        pp_prelabel.load_config(str(cfg_path), [])

            cfg_path.write_text(
                json.dumps({"range_m": 50, "thr_car": 0.3, "extra": {"note": "Đợt 1"}}),
                encoding="utf-8-sig",
            )
            valid_cfg = pp_prelabel.load_config(str(cfg_path), [])
            self.assertEqual(valid_cfg.range_m, 50.0)
            self.assertIsInstance(valid_cfg.range_m, float)
            self.assertEqual(valid_cfg.thr_car, 0.3)
            self.assertEqual(valid_cfg.extra, {"note": "Đợt 1"})

            cfg_path.write_text(json.dumps({"voxel_size": 0}), encoding="utf-8")
            repaired_cfg = pp_prelabel.load_config(str(cfg_path), ["voxel_size=0.16"])
            self.assertEqual(repaired_cfg.voxel_size, 0.16)


if __name__ == "__main__":
    unittest.main()
