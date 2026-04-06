"""Tests for --output json CLI flag."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

BRIEF = str(Path(__file__).parent.parent / "briefs" / "13b.yaml")
CLI = [sys.executable, "-m", "causalcompute.cli"]


def _run_json(brief=BRIEF, extra_args=None):
    """Run the CLI with --output json and return the parsed dict."""
    cmd = CLI + [brief, "--output", "json"] + (extra_args or [])
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    return json.loads(result.stdout)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

class TestStructure:
    def test_output_is_valid_json(self):
        cmd = CLI + [BRIEF, "--output", "json"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        parsed = json.loads(result.stdout)  # raises if invalid
        assert parsed is not None

    def test_top_level_keys_present(self):
        d = _run_json()
        for key in ("feasible", "bundle0", "design", "bottleneck",
                    "thermals", "network", "storage", "bom", "cost"):
            assert key in d, f"missing key: {key}"

    def test_feasible_is_bool(self):
        d = _run_json()
        assert isinstance(d["feasible"], bool)

    def test_feasible_true_for_valid_brief(self):
        d = _run_json()
        assert d["feasible"] is True

    def test_all_pipeline_sections_non_null_when_feasible(self):
        d = _run_json()
        if d["feasible"]:
            for key in ("bundle0", "design", "bottleneck", "thermals",
                        "network", "storage", "bom", "cost"):
                assert d[key] is not None, f"section '{key}' is null for feasible design"

    def test_bundle0_has_expected_subsections(self):
        d = _run_json()
        b0 = d["bundle0"]
        for section in ("req", "instant", "device_bounds", "movement",
                        "stepfacts", "timecmp", "ckpt", "meta"):
            assert section in b0

    def test_design_has_solution_when_feasible(self):
        d = _run_json()
        if d["feasible"]:
            assert d["design"]["solution"] is not None

    def test_bottleneck_has_binding_key(self):
        d = _run_json()
        if d["feasible"]:
            assert "binding" in d["bottleneck"]
            assert d["bottleneck"]["binding"]["key"] in ("compute", "state", "instant")

    def test_cost_has_cost_per_token(self):
        d = _run_json()
        if d["cost"] is not None:
            cpt = d["cost"]["cost_per_token"]
            assert "amortised" in cpt
            assert "full_capex" in cpt


# ---------------------------------------------------------------------------
# JSON validity — no inf/nan
# ---------------------------------------------------------------------------

class TestFiniteValues:
    def _collect_floats(self, obj):
        """Recursively collect all float values in a nested structure."""
        floats = []
        if isinstance(obj, float):
            floats.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                floats.extend(self._collect_floats(v))
        elif isinstance(obj, list):
            for v in obj:
                floats.extend(self._collect_floats(v))
        return floats

    def test_no_inf_in_output(self):
        d = _run_json()
        for f in self._collect_floats(d):
            assert not math.isinf(f), f"inf found in JSON output: {f}"

    def test_no_nan_in_output(self):
        d = _run_json()
        for f in self._collect_floats(d):
            assert not math.isnan(f), f"nan found in JSON output: {f}"

    def test_output_round_trips(self):
        """JSON output can be serialized again without error."""
        d = _run_json()
        re_encoded = json.dumps(d)
        re_decoded = json.loads(re_encoded)
        assert re_decoded["feasible"] == d["feasible"]


# ---------------------------------------------------------------------------
# CLI behaviour
# ---------------------------------------------------------------------------

class TestCLIBehaviour:
    def test_json_output_goes_to_stdout_only(self):
        """With --output json, stderr should be empty on success."""
        cmd = CLI + [BRIEF, "--output", "json"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert result.stderr == ""

    def test_text_output_is_default(self):
        """Without --output, output is human-readable text, not JSON."""
        cmd = CLI + [BRIEF]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        with pytest.raises((json.JSONDecodeError, ValueError)):
            json.loads(result.stdout)

    def test_invalid_brief_exits_nonzero(self):
        cmd = CLI + ["nonexistent_brief.yaml", "--output", "json"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode != 0

    def test_json_and_story_flags_independent(self):
        """--story prints text narrative and exits; --output json is irrelevant."""
        cmd = CLI + [BRIEF, "--story"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        # story output is text, not JSON
        with pytest.raises((json.JSONDecodeError, ValueError)):
            json.loads(result.stdout)
