import subprocess
import sys
from pathlib import Path

import pytest

import cast_trees

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True,
                          check=True, timeout=120).stdout


def test_import_does_not_pull_torch():
    # A fresh interpreter, since pytest plugins may have imported torch already.
    out = run("-c", "import sys, cast_trees; print('torch' in sys.modules)")
    assert out.strip() == "False"


def test_quickstart_prints_what_the_readme_shows():
    out = run(str(ROOT / "examples" / "quickstart.py"))
    assert "the rule stops at N=120, N* = 127" in out
    assert "CAST, N=127   127 nodes, expected accepted tokens 3.54" in out


def test_cli_help_and_demo():
    assert "demo" in run("-m", "cast_trees", "--help")
    assert "N* = 127" in run("-m", "cast_trees", "demo")


def test_cli_width_on_a_probe_file(tmp_path):
    probe = tmp_path / "cost.json"
    probe.write_text('{"verify_ms": {"1024": {"16": 30.6, "32": 30.6, "64": 31.5, "128": 32.6}}, "draft_ms": 3.8}')
    out = run("-m", "cast_trees", "width", "--probe", str(probe))
    assert "N* = 127" in out


def test_lazy_names_are_listed():
    assert "cast_generate" in dir(cast_trees)
    with pytest.raises(AttributeError):
        cast_trees.no_such_name  # noqa: B018


@pytest.mark.gpu
def test_topk_marginals_on_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    idx, lp = cast_trees.topk_marginals(torch.randn(15, 1000, device="cuda"), 8)
    assert idx.shape == (15, 8) and lp.device.type == "cpu"
