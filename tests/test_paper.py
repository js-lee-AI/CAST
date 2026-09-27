"""Every experiments/ script reproduces the numbers the paper prints."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(script):
    return subprocess.run([sys.executable, str(ROOT / "experiments" / script)], capture_output=True,
                          text=True, check=True, timeout=300).stdout


@pytest.mark.parametrize("script, fragments", [
    ("table1_h100.py", ["Q3-8B CAST (N*=127)     |  5.81x 8.46 |  6.74x 9.87 |  6.16x 9.04 |  6.25x 9.09 |  3.16x 5.52 |  5.62x 8.40",
                        "Q3-4B CAST (N*=127)     |  5.79x 8.47 |  6.62x 9.76 |  6.04x 9.02 |  6.34x 9.11 |  3.29x 5.78 |  5.62x 8.43",
                        "Q3-8B DFlash (16)       |  4.11x 5.69",
                        "MBPP           512  24.76   4.99   4.04   4.03   3.94   3.90    6.72   9.11"]),
    ("table2_width_rule.py", ["Blackwell, Qwen3-8B (b16)                63     +19.8%      95   +21.5%  -1.7       +2.4%",
                              "Blackwell, Qwen3-4B (b16)                95     +27.3%     127   +29.0%  -1.7      +29.0%",
                              "A6000, Qwen3-4B (b16)                    95     +32.6%      95   +32.6%  +0.0     +31.7%",
                              "H100 SXM, Qwen3-Coder-30B-A3B (b16)     127     +35.6%     127   +35.6%  +0.0      +35.6%"]),
    ("table3_bootstrap.py", ["+18.1% [16.4, 19.8]", "+33.2% [29.4, 37.0]", "+22.3% [20.3, 25.3]",
                             "+16.7%  +16.0 to +18.0%"]),
    ("table4_llama.py", ["CAST (N*=127)         3.82x      3.59x      4.38x      4.49x      3.05x      3.87x"]),
    ("table5_cost_probes.py", ["Blackwell, Qwen3-8B              17.2   21.9   23.3   23.6   24.1   24.6   24.4   30.3",
                               "Qwen3-8B          16k   23.7   56.7   58.0   59.9   60.0   0.029"]),
    ("table7_cross_hardware.py", ["Blackwell 8B, N*=63        GSM8K   18.0    4.25   3.65  +16.4%  7.97",
                                  "A6000     Qwen3-4B   HumanEval   +15.6%    +6.3%   +28.5%   +28.0%   +30.4%  +29.1%",
                                  "Blackwell Qwen3-4B    MT-Bench     3.11    4.00      8.77     6.79     +29.2%"]),
    ("table17_coder.py", ["AR ms/token                62.3         63.8         62.6",
                          "CAST (N*=127)         5.34 6.86   7.78 10.34    6.86 8.89    6.66 8.70"]),
    ("figure4c_masses.py", ["GSM8K  0.8595  0.5112  0.2940  0.0760  0.0159  0.0112"]),
])
def test_script_reproduces_the_paper(script, fragments):
    out = run(script)
    for f in fragments:
        assert f in out, f
