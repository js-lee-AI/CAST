# Measured summaries

Every table that the scripts in `experiments/` rebuild is a function of the files in this folder, so all of them run on a CPU. The files hold timings and drafter scores only, with no prompt or output text.

| file | what it holds | used for |
|---|---|---|
| `decoding.json` | decode-only ms/token and committed round length R of AR, DFlash and CAST at each width, by setting, run and domain | Tables 1, 2, 4, 7, 8, 9, 12, 16 and 17, Figures 3 and 5 |
| `prompt_level.json` | per-prompt ms/token of DFlash and CAST at N\*, averaged over three repeats, on 200 GSM8K, 80 MT-Bench and 164 HumanEval prompts | Table 3, greedy rows |
| `sampling.json` | per-seed ms/token and mean R at T=1 with the stochastic verifier, three seeds | Table 3, sampling row, and Table 10 |
| `cost_probes.json` | batch-one target-forward latency by packed-token count, with the drafter forward, for each GPU, target and KV-prefix length | Tables 5 and 6, Figure 2 |
| `masses.json` | sorted prefix masses ρ<sub>(N)</sub>, N = 1 to 128, from 1,500 logged greedy rounds of Qwen3-8B in each domain | Figure 4c, and the default mass curve of the width rule |

A setting in `decoding.json` looks like this, with one entry in `runs` for each repeat of the whole sweep.

```json
{"gpu": "Blackwell", "device": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
 "target": "Qwen/Qwen3-8B", "drafter": "z-lab/Qwen3-8B-DFlash-b16", "block": 16,
 "temperature": 0.0, "n_star": 63, "prompts": 40,
 "runs": [{"gsm8k": {"max_new": 256, "ar_ms": 18.00889,
                     "dflash": {"ms": 4.24998, "R": 6.01525},
                     "cast": {"15": {"ms": 3.98402, "R": 6.71001}, "31": {"ms": 3.74744, "R": 7.46531}}}}]}
```

`n_star` is the width the paper deploys in that setting. The H100 SXM settings at T=1 hold three-seed means. Blackwell settings hold three repeats, and each paper cell comes from the repeat with the median gain. The two `a6000_second_host_*` settings are the second A6000 host that measured N=127.

All timings are decode-only steady state in bf16 at batch one (paper Appendix D). The mass curve of GSM8K is also packaged with the library as `cast_trees.GSM8K_MASSES`, since the paper uses that one curve for every deployment.
