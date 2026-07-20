# Specializing Speculative-Decoding Drafters with LoRA: Headroom, Interference, and the Train–Serve Alignment Tax

**Shrey Birmiwal** · draft v0.1, 2026-07-20
*(target: MLSys / EMNLP-industry / arXiv preprint — ~8 pages + appendix)*

---

## Abstract

Speculative decoding accelerates LLM inference losslessly, but its drafters are
small models whose acceptance rates swing by an order of magnitude across
domains (4–39% for a 1B block-diffusion drafter; 5–63% for an EAGLE3 head, over
51 domains). We ask whether cheap LoRA adapters (0.01–0.2% of drafter
parameters), trained by self-distillation on the target's own generations, can
close these gaps — and whether one *combined* adapter matches per-domain
specialists. Across three drafter architectures for a frozen Qwen3-8B target,
we find: **(1) a headroom law** — LoRA gain is inversely proportional to the
base drafter's strength on that domain, and rank keeps paying exactly where the
base is weakest; **(2) near-free combining** — a single adapter exactly matches
specialists at 3–5 domains (even at rank 4), and retains ~2/3 of specialist
gains at 40 domains, with an interference tax that *saturates* (−0.3pp) rather
than growing, concentrated on the highest-gain domains; **(3) an alignment
caveat to the headroom law** — a strong feature-conditioned drafter (EAGLE3)
resists specialization not because it lacks headroom but because its training
must reproduce the serving-time feature pipeline exactly, whereas an
*independent* drafter (Qwen3-0.6B, vanilla two-model speculation) specializes
with 30 lines of plain cross-entropy on 5/5 domains despite the highest base
acceptance we test (up to 77%). Full fine-tuning of the 1B drafter matches base
at best, twice. All configurations are lossless at temperature 0. We
characterize the resulting design space: feature-conditioned drafters buy cheap
drafts at the price of a fragile training contract; independent drafters buy a
trivial training contract at the price of expensive drafts — *specialization is
easy exactly where speculation is slow.*

---

## 1 Introduction

Speculative decoding [Leviathan et al. 2023; Chen et al. 2023] commits several
tokens per target forward pass by letting a small drafter propose and the
target verify, with provably unchanged outputs. Its economics rest entirely on
the drafter's acceptance rate — and drafters are, by construction, models too
small to cover their target's full distribution. We first quantify how badly
this bites: benchmarking two public drafters for Qwen3-8B across 51 domains
(natural languages, programming languages, tasks, out-of-distribution
specialties), acceptance spans **4.0–38.5%** for the DFlash 1B block-diffusion
drafter and **5.2–63.3%** for the EAGLE3 head. The ranking is consistent —
math/structured tasks on top, non-Latin-script languages at the bottom — and
the bottom is exactly where speculation stops paying.

This suggests an appealingly cheap fix: per-domain LoRA adapters on the
drafter, optionally selected by a router — a "mixture of LoRAs" for
speculation. Small drafters should suffer *interference* when one set of
weights must serve many domains; specialists plus routing should beat one
adapter trained on everything. This paper tests that hypothesis end to end and
finds the design space is shaped by two forces the hypothesis missed.

**Contributions.**

1. **A 51-domain acceptance atlas** for two public Qwen3-8B drafters
   (block-diffusion and EAGLE3), on synthetic, wild (WildChat), and
   downloaded-HF prompt sources (§4).
2. **The headroom law** (§5, §6): LoRA gain is inversely proportional to base
   drafter strength per domain. Weak domains (3–5% base) gain up to +66%
   relative and keep improving with rank (r64 ≫ r16); moderate domains saturate
   by rank 16 (rank 4 already captures ~90%); strong domains gain nothing.
   Full fine-tuning (1B params, ≤2M supervised tokens) matches base at best.
3. **Interference is small and saturating** (§5.4): one combined adapter
   *exactly* matches specialists at 3–5 heterogeneous domains at every rank
   tested (4/16/64); the first statistically measurable gap appears at N≈10
   domains (−0.21pp, 95% CI excluding 0), saturates by N=40 (−0.28pp, ~67%
   of specialist gain retained), and concentrates on the domains with the
   largest specialist gains. There is no phase boundary through 40 domains.
4. **The alignment-channel caveat** (§5.5–5.6): the headroom law is
   per-alignment-channel, not absolute. EAGLE3 (strong, feature-conditioned)
   gains only on its single weakest domain, which we trace to the fragility of
   reproducing its serving-time feature pipeline at training time — not to
   saturation: an independent Qwen3-0.6B drafter with *higher* per-proposal
   acceptance (35–77%) gains on 5/5 domains (+2.6..+5.5pp) trained with plain
   CE, where train and serve channels coincide trivially.
5. **Systems accounting** (§6): where specialization pays most (weak domains),
   base acceptance is too low for the gain to move end-to-end wall-clock at
   batch 1; where acceptance is high, little is left to specialize. The
   deployable recipe is one combined LoRA, with routing reserved for the few
   high-gain domains — and we show routing is essentially free (a 3-layer MLP
   on target hidden states already computed at prefill reaches 100% test
   accuracy on 596 held-out prompts).

We also document, in the spirit of honest reporting, the training-side bugs
that invalidated our first EAGLE results — including a variable-shadowing bug
that fed the head *reversed* auxiliary features every training step — and the
byte-exactness verification suite that localized them (Appendix A).

## 2 Background and related work

**Speculative decoding.** Draft-and-verify sampling [Leviathan et al. 2023;
Chen et al. 2023] is lossless: at temperature 0 the emitted text equals the
target's greedy output regardless of drafter quality; the drafter changes only
speed. Architectures divide into *independent* drafters (a separate small LM;
the original formulation) and *feature-conditioned* drafters that consume the
target's hidden states — single-layer autoregressive heads (EAGLE/EAGLE3
[Li et al. 2024]) and block-diffusion drafters that propose a block per forward
(DFlash). Feature conditioning buys dramatically cheaper drafts; we show it
also creates a training contract that dominates specialization behavior.

**Parameter-efficient adaptation.** LoRA [Hu et al. 2021] and multi-adapter
serving (S-LoRA, Punica) make per-domain specialization operationally cheap:
adapters can be hot-swapped or batched over one resident backbone. Mixture-of-
LoRA / routing approaches (MoLA and successors) assume specialists beat a
merged generalist; our interference ladder quantifies exactly how much that
assumption is worth for drafters (answer: −0.2..−0.3pp at 10–40 domains).

**Drafter adaptation.** Prior work trains drafters from scratch per
task or online-adapts them. We instead ask the minimal question — can a
frozen public drafter be specialized with ~2M supervised tokens and ~0.1% new
parameters? — and use *self-distillation*: the target generates every training
answer, so the drafter learns the target's actual conditional distribution
(including its errors), never the dataset's gold answers.

## 3 Experimental setup

| component | choice |
|---|---|
| target | `Qwen/Qwen3-8B`, frozen everywhere |
| drafter A | **DFlash** `z-lab/Qwen3-8B-DFlash-b16` — 1B block-diffusion, 15 drafts/step |
| drafter B | **EAGLE3** `RedHatAI/Qwen3-8B-speculator.eagle3` — 1-layer AR head, 3 drafts/step |
| drafter C | **independent** `Qwen/Qwen3-0.6B` — vanilla two-model speculation, k=4, no feature conditioning |
| adaptation | unmerged LoRA on q/k/v/o; rank 16, α=32 unless stated (~2M params ≈ 0.2% of drafter A) |
| training signal | self-distillation on the target's own generations, conditioned on the target's live hidden states (A, B) or token stream only (C) |
| losses | A: SpecForge exponentially-weighted block CE (γ=7); B: canonical `speculators` TTT forward (3-step unroll, soft distillation); C: plain CE |
| evaluation | temperature 0, held-out test prompts, n=100/domain (150 for legal); pooled acceptance rate, mean accept length, speedup vs target-only |
| data | 800 train / 100 val / 100 test prompts per domain (8000 train for legal) |

**Metrics.** Acceptance rate (accepted ÷ proposed drafts) is the only fair
cross-drafter number — it normalizes for drafts-per-step (15 vs 3 vs 4). Mean
accept length (tokens committed per target pass) is a within-drafter speed
proxy with different ceilings (~16 / ~4 / ~5). All runs are integrity-checked
to full n; all results regenerate from committed per-example logs.

**Prompt sources.** Synthetic (Claude-generated, 51 domains), wild (real
WildChat prompts sorted into the same domains), and downloaded (purpose-built
HF datasets: SQL, legal, medical, financial) — the latter two control for
synthetic prompts inflating acceptance.

## 4 The acceptance atlas: base drafters across 51 domains

Both drafters rank domains similarly (Spearman agreement is high): structured
generation on top (math reasoning 38.5% / 50.6% for DFlash/EAGLE3; JSON
extraction 34.1% / 63.3%), code in the middle (18–21% / 44–48%), non-English
languages at the bottom (Korean 4.0% / 9.2%; Japanese 5.3% / 6.0%; Arabic
5.1% / 5.2%). Pooled acceptance: DFlash 11.7%, EAGLE3 29.9%. The spread is
~10× within each drafter — out-of-distribution domains do not degrade tiny
speculators gracefully; they collapse them. EAGLE3's spread is, if anything,
the wider one (12× vs 10×), concentrated in its language tail.

*(Figures: per-domain bars for each drafter × source; drafter overlay;
synthetic-vs-wild-vs-downloaded comparison. From `benchmarking/results/charts/`.)*

## 5 Results

### 5.1 LoRA beats full fine-tuning (and full fine-tuning does nothing)

DFlash, two downloaded-HF domains:

| | code_sql (n=100) ||| indian legal (n=150) |||
|---|--:|--:|--:|--:|--:|--:|
| | accept | len | speedup | accept | len | speedup |
| base | 25.0% | 5.07 | 3.70× | 10.9% | 2.67 | 1.94× |
| full FT (1.05B params) | 25.1% | 5.12 | 3.68× | 11.1% | 2.70 | 1.96× |
| **LoRA r16** | **28.3%** | **6.03** | **4.11×** | **13.5%** | **3.09** | **2.11×** |

Full fine-tuning ≈ base twice: moving 10⁹ parameters with ≤2×10⁶ supervised
tokens is data-starved. The 2M-parameter adapter specializes cleanly on the
same data. Emitted text is exact-match identical to the target's greedy output
in every variant (losslessness check).

### 5.2 Multilingual specialization and rank scaling (DFlash)

Five languages; one LoRA each plus one combined adapter trained on all five:

| language | base | own r16 | comb r16 | own r64 | comb r64 |
|---|--:|--:|--:|--:|--:|
| polish | 3.1% | 4.4% | 4.5% | **5.0%** | 5.1% |
| korean | 3.5% | 5.1% | 5.1% | **5.8%** | 5.8% |
| italian | 8.1% | 8.5% | 8.4% | 8.7% | 8.5% |
| japanese | 5.0% | 5.9% | 5.9% | 6.3% | 6.2% |
| german | 6.8% | 7.2% | 7.1% | 7.3% | 7.1% |

Own beats base 5/5. Relative gains concentrate where base is weakest (Korean
+46% at r16, +66% at r64; Polish +42%/+61%), and *rank keeps paying only
there*: r64's extra gain is +0.6–0.7pp on Polish/Korean vs +0.1–0.2pp on
German/Italian. Combined ≈ own at both ranks — five languages coexist in one
adapter with zero measurable interference.

### 5.3 Heterogeneous tasks and the rank-4 result (DFlash)

Translation / roleplay / poetry — deliberately different *task types*, the
stronger interference test:

| domain | base | r4 own | r4 comb | r16 own | r16 comb | r64 own | r64 comb |
|---|--:|--:|--:|--:|--:|--:|--:|
| translation | 8.7% | 9.2% | 9.4% | 9.5% | 9.4% | 9.4% | 9.1% |
| roleplay | 8.1% | 8.5% | 8.3% | 8.5% | 8.5% | 8.5% | 8.4% |
| poetry | 7.0% | 7.5% | 7.4% | 7.6% | 7.6% | 7.8% | 7.6% |

Own beats base 3/3 at every rank. **Rank 4 (~130K params, 0.01% of the
drafter) captures ~90% of the achievable gain**, and combined matches own even
at rank 4 — three different skills share the scarcest capacity without
conflict. Together with §5.2 this suggests domain adaptation here is a broad,
intrinsically low-rank "steering" direction, not stored knowledge; gains
saturate by r16 when base acceptance is moderate (7–9%) and keep scaling with
rank only in the 3–5%-base regime.

### 5.4 The interference ladder: 10, 20, 40 domains in one adapter

Core-plus-distractors design: 10 diverse evaluated domains (each with its own
specialist), plus 10/30 distractor domains that join only the combined
training sets (comb10 ⊂ comb20 ⊂ comb40; 800 examples/domain; rank 16).

Mean combined−own gap (paired bootstrap, 95% CI):

| N domains | gap | CI |
|--:|--:|---|
| 3–5 (§5.2–5.3) | ≈ 0 | — |
| 10 | −0.21pp | [−0.29, −0.13] |
| 20 | −0.27pp | [−0.34, −0.20] |
| 40 | −0.28pp | [−0.36, −0.19] |

Findings: (i) combined beats base 10/10 at every N; (ii) the gap becomes
statistically measurable at N≈10 and then **saturates** — going 20→40 costs
~0.01pp; no phase boundary; (iii) the tax concentrates on the domains with the
largest specialist gains (math reasoning gives back −1.1pp of its +2.0pp own
gain at N=40; Korean/Polish −0.3/−0.4pp of +1.6/+1.3pp) and is near zero on
small-gain domains. Combined retains ~74% of mean specialist gain at N=10,
~67% at N=40. Interpretation: the shared steering component is free to share;
only the largest domain-specific shifts compete for the low-rank subspace.

**Systems vs science.** As a deployment decision the tax is invisible: mean
end-to-end speedup is base 2.21×, specialists 2.20×, comb40 2.16× — inside
container-to-container wall-clock noise (±0.05×), which does not even separate
specialists from base. As a science result it is real: measured against the
right denominator (the specialization gain itself), the combined adapter gives
back about a third.

### 5.5 EAGLE3: the strong feature-conditioned head resists

Same data and recipe on the EAGLE3 head (after fixing the training bugs of
Appendix A; merge and vLLM-serve paths verified byte-exact):

| multilingual | base | own | comb | | weird | base | own | comb |
|---|--:|--:|--:|---|---|--:|--:|--:|
| polish | 9.1% | 6.0% | 5.9% | | translation | 20.6% | 19.2% | 17.8% |
| korean | 7.9% | 7.2% | 6.4% | | roleplay | 36.0% | 33.5% | 32.7% |
| italian | 13.9% | 10.0% | 9.8% | | poetry | 33.5% | 29.4% | 27.9% |
| **japanese** | **4.9%** | **6.4%** | 6.5% | | | | | |
| german | 12.2% | 8.3% | 8.5% | | | | | |

The only gain is Japanese (+1.6pp, +33% relative) — precisely EAGLE3's weakest
base domain (4.9%). Strong-base domains tip *negative* rather than flat, which
we attribute to a residual train/serve feature-reconstruction gap: the public
head's TTT loss remains ~20 under our reconstructed training view, implausibly
high for a head this strong on its own training convention. The verification
suite (Appendix A) confines any residual error to training-side feature
reconstruction. On its own, this result is ambiguous between "no headroom" and
"broken channel" — §5.6 disambiguates.

### 5.6 The independent drafter: strongest base, easiest specialization, slowest drafts

Vanilla two-model speculation with an off-the-shelf Qwen3-0.6B drafting k=4
tokens autoregressively; the only train/serve channel is the token stream;
training is plain CE on the target's generations (a 30-line loop, zero
training-side bugs). Five domains spanning the headroom curve:

| domain | base | own | combined |
|---|--:|--:|--:|
| code_sql | 50.0% | **55.5%** (+5.5pp) | 54.7% |
| lang_polish | 35.5% | **39.7%** (+4.1pp) | 38.2% |
| lang_korean | 37.3% | **40.9%** (+3.6pp) | 38.2% |
| ood_legal | 41.8% | **47.0%** (+5.2pp) | 42.4% |
| task_math_reasoning | 77.3% | **79.9%** (+2.6pp) | 78.1% |

Three results: **(i)** specialization works 5/5 with the largest per-proposal
gains of any drafter tested — on base acceptances of 35–77%, far above
EAGLE3's. Strong base alone does not kill specialization; a leaky alignment
channel does. EAGLE3's §5.5 flatness now reads as *alignment*, not
saturation. **(ii)** Interference arrives early: combined < own on 5/5
(−0.8..−4.6pp) at N=5, where DFlash showed exactly zero — consistent with
§5.4's law, since these are the largest specialist gains in the study.
**(iii)** The method loses wall-clock at batch 1 (0.58–0.97× base): eager
decoding latency is layer-count-bound, so a 28-layer 0.6B forward (~20ms)
costs nearly as much as the 36-layer 8B target's (~21–27ms); four drafts plus
one verify ≈ five target-tokens of latency for 2.4–4.2 committed tokens. This
is precisely the niche single-forward and single-layer drafters exist to fill.

### 5.7 Routing is free

An MLP (20480→512→6) over the target's hidden states at the drafter's own
conditioning layers — tensors already computed during prefill, so routing
costs one small matmul — classifies 5 languages + "other" (trained with hard
negatives: English/French/Spanish, code, tasks) at **100% accuracy on 596
held-out prompts** with a perfect confusion diagonal, and plugs directly into
per-sequence multi-adapter serving over a shared backbone. Routing is not the
bottleneck; whether specialists are worth routing *to* is (§5.4).

## 6 Analysis: the headroom law, per alignment channel

Every experiment fits one curve, with one crucial caveat:

| base acceptance | LoRA gain | evidence |
|---|---|---|
| 3–5% (drafter nearly useless) | large; rank keeps paying (r64 ≫ r16) | Polish, Korean; Japanese-EAGLE |
| 7–10% (weak) | solid; saturates by r16 (≈ r4) | weird domains, Italian, German |
| ~25% (decent, templated) | largest absolute (+3.4pp) | code_sql (low-entropy output) |
| 12–36% (strong, feature-conditioned) | ≈0 to negative | EAGLE3 strong-base domains |
| 35–77% (strongest, independent) | **+2.6..+5.5pp** | independent drafter, 5/5 |

The last two rows are the point: the law is **per alignment channel**. Where
the training channel trivially equals the serving channel (independent
drafter: tokens in, tokens out), specialization keeps paying even at 77% base
acceptance. Where training must *reconstruct* the serving-time feature
pipeline (EAGLE3: which layers, what normalization, what concatenation order),
the channel itself is the binding constraint — three of the four bugs we hit
lived there, and a residual gap plausibly explains the negative tail.

**The deployable recipe.** One combined LoRA per drafter; route per-domain
only for the few high-gain domains the interference ladder flags (or for
tenant isolation). Full fine-tuning is dominated at this data scale. And the
drafter-architecture choice is a two-sided trade the field should name:
feature-conditioned drafters buy cheap drafts at the price of a fragile
training contract; independent drafters buy a trivial training contract at the
price of expensive drafts. **Specialization is easy exactly where speculation
is slow.**

**The wall-clock asymmetry.** Acceptance-pp gains translate to end-to-end
speedup superlinearly in base acceptance — so the weak domains where LoRA
gains most (3–8% base) are exactly where even +66% relative leaves speculation
barely profitable, while SQL's +3.4pp on a 25% base bought +0.41×. Reviewers
should read the acceptance tables as the science and the speedup columns as
the (noisier, batch-1) systems reality.

## 7 Limitations

- **One target model** (Qwen3-8B) and one target scale; the headroom and
  interference laws should be replicated across targets and drafter scales.
- **n=100 per domain** (150 for legal) and a single training seed per cell;
  the interference-ladder CIs are bootstrap-over-prompts, not over seeds.
- **Batch-1, greedy, temperature-0 evaluation.** Batched serving changes
  speedup arithmetic (not acceptance); sampled decoding changes acceptance.
- **The EAGLE3 negative result is partially unresolved:** we verified the
  merge and serve paths byte-exact but could not fully close the training-side
  feature-reconstruction gap (base TTT loss ~20). The alignment interpretation
  rests on the independent-drafter contrast, not on a repaired EAGLE run.
- **Synthetic prompts dominate training data**; wild/downloaded controls exist
  for the base atlas but not for every fine-tuning cell.
- Wall-clock numbers carry ±0.05× container-to-container noise; we therefore
  make no speedup claims finer than that.

## 8 Future work

Latent (unlabeled) domain discovery — cluster the same target hidden states
the router consumes and train adapters per cluster, testing whether learned
partitions beat named ones. Harness/agent-trace specialization (Claude Code,
Codex-style scaffolds): heavily templated, low-entropy output resembling SQL,
the regime with our largest absolute gain — likely the highest-value
deployment. Closing the EAGLE training channel by capturing serving-path
features directly (e.g., from the vLLM forward itself) instead of
reconstructing them. A *shallow* independent drafter (or early-exit draft
path) to keep the trivial CE alignment channel while making drafts cheap
enough to win wall-clock.

## 9 Conclusion

For speculative decoding, parameter-efficient specialization is reliable,
nearly interference-free at deployment-relevant domain counts, and governed by
two quantities you can measure before training anything: the drafter's base
acceptance on the domain (headroom) and the fidelity of its train/serve
alignment channel. The mixture-of-specialists hypothesis we set out to confirm
survives only in a weak form — specialists beat one combined adapter by ~1/3
of the specialization gain at 40 domains, a tax that is real to science and
invisible to systems.

---

## Appendix A: Negative results and the verification suite

Our first EAGLE3 runs (archived, never deleted) were invalidated by a chain of
training-side bugs, reported here because the diagnostic path is half the
value: (1) a torch/torchvision ABI clash; (2) flex-attention packs not padded
to the 128-token block-mask size; (3) double-normalized distillation targets
(HF's `hidden_states[-1]` is already post-final-norm); and (4) the killer — a
variable-shadowing bug in which the auxiliary-layer concatenation convention
string was clobbered by the epoch shuffle list, so every v1 training step fed
the head **reversed** aux features ([33,18,2] at train vs [2,18,33] at serve).
The adapters were trained to fight a scrambled input.

The resulting verification suite is the reusable artifact: (a) a zero-adapter
merge must be byte-identical to hub weights; (b) a zero-adapter benchmark
through the serving stack must reproduce base digit-for-digit; (c) a held-out
objective test separates "training broken" from "objective misaligned with
serving." (a) and (b) pass, localizing any residual EAGLE error to
training-side feature reconstruction. Operational rules that mattered:
detached spawn-only launches (flaky client networks silently duplicate jobs),
and never trusting a result file without an n-count integrity check (stopping
containers mid-run commits partial files over complete ones).

## Appendix B: Reproducibility

All results regenerate from committed per-example logs
(`experiments/*/results/`); every pipeline exposes `launch` (detached),
`agg_only`, and chart entry points. Trained adapters live on the Modal volume
`code-sql-pipeline`. Data: 51-domain synthetic corpus (Claude-generated),
WildChat-sorted wild prompts, and HF downloads, in identical
train/val/test.jsonl layout under `data/`.
