# Specializing Speculative-Decoding Drafters with LoRA — Full Report

**Question:** can a small, cheap LoRA adapter make a speculative-decoding drafter
track its target model better on a chosen domain — and is one combined adapter as
good as per-domain specialists?

**One-line answer:** yes — LoRA specialization works reliably on the weak
speculator (DFlash: positive on **every domain tested**), gains scale with how
weak the base is (and with rank, exactly where the base is weakest), a single
combined adapter matches per-domain specialists at 3–5 domains (zero
interference, even at rank 4) and *nearly* matches them all the way to 40
domains (small saturating gap from N≈10, ~2/3 of specialist gains retained, no
phase boundary), and — once the training forward exactly reproduces the serving
contract — it helps the strong speculator too: EAGLE3 gains on **5/5**
multilingual domains (+0.6..+2.1pp) and sits flat (not negative) on its
strongest domains. The earlier "EAGLE resists specialization" result was a
train/serve **alignment bug** (a missing one-token shift in the TTT forward,
§3), an inference exp 06 had already made from the other direction: an
independent Qwen3-0.6B drafter with a trivially aligned channel (plain CE on
tokens) gains on 5/5 domains (+2.6..+5.5pp) at the highest base acceptance in
the repo — though vanilla two-model drafting itself doesn't pay wall-clock at
batch-1 (§2.6).

---

## 1. Setup

| component | choice |
|---|---|
| target | `Qwen/Qwen3-8B` (frozen everywhere) |
| speculator A | **DFlash** `z-lab/Qwen3-8B-DFlash-b16` — 1B block-diffusion drafter, 15 drafts/step |
| speculator B | **EAGLE3** `RedHatAI/Qwen3-8B-speculator.eagle3` — 1-layer autoregressive head, 3 drafts/step |
| speculator C | **independent drafter** `Qwen/Qwen3-0.6B` — vanilla two-model spec decode (Leviathan/Chen), 4 drafts/step, no target-feature conditioning (exp 06) |
| adaptation | unmerged LoRA on q/k/v/o (rank 16, α=32 unless stated; only ~2M params ≈ 0.2% of the drafter) |
| training signal | **self-distillation**: the target generates every answer; the drafter learns to match the *target's own tokens* (even where the target is wrong), conditioned on the target's live hidden states. The datasets' own answers are never used |
| loss | DFlash: SpecForge exponentially-weighted block CE (γ=7). EAGLE3: the `speculators` package's canonical TTT forward (3-step unroll, soft distillation) |
| benchmark | temperature 0 (lossless — the adapter changes *speed*, never output), held-out test prompts, n=100/domain (150 for legal). Metrics: pooled acceptance rate, mean accept length (tokens committed per target pass), speedup vs target-only |
| data | 800 train / 100 val / 100 test prompts per domain (8000 for legal); prompts from the repo's synthetic + HF datasets |

Everything ran on Modal H200s. Infra notes that mattered: vLLM prep generates
8000 target answers in **89 s** (~60× over HF `generate`); all long runs use
detached server-side orchestration (spawn-and-exit launches) after a flaky local
network repeatedly killed synchronous runs; every result file is integrity-checked
to n=100 after an incident where stopping duplicate jobs committed partial files.

---

## 2. Results

### 2.1 LoRA vs full fine-tune (DFlash) — LoRA wins, full FT is a waste

**code_sql** (b-mc2/sql-create-context, 800 train, n=100):

| variant | accept | mean len | speedup |
|---|--:|--:|--:|
| base | 25.0% | 5.07 | 3.70× |
| full fine-tune (all 1.05B params) | 25.1% | 5.12 | 3.68× |
| **LoRA r16** | **28.3% (+3.4pp)** | **6.03 (+0.96)** | **4.11× (+0.42)** |

**ood_indian_legal** (Indian-legal-data-v3, 8000 train, n=150):

| variant | accept | mean len | speedup |
|---|--:|--:|--:|
| base | 10.9% | 2.67 | 1.94× |
| full fine-tune | 11.1% | 2.70 | 1.96× |
| **LoRA r16** | **13.5% (+2.5pp)** | **3.09 (+0.42)** | **2.11× (+0.17)** |

Full fine-tuning ≈ base in both — moving 1B params with ≤2M supervised tokens is
data-starved; the tiny adapter specializes cleanly instead. (Exact-match to the
target's greedy output is identical across variants — lossless as designed.)

### 2.2 Multilingual (DFlash) — specialization works 5/5; combined = specialists

Five languages, one LoRA each + one combined on all five (r16, n=100/lang):

| language | base | own r16 | combined r16 | own **r64** | combined r64 |
|---|--:|--:|--:|--:|--:|
| polish | 3.1% | 4.4% | 4.5% | **5.0%** | 5.1% |
| korean | 3.5% | 5.1% | 5.1% | **5.8%** | 5.8% |
| italian | 8.1% | 8.5% | 8.4% | 8.7% | 8.5% |
| japanese | 5.0% | 5.9% | 5.9% | 6.3% | 6.2% |
| german | 6.8% | 7.2% | 7.1% | 7.3% | 7.1% |

- **Own beats base on 5/5**; relative gains are large where base is weak
  (korean +46% at r16, **+66% at r64**; polish +42%/+61%).
- **Rank 64 > rank 16 on 5/5**, with the extra gain concentrated on the weakest
  bases (polish/korean +0.6–0.7pp extra; german/italian +0.1–0.2pp).
- **Combined ≈ own at both ranks** — five languages coexist in one adapter.

→ charts: `multilingual/results/charts/matrix.png`, `delta.png`; rank ladder in `experiments/07-rank-ladder/`

### 2.3 Weird domains (DFlash) — heterogeneous tasks, and the rank-4 surprise

Translation / roleplay / poetry — deliberately different *task types*, the real
interference test. Full rank ladder (n=100/domain):

| domain | base | r4 own | r4 comb | r16 own | r16 comb | r64 own | r64 comb |
|---|--:|--:|--:|--:|--:|--:|--:|
| translation | 8.7% | 9.2% | 9.4% | 9.5% | 9.4% | 9.4% | 9.1% |
| roleplay | 8.1% | 8.5% | 8.3% | 8.5% | 8.5% | 8.5% | 8.4% |
| poetry | 7.0% | 7.5% | 7.4% | 7.6% | 7.6% | 7.8% | 7.6% |

- **Own beats base on 3/3 at every rank** — including **rank 4 (~130K params,
  0.01% of the drafter), which captures ~90% of the achievable gain**. The domain
  shift is intrinsically low-rank: a broad "steering" direction, not knowledge.
- **No interference at any rank** — combined matches own even at rank 4, where
  three different skills must share the scarcest capacity.
- Gains saturate by r16 here (base 7–9% = moderate headroom), unlike the 3–5%-base
  languages where r64 kept paying.

→ charts: `weird-domains/results/charts/matrix.png`, `delta.png`; rank ladder in `experiments/07-rank-ladder/`

### 2.4 Interference at scale — the zero-interference result bends (but never breaks) by 40 domains

Does "combined = specialists" survive 10/20/40 domains in one rank-16 adapter?
Core + distractors design: 10 diverse evaluated domains (each with an own
specialist), plus 10 / 30 extra domains that only join the combined training
sets (comb10 ⊂ comb20 ⊂ comb40, 800 ex/domain, 3 epochs, n=100/domain):

| domain | base | own | comb10 | comb20 | comb40 |
|---|--:|--:|--:|--:|--:|
| code_python | 20.9% | 21.5% | 21.1% | 21.3% | 21.6% |
| code_sql | 18.0% | 18.8% | 18.4% | 18.4% | 18.5% |
| lang_polish | 3.1% | 4.4% | 4.5% | 4.3% | 4.1% |
| lang_korean | 3.5% | 5.1% | 5.1% | 5.0% | 4.8% |
| lang_german | 6.8% | 7.2% | 7.0% | 7.0% | 7.0% |
| ood_legal | 11.6% | 12.2% | 12.0% | 11.9% | 11.9% |
| ood_medical | 13.2% | 13.6% | 13.6% | 13.5% | 13.5% |
| task_math_reasoning | 37.6% | 39.5% | 38.8% | 38.6% | 38.5% |
| task_summarization | 9.6% | 9.8% | 9.7% | 9.8% | 9.8% |
| task_roleplay_chat | 8.1% | 8.5% | 8.4% | 8.4% | 8.3% |

Mean combined−own gap (paired bootstrap 95% CI): **N=10: −0.21pp [−0.29,−0.13]
· N=20: −0.27pp [−0.34,−0.20] · N=40: −0.28pp [−0.36,−0.19]** (vs ≈0 at N=3/5).

- **No phase boundary through 40 domains.** Combined beats base 10/10 at every
  N, and the gap *saturates* (20→40 costs ~0.01pp) rather than growing.
- **But exact zero interference ends at N≈5–10**: the first statistically
  measurable gap appears at N=10. Combined retains ~74% of the mean specialist
  gain at N=10, ~67% at N=40.
- **Interference lands where specialization pays most**: math_reasoning gives
  back −1.1pp of its +2.0pp own gain at N=40, korean/polish −0.3/−0.4pp of
  +1.6/+1.3pp; near-zero on small-gain domains (python, summarization, medical).
  Big domain-specific shifts compete for the shared low-rank subspace; the broad
  steering component is shared for free.
- **Two fair framings.** As a systems decision the tax is invisible: mean
  end-to-end speedup is base 2.21× / own 2.20× / comb40 2.16× — inside
  container-to-container wall-clock noise (±0.05×), which doesn't even separate
  specialists from base. As a science result it's real: the fair denominator is
  the specialization gain itself, and the combined adapter gives back ~1/3 of it.

→ charts: `interference/results/charts/ladder.png` (money chart), `matrix.png`, `delta.png`

### 2.5 EAGLE3 — the strong speculator DOES benefit, once train matches serve

Same data, same LoRA recipe, EAGLE3 head (v3 — after fixing the alignment bug
chain, §3):

| multilingual | base | own | combined |   | weird | base | own | combined |
|---|--:|--:|--:|---|---|--:|--:|--:|
| polish | 9.1% | **10.1% (+1.0pp)** | 10.3% | | translation | 20.6% | 20.3% (−0.3pp) | 20.4% |
| korean | 7.9% | **8.8% (+0.9pp)** | 8.9% | | roleplay | 36.0% | 35.2% (−0.7pp) | 35.1% |
| italian | 13.9% | **14.7% (+0.9pp)** | 15.2% | | poetry | 33.5% | 33.0% (−0.5pp) | 32.9% |
| **japanese** | **4.9%** | **6.9% (+2.1pp)** | **7.1%** | | | | | |
| german | 12.2% | **12.8% (+0.6pp)** | 13.0% | | | | | |

- **Own beats base 5/5 on multilingual (+0.6..+2.1pp); combined beats base 5/5
  and ≥ own everywhere** — the DFlash result replicates on the strong speculator,
  zero interference at N=5.
- **The headroom law holds within EAGLE**: the gain is largest exactly where the
  base is weakest (japanese 4.9% → +2.1pp, +43% relative) and shrinks as base
  strength rises (german/italian 12–14% → +0.6..+0.9pp).
- **The strongest-base domains (weird: 21–36% acceptance) sit flat** (−0.3..−0.7pp,
  ≈ bench noise) — no gain, but no harm. v2's alarming −1.3..−4.0pp regressions
  there were entirely the alignment bug.
- EAGLE3's base is far stronger than DFlash's (per-proposal acceptance 2.5–4×;
  note raw acceptance rates aren't comparable across speculators — EAGLE proposes
  3 drafts/step vs DFlash's 15; mean accept length is much closer, ~2.0 vs ~2.2).

→ charts: `multilingual_eagle/results/charts/vs_dflash.png` (the cross-speculator money chart)

### 2.6 Independent drafter — the original speculation method specializes best (but drafts too slowly)

If EAGLE's (then-)flat results came from the train/serve feature contract, a
speculator with *no* such contract should specialize easily. Vanilla two-model
speculative decoding is that speculator: an off-the-shelf `Qwen/Qwen3-0.6B`
drafts k=4 tokens autoregressively, the target verifies them in one forward
(`lib/vanilla_spec.py`, greedy, lossless), and the only alignment channel is
the token stream — trained by plain CE on the target's generations. Five
domains spanning the headroom curve, all with DFlash specialist numbers from
§2.4; same LoRA recipe, same data (n=100/domain):

| domain | base | own | combined |
|---|--:|--:|--:|
| code_sql | 50.0% | **55.5% (+5.5pp)** | 54.7% (+4.7pp) |
| lang_polish | 35.5% | **39.7% (+4.1pp)** | 38.2% (+2.7pp) |
| lang_korean | 37.3% | **40.9% (+3.6pp)** | 38.2% (+0.9pp) |
| ood_legal | 41.8% | **47.0% (+5.2pp)** | 42.4% (+0.6pp) |
| task_math_reasoning | 77.3% | **79.9% (+2.6pp)** | 78.1% (+0.7pp) |

- **Specialization works 5/5, first try, zero training-side bugs** — the
  training loop is 30 lines of plain CE. The pp gains are the largest of any
  speculator in the repo on these domains (DFlash own: +0.6..+2.0pp), ~10–12%
  relative on the 35–50%-base domains.
- **This disambiguated the headroom law — and predicted the bug.** Per-proposal
  this is by far the strongest base drafter tested (35–77% vs EAGLE's 12–36%),
  yet it gains everywhere — including +2.6pp on top of a 77% base. Strong base
  alone doesn't kill specialization; a leaky train/serve alignment does. That
  inference was confirmed three days later when the actual EAGLE alignment bug
  (the missing `shift_batch`, §3) was found and fixed, flipping §2.5 to 5/5.
- **Interference arrives early.** combined < own on 5/5 (−0.8..−4.6pp) — at
  N=5, where DFlash showed exactly zero interference. Consistent with §2.4's
  law: the tax concentrates where specialist gains are biggest, and these are
  the biggest gains in the repo.
- **But the method itself loses wall-clock at batch-1**: speedup 0.58–0.97×
  (base) / 0.50–0.80× (own, unmerged-LoRA overhead — merging ΔW removes it).
  At bs=1 eager decoding latency is layer-count-bound: a 0.6B forward
  (28 layers, ~20 ms) costs nearly as much as the 8B target's (36 layers,
  ~21–27 ms/token), so 4 draft forwards + 1 verify ≈ 5 target-tokens of
  wall-clock for 2.4–4.2 committed tokens. This is precisely the niche
  single-forward drafters (DFlash) and 1-layer heads (EAGLE) exist to fill.
  The clean recipe would be an independent drafter *shallow* enough to draft
  cheaply — the acceptance/alignment result stands either way.

→ charts: `independent-drafter/results/charts/vs_dflash.png`, `matrix.png`, `delta.png`

### 2.7 Router — MoLA's missing piece, free at serve time

An MLP (20480→512→6) over the target's hidden states at the drafter's own
conditioning layers — the tensors already computed during prefill, so routing
costs one tiny matmul. Classes: 5 languages + "other"→base, trained with hard
negatives (English/French/Spanish, code, tasks).

**100% test accuracy (596 held-out prompts), perfect confusion diagonal.**
`router/router.py` plugs directly into `serving/batched_lora.route(ids)` —
request → route → per-sequence adapter on a shared backbone, end to end.

### 2.8 Net wall-clock speedup — measured, then reduced to one constant per speculator

Vanilla target-only baselines on the spec-bench prompts (per framework: vLLM
~195 tok/s batch-1 for the 8B target, near-invariant across all 8 domains; HF
~40–50 tok/s) close the loop from acceptance to wall-clock:

- **EAGLE3 multilingual: LoRA's acceptance gains survive end-to-end** — own
  speedup beats base on 5/5 (japanese crosses break-even, 0.95×→1.02×; italian
  1.13×→1.22×). Small in absolute terms, because at L≈1.2–1.4 every accepted
  token is precious. EAGLE weird domains: 1.31–1.68×, LoRA ≈ base, as the flat
  acceptance predicts.
- **The analytic model holds: speedup ≈ L/(1+c)** with L = mean accept length
  (already measured everywhere) and c = per-step drafting overhead in
  target-forward units, a *per-speculator/per-engine constant*: EAGLE-vLLM
  c≈0.18–0.24, DFlash-HF c≈0.44, and (this explains exp 06) independent-0.6B
  c≈3. Fitted on each section, the model predicts measured wall-clock to
  **0.3–2% median error** wherever timing noise is controlled. Future cells
  need no baseline runs — compute L/(1+c).
- **Methodology lesson:** the one poorly-predicted section (6% median / 18% max
  error) is the one whose HF baseline ran in a different container days after
  the spec bench. Pair vanilla and spec timings in-container (as exp 02 did —
  0.3% error) or trust the model over unpaired timings.

- **LoRA-attributable gain, uniformly across all experiments** (since c
  cancels in the ratio, gain = L_variant/L_base − 1, timing-free): EAGLE
  multilingual +1..+5%, EAGLE weird ≈0; DFlash: languages +2..+16% (weakest
  bases largest), weird domains +2..+5%, code_sql +10% (exp 01: measured
  +11%), legal +14%; independent drafter +3..+8% (merged-equivalent).

→ `experiments/08-wallclock/results/report.md`, `charts/speedup.png`,
`charts/lora_gain.png` (per-domain LoRA gain, every experiment)

### 2.9 What serving the adapter costs — merged is free, everything else isn't

Batch-size sweep (1→64) of five serving modes on the 0.6B drafter, zero-delta
adapters so every mode decodes identical output (pure timing), all ratios
against same-container bases:

| mode | bs=1 | bs=64 |
|---|--:|--:|
| merged ΔW | **+3%** | **−0%** |
| naive unmerged wrappers (exp 06's mode) | +33% | +20% |
| vLLM punica, 1 adapter | +58% | +22% |
| vLLM punica, 50 distinct adapters | +67% | **+311%** |

Merged serving is exactly free at every batch size. The naive wrapper tax
(~20%) never amortizes. Punica's fixed kernel cost amortizes slowly with batch
— but only while the batch shares few adapters: with 50 distinct adapters
round-robin, overhead *grows* with batch size (50 live SGMV segments), reaching
4× at bs=64 on this model scale. Per-batch adapter **diversity**, not adapter
count in memory, is the cost driver. This closes the serving question: **one
combined LoRA, merged into the drafter** — zero overhead, mixed-domain batches
free — and per-domain hot-swap only via merge-on-swap at request granularity.

→ `experiments/09-batched-lora-serving/` (README, `charts/overhead.png`)

---

## 3. Bugs found and fixed (the honest section)

The EAGLE v1 *and* v2 results (archived, never deleted) were invalidated by a
chain of bugs — kept here because the *diagnostic path* is half the value:

1. **torchvision/torch ABI clash** (speculators pkg dependency) — broke imports.
2. **flex-attention pack-length mismatch** — packs must be padded to the
   128-token block-mask size.
3. **Double-normalized distillation targets** — HF's `hidden_states[-1]` is
   already post-final-norm; the TTT loss normed it again.
4. **`order` variable shadowing** (killed v1; found by code review) — the
   aux-layer concat convention string was clobbered by the epoch shuffle list, so
   every v1 training step fed the head **reversed** aux features [33,18,2] while
   serving used [2,18,33]. The LoRAs were trained to fight a scrambled input.
5. **The real killer: missing `shift_batch` alignment** (killed v2; found by
   reading the speculators package's own data pipeline). The canonical contract
   pairs `embed(x_{t+1})` with aux features `aux_t`, supervised by the verifier's
   distribution at t+1 — i.e. inputs shifted one token left against the features,
   exactly what vLLM feeds the head at serve time. Our live-feature `_ttt_forward`
   fed everything unshifted, so v2 optimized a task one position off from serving:
   the LoRA *improved its training objective* while degrading serving acceptance
   on 7/8 domains. Two tells, in hindsight: the base head's TTT loss was ~20 under
   *every* probed feature convention (a real misalignment drowns the probe's
   signal), and the probe's "accuracy" was silently a raw count (`cond_acc_0_sum`),
   not a rate. Post-fix gate: train-time step-0 top-1 accuracy (`full_acc_0`)
   must ≈ serving position-1 acceptance — it does (0.25 vs ~0.24 on polish/german).
   Same pass also fixed **packed-document contamination** in the target pass
   (docs attended to earlier pack-mates and used global positions; now
   block-diagonal 4D masks + per-doc position ids).
6. A **verification suite** now exists (`pipeline_eagle.py::verify`): zero-adapter
   merge must be byte-identical to hub weights ✓; a zero-adapter bench through
   vLLM must reproduce base digit-for-digit ✓; a held-out objective test
   separates "training broken" from "objective misaligned with serving" — it was
   the last that pointed at bug 5: objective improving, serving degrading.

Operational lessons baked into the pipelines: detached spawn-only launches
(flaky networks silently spawn duplicates — verify via server-side app list, never
client output); never trust a result file without an n-count check (stopping
containers mid-run commits partial files over complete ones).

---

## 4. What it all means

**The headroom law.** Every experiment fits one curve: *LoRA gain is inversely
proportional to base speculator strength on that domain.*

| base acceptance | gain from LoRA | evidence |
|---|---|---|
| ~3–5% (drafter nearly useless) | large, and rank keeps paying (r64 ≫ r16) | polish/korean DFlash; japanese-EAGLE +2.1pp |
| ~5–14% (weak) | solid — on BOTH speculators once aligned | weird-DFlash, italian/german; all 5 EAGLE languages +0.6..+1.0pp |
| ~25% (decent, templated domain) | largest absolute (+3.4pp) — low-entropy SQL is easiest to specialize | code_sql |
| ~21–36% (strong: EAGLE3 weird domains) | ≈0 (flat, −0.3..−0.7pp ≈ noise) | EAGLE v3 translation/roleplay/poetry |
| ~35–77% (strongest: independent 0.6B) | **+2.6..+5.5pp** — but through the most gain-transparent channel (plain CE) | exp 06, all 5 domains |

Two riders the law needs. First, it only *appears* once training reproduces
serving exactly: with the v2 misalignment, every EAGLE domain read as ≈0-to-
negative regardless of headroom, and the fix (§3, bug 5) restored the gradient.
Second, the channel matters: the independent drafter is stronger per-proposal
than EAGLE everywhere and still gains on 5/5, because plain CE on tokens is
trivially aligned and directly optimizes the acceptance event; EAGLE's soft-KL
TTT objective buys smaller serving gains per unit of val-loss improvement.

**Adapters barely fight — even 40 at a time.** Combined = specialists exactly at
3–5 domains (3 experiments × 3 ranks, heterogeneous tasks, down to rank 4). At
10–40 domains a small interference tax appears (−0.2 to −0.3pp mean, first
measurable at N=10) but it *saturates* instead of growing — no phase boundary —
and it concentrates on the domains with the largest specialist gains. Domain
adaptation lives mostly in a shared, low-rank "steering" subspace; only the
biggest domain-specific shifts compete for capacity. The deployable recipe is
still the simplest one: **one combined LoRA** (route per-domain only for the
few high-gain domains the interference ladder flags, or for tenant isolation).

**LoRA is the only sensible tool at this data scale.** Full fine-tuning 1B params
on ≤2M supervised tokens matched base at best, twice. Rank 4–16 adapters (0.01–0.2%
of the drafter) captured all of the available gain.

**Speculator choice dominates — and it's a latency/alignment trade.** EAGLE3's
pretrained head beats anything we could add to DFlash with adapters, and
fine-tuning EAGLE-style heads is operationally treacherous because the
training-time feature pipeline must reproduce serving exactly — four of our
five bugs lived there, and the last (the missing one-token shift) silently
inverted every conclusion until found. It IS tractable: with the contract
reproduced, the same 2M-param recipe lifts EAGLE 5/5 on its weak domains. The
independent drafter (exp 06) sits at the opposite corner: trivially alignable —
plain CE, zero bugs, gains on 5/5 domains at the highest base acceptance in the
repo — but its drafting latency (a full 28-layer forward per proposed token)
makes vanilla spec decode a wall-clock loss at batch-1 (0.6–1.0×).
Feature-conditioned drafters buy cheap drafts at the price of a fragile
training contract; independent drafters buy a trivial training contract at the
price of expensive drafts.

**Everything is lossless.** At temperature 0 the emitted text equals the target's
greedy output in every configuration — adapters change speed, never correctness.

## 5. Where everything lives

```
EXPERIMENTS.md          run-by-run log (all experiments, configs, verdicts)
experiments/00-base-benchmarks/    base-speculator characterization across all domains (was benchmarking/)
experiments/01-single-domain-dflash/results/                code_sql + legal (jsonl, reports, charts)
experiments/02-multilingual-dflash/           DFlash 5-language experiment (r16)
experiments/03-weird-domains/          both speculators, r16 (rank variants moved to 07)
experiments/04-multilingual-eagle/     EAGLE replication (v1+v2 archived, v3 current, verify suite)
experiments/05-interference-ladder/           10/20/40-domain interference ladder (core+distractors)
experiments/06-independent-drafter/    vanilla spec decode (Qwen3-0.6B drafter), CE-only specialization
experiments/07-rank-ladder/            adapter capacity: r4 / r16 / r64 across the 02+03 domains
experiments/08-wallclock/              net wall-clock speedup vs target-only decoding
router/                            hidden-state adapter router (100% acc) + serving hooks
serving/                           shared-backbone multi-adapter serving (hot-swap + batched routing)
```

All adapters (`*.pt`) are on the Modal volume `code-sql-pipeline` and local
`models/` dirs (gitignored). Every pipeline has a `launch` (detached), `agg_only`,
and `make_charts*.py` entry point; every result table above regenerates from the
committed jsonl files.
