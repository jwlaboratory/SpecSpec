# Results — DFlash vs EAGLE3 across domains

Two 1B speculators for the same 8B target `Qwen/Qwen3-8B`, benchmarked per domain on
the held-out **test** split, across three prompt sources:

- **synthetic** — Claude-generated prompts (51 domains)
- **wild** — real WildChat prompts, sorted into domains (45 domains)
- **downloaded** — real purpose-built HF datasets for specialised domains (medical, financial, legal)

| Speculator | proposes/step (k) | model |
|---|---|---|
| DFlash | 15 (block of 16) | `z-lab/Qwen3-8B-DFlash-b16` |
| EAGLE3 | 3 | `RedHatAI/Qwen3-8B-speculator.eagle3` |

Charts: `charts/alldomains.png` (every source/domain, both metrics), `charts/overview_*`
(source × speculator), `charts/compare_<source>_*` (per-domain, per source).

---

## Two metrics, and why they disagree

- **Acceptance rate** = accepted ÷ proposed draft tokens. The drafter's *prediction
  quality* — how often it guesses the target's next token. k-independent.
- **Mean length accepted** = tokens committed per 8B target forward pass.

They're linked exactly by **mean length ≈ 1 + acceptance × k**. So the two speculators
sit on opposite sides of every comparison:

| source | DFlash acc / len | EAGLE3 acc / len |
|---|---|---|
| synthetic | 14.8% / 3.21 | **35.6%** / 2.07 |
| wild | 14.1% / 3.11 | **31.5%** / 1.94 |
| downloaded | 12.5% / 2.87 | **37.6%** / 2.13 |

**EAGLE3 predicts far better** (2–3× the acceptance rate) but is **capped at k+1 = 4**
tokens/pass; **DFlash predicts worse but proposes 15**, so it commits more per pass
(cap 16). Neither number alone is "speed" — mean length ignores the drafter's
launch/overhead latency, and acceptance ignores the block size. Acceptance rate is the
number fine-tuning actually moves.

---

## 1. Where do they degrade? (the off-distribution question)

**The weak spot is language/script — not "specialised" topics.** Both speculators'
worst domains are **natural languages, especially non-Latin / lower-resource ones**.
On the synthetic set, the bottom-5 for *both* speculators are all languages:

- DFlash worst: Arabic 5%, Vietnamese 5%, Korean 4%, Turkish 4%, Polish 4%
- EAGLE3 worst: Vietnamese 9%, Chinese 8%, Russian 6%, Japanese 6%, Arabic 5%

Meanwhile the deliberately "out-of-distribution" **specialised domains do fine** —
`ood` (medical, legal, regex, shell, chemistry, ascii-art, poetry) averages **42.7%**
for EAGLE3, *higher* than the language group. So for these Qwen3 drafters,
"off-distribution" means **the script/language the target is speaking, not the topic**.

Per-group acceptance (synthetic):

| group | DFlash | EAGLE3 | EAGLE3/DFlash |
|---|---|---|---|
| languages (16) | 7.2% | 16.3% | 2.26× |
| coding (15) | 18.5% | 45.2% | 2.45× |
| tasks (11) | 19.6% | 44.8% | 2.29× |
| ood/specialised (9) | 16.1% | 42.7% | 2.66× |

**Does one speculator lose its edge off-distribution? No — EAGLE3 keeps (even widens)
its advantage.** The EAGLE3/DFlash ratio is ~2.3–2.7× on *every* group, and is actually
**highest on the specialised `ood` group (2.66×)**. EAGLE3 doesn't collapse relative to
DFlash when the domain gets weird; it stays proportionally ahead.

**But EAGLE3 swings more.** Its per-domain acceptance ranges 5%→63% (std 16.3) vs
DFlash's 4%→38% (std 8.2). That's a direct consequence of the higher ceiling: EAGLE3 is
excellent where the target is predictable (code, structured tasks) and craters on hard
languages, while DFlash is flatter but uniformly lower.

Zooming into "in-distribution vs off-distribution" pairs (synthetic):

| slice | DFlash | EAGLE3 |
|---|---|---|
| high-resource langs (en/es/fr/de/zh/pt) | 8.3% | 19.8% |
| low-resource langs (sw/vi/pl/tr/ko/ar) | 6.7% | 17.2% |
| mainstream code (py/js/sql/java/c/cpp) | 19.6% | 46.0% |
| niche code (haskell/kotlin/swift/r/rust/go) | 17.4% | 44.1% |

The drops from in→off-distribution are **modest and similar for both** — and *smaller*
for code than for languages. Programming language barely matters (both drafters
generalise across Python↔Haskell); human language matters a lot (English↔Arabic).

---

## 2. Further analysis

**Best domains are structured / low-entropy.** Both speculators peak on
`json_extraction`, `data_generation`, `tabular_data`, `math_reasoning`, and code — where
the target's output is templated and predictable. EAGLE3 tops out at **json_extraction
63%**. The common thread with the weak domains is **target entropy**: acceptance rate is
really measuring *how predictable the 8B target's greedy output is on that domain*, so
part of the low language scores reflect the target itself being higher-entropy on
free-form non-English text — not purely a drafter failing.

**The two drafters share a weakness profile.** The domains DFlash struggles on are the
same ones EAGLE3 struggles on (languages, esp. non-Latin). Their strengths and
weaknesses are correlated, which points at a shared cause — both were distilled on
English + code-heavy data against the same target, so they inherit the same blind spots.

**Synthetic prompts don't inflate the numbers.** Per-group and overall acceptance track
closely across synthetic / wild / downloaded (e.g. DFlash overall 14.8 / 14.1 / 12.5%;
EAGLE3 35.6 / 31.5 / 37.6%). The clean Claude-generated set is a fair stand-in for real
prompts — the "are we measuring an artefact of synthetic data" worry doesn't hold up.
If anything, real WildChat *languages* score slightly higher than synthetic (DFlash lang
8.8% wild vs 7.2% synthetic), likely because real chat prompts skew toward more common,
more predictable phrasings.

**EAGLE3 has surprising OOD hits.** Its synthetic top-5 includes `ascii_art` (59%) and
`swahili` (56%) — domains we expected to be hard. This suggests EAGLE3's hidden-state
conditioning generalises better to unusual output structure than a block-diffusion
drafter does.

---

## Implications for fine-tuning

- **The prize is non-English languages.** Both speculators are weakest there by a wide
  margin, so a domain-targeted fine-tune (e.g. a per-language or multilingual LoRA on
  the drafter) has the most headroom. Code/task/structured domains are already good.
- **Acceptance rate is the training target**, not mean length (which is mostly a
  function of the fixed block size k). Improving acceptance on the weak domains raises
  mean length automatically via `1 + acceptance × k`.
- **EAGLE3's higher acceptance is worth studying** for the DFlash improvement work — same
  target, same 1B budget, ~2.3× better prediction, so its conditioning/training recipe is
  a strong reference point.

## Caveats

- **Mean length ≠ wall-clock speedup** — it ignores drafter launch/overhead latency and
  block-size cost per verify; it's the target-passes-saved proxy, not measured tok/s.
- **Short real-data domains are noisy** — a few `wild` domains have small test sets (rare
  languages/code langs WildChat barely contains), and `downloaded` is only 3 domains.
- **Acceptance conflates drafter quality with target entropy** — low scores on free-form
  domains partly reflect a less-predictable target, not only a weaker drafter.
