"""
Route a real WildChat user prompt into one of the DataGen domains.

Two classifiers:
  - heuristic (default): free, deterministic. Uses WildChat's own detected
    `language` label for the lang_* domains, and keyword / code-fence rules for
    coding / task / ood domains. High precision, limited recall — clearly-matching
    prompts land in their domain, everything else in the language fallback.
  - claude (optional): batch-classify prompts against the full 51-domain taxonomy
    for balanced coverage. The prompts stay real human text — only the *label*
    comes from Claude, so this does NOT reintroduce synthetic-text bias.

Domain keys and the language taxonomy are imported from ../DataGen/domains so the
buckets line up 1:1 with the synthetic DataGen sets.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_DATAGEN = Path(__file__).resolve().parent.parent / "DataGen"
sys.path.insert(0, str(_DATAGEN))
from domains import REGISTRY  # noqa: E402

# name -> lang_ key, e.g. {"english": "lang_english", ...}
LANG_NAME_TO_KEY = {
    spec["description"].split(": ", 1)[1].strip().lower(): key
    for key, spec in REGISTRY.items()
    if spec["group"] == "languages" and ": " in spec["description"]
}

VALID_DOMAINS = set(REGISTRY)


# --------------------------------------------------------------------------- #
# Heuristic classifier                                                          #
# --------------------------------------------------------------------------- #
# Fenced code-block language token -> code_ domain key.
_FENCE_LANG = {
    "python": "code_python", "py": "code_python",
    "javascript": "code_javascript", "js": "code_javascript", "jsx": "code_javascript",
    "typescript": "code_typescript", "ts": "code_typescript", "tsx": "code_typescript",
    "sql": "code_sql", "postgresql": "code_sql", "mysql": "code_sql",
    "cpp": "code_cpp", "c++": "code_cpp",
    "c": "code_c",
    "ruby": "code_ruby", "rb": "code_ruby",
    "rust": "code_rust", "rs": "code_rust",
    "go": "code_go", "golang": "code_go",
    "bash": "code_bash", "sh": "code_bash", "shell": "code_bash", "zsh": "code_bash",
    "java": "code_java",
    "kotlin": "code_kotlin", "kt": "code_kotlin",
    "swift": "code_swift",
    "haskell": "code_haskell", "hs": "code_haskell",
    "r": "code_r",
}
_FENCE_RE = re.compile(r"```([a-zA-Z0-9+#]+)")

# Strong per-language keyword signals (checked only if no fenced language found).
_CODE_SIGNALS = [
    ("code_sql", re.compile(r"\bselect\b.+\bfrom\b", re.I | re.S)),
    ("code_python", re.compile(r"\bdef\s+\w+\s*\(|\bimport\s+\w+|print\(")),
    ("code_java", re.compile(r"public\s+static\s+void\s+main|System\.out\.print")),
    ("code_cpp", re.compile(r"#include\s*<[^>]+>.*std::|cout\s*<<", re.S)),
    ("code_c", re.compile(r"#include\s*<[^>]+>|printf\s*\(")),
    ("code_go", re.compile(r"\bpackage\s+main\b|\bfunc\s+\w+\(")),
    ("code_rust", re.compile(r"\bfn\s+\w+\(|println!|let\s+mut\b")),
    ("code_typescript", re.compile(r":\s*(string|number|boolean)\b|\binterface\s+\w+")),
    ("code_javascript", re.compile(r"console\.log\(|=>|\bconst\s+\w+\s*=")),
    ("code_ruby", re.compile(r"\bputs\b|\bdef\b.+\bend\b", re.S)),
    ("code_r", re.compile(r"<-\s|\bggplot\(|\bdata\.frame\(")),
    ("code_haskell", re.compile(r"::\s*\w+\s*->|\bmodule\s+\w+\s+where")),
    ("code_kotlin", re.compile(r"\bfun\s+\w+\(.*\)\s*:|\bval\s+\w+\s*=")),
    ("code_swift", re.compile(r"\bfunc\s+\w+\(.*\)\s*->|\bvar\s+\w+\s*:\s*\w")),
]

# Ordered (domain, keyword-regex) rules. First match wins. Technical / specific
# domains come before broad ones so they aren't swallowed by a generic bucket.
_RULES = [
    # --- specialised / OOD (technical) --- #
    ("ood_regex", r"\bregex\b|\bregular expression\b|pattern that matches|\\d\+|\\w\+"),
    ("ood_shell", r"\b(grep|awk|sed|kubectl|docker|systemctl)\b|command[- ]line|one[- ]liner|\bin (bash|the terminal|shell)\b"),
    ("ood_chemistry", r"\bstoichiometr|balance (the )?(chemical )?(equation|reaction)|\bmole(s|cular|cule)\b|periodic table|\b[A-Z][a-z]?\d*(\s?[+-])?\s*(->|→)"),
    ("ood_ascii_art", r"\bascii art\b|in ascii|text[- ]based (diagram|drawing)"),
    ("ood_poetry", r"\bsonnet\b|\bhaiku\b|\bvillanelle\b|\blimerick\b|rhyme scheme|iambic"),
    # --- tasks (explicit intent) --- #
    ("task_translation", r"\btranslate\b|\btranslation of\b|into (english|spanish|french|german|chinese|japanese|korean|arabic|russian|hindi)\b"),
    ("task_summarization", r"\bsummar(ize|ise|y)\b|\btl;?dr\b|in a few sentences summar"),
    ("task_json_extraction", r"\bas json\b|json format|extract .*(fields|data).*json|return .*json"),
    ("task_email_writing", r"\bwrite (an|a) e?mail\b|\bdraft (an|a) e?mail\b|\bemail to\b"),
    ("task_data_generation", r"\b(generate|create|make) .*(sample|mock|fake|synthetic|test|dummy) (data|dataset|records?)\b|\bcsv rows\b"),
    ("task_tabular_data", r"\bgiven the (table|following table)\b|\|.*\|.*\|"),
    ("task_logic_reasoning", r"\b(riddle|logic puzzle|brain[- ]?teaser)\b"),
    ("task_math_reasoning", r"\b(solve|calculate|compute)\b.*\d|\bwhat is\b.*\d+\s*[-+*/x]\s*\d+|probability that"),
    ("task_roleplay_chat", r"\byou are (a|an)\b|\bpretend (to be|you are)\b|\bact as\b|\broleplay\b"),
    ("task_creative_writing", r"\bwrite (a|an|me a) (short )?(poem|story|song|script|dialogue|tale)\b"),
    # --- specialised / OOD (topical) --- #
    ("ood_medical", r"\b(symptom|diagnos|disease|medication|dosage|prescription|blood pressure|infection|tumou?r|antibiotic)\b"),
    ("ood_financial", r"\b(invest(ing|ment)?|stock market|portfolio|interest rate|mortgage|dividend|revenue|balance sheet|cryptocurrency|401k)\b"),
    ("ood_legal", r"\b(lawsuit|liability|plaintiff|defendant|contract clause|terms of service|gdpr|copyright|jurisdiction|attorney)\b"),
    ("ood_customer_support", r"\b(refund|my order|cancel my|reset my password|warranty|support ticket|not working|billing issue|track my (order|package))\b"),
    ("task_question_answering", r"^\s*(what|why|how|who|when|where|which)\b.*\?"),
]
_RULES = [(d, re.compile(pat, re.I)) for d, pat in _RULES]


def _detect_code(text: str):
    for tok in _FENCE_RE.findall(text):
        key = _FENCE_LANG.get(tok.lower())
        if key:
            return key
    # only trust keyword signals when there's an actual code fence somewhere
    if "```" in text or re.search(r";\s*$|\{\s*$", text, re.M):
        for key, rx in _CODE_SIGNALS:
            if rx.search(text):
                return key
    return None


def classify_heuristic(prompt: str, wc_language: str | None):
    """Return a domain key or None (unroutable)."""
    ck = _detect_code(prompt)
    if ck:
        return ck
    for domain, rx in _RULES:
        if rx.search(prompt):
            return domain
    # language fallback: use WildChat's own detected language label.
    lk = LANG_NAME_TO_KEY.get((wc_language or "").strip().lower())
    return lk


# --------------------------------------------------------------------------- #
# Claude classifier (optional, batch)                                           #
# --------------------------------------------------------------------------- #
def _taxonomy_block():
    lines = [f"- {k}: {REGISTRY[k]['description']}" for k in REGISTRY]
    return "\n".join(lines)


_CLAUDE_SYSTEM = (
    "You label real user prompts by domain for a benchmark. For each prompt, "
    "return the single best-matching domain key from the taxonomy, or \"none\" "
    "if none fits well. For non-English prompts, prefer the matching lang_* "
    "domain unless the prompt is clearly about code or a specialised topic. "
    "Return only the structured JSON."
)

_LABELS_SCHEMA = {
    "type": "object",
    "properties": {"labels": {"type": "array", "items": {"type": "string"}}},
    "required": ["labels"],
    "additionalProperties": False,
}


def classify_claude_batch(client, prompts, model, max_retries=4):
    """Classify a list of prompts; returns a list of domain keys / None, aligned."""
    import json
    import time

    numbered = "\n".join(f"{i}. {p[:600]}" for i, p in enumerate(prompts))
    user = (
        "Taxonomy (domain key: description):\n"
        f"{_taxonomy_block()}\n\n"
        f"Label each of these {len(prompts)} prompts. Return a `labels` array of "
        f"exactly {len(prompts)} domain keys (or \"none\"), in order:\n\n{numbered}"
    )
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model, max_tokens=4000, system=_CLAUDE_SYSTEM,
                messages=[{"role": "user", "content": user}],
                output_config={"format": {"type": "json_schema", "schema": _LABELS_SCHEMA}},
            )
            if resp.stop_reason == "refusal":
                return [None] * len(prompts)
            text = next((b.text for b in resp.content if b.type == "text"), "")
            labels = json.loads(text).get("labels", [])
            out = []
            for i in range(len(prompts)):
                lab = labels[i] if i < len(labels) else None
                out.append(lab if lab in VALID_DOMAINS else None)
            return out
        except Exception as e:  # noqa: BLE001
            wait = min(2 ** attempt, 20)
            print(f"    ! classify batch error ({type(e).__name__}): {e} — retry in {wait}s")
            time.sleep(wait)
    return [None] * len(prompts)
