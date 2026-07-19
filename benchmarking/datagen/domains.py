"""
Domain registry for the DFlash drafter dataset.

Each domain is a name -> spec mapping. `generate.py` walks this registry and,
for every domain, asks Claude to produce a large, diverse set of *user prompts*
(not answers) that exercise the target/drafter on that domain. The point is
BREADTH: natural languages, programming languages, general tasks, and a batch
of deliberately out-of-distribution / specialised domains so we can see where a
tiny 1B block-diffusion drafter tracks the 8B target and where it falls apart.

Spec fields
-----------
group        : coarse bucket, used by --group and for reporting.
description  : one-line human summary.
instruction  : the domain-specific brief handed to Claude. It must describe
               *what a user prompt in this domain looks like* — Claude returns a
               JSON array of such prompts. For natural languages the brief
               requires the prompts themselves to be written natively in that
               language.
legacy_key   : if set, the matching category in the old ../prompts.py generator.
               generate.py seeds the accumulator with those deterministic
               prompts (reusing existing work) before topping up via the API.

Keys are kept identical to ../prompts.py where a category already existed, so
seeding is a straight lookup.
"""
from collections import OrderedDict

# --------------------------------------------------------------------------- #
# Natural languages. The instruction is templated so every language shares the  #
# same brief, differing only in the language name / native-script requirement.  #
# --------------------------------------------------------------------------- #
_LANG_INSTRUCTION = (
    "Generate realistic user prompts written ENTIRELY in {lang} ({script}). "
    "These are things a native {lang} speaker would actually type to an AI "
    "assistant: questions, requests to explain or summarise, asks for advice, "
    "short-story or email requests, comparisons, how-to questions, opinions, "
    "and everyday tasks. Cover many topics (science, history, health, money, "
    "culture, technology, daily life, work, relationships). Vary length and "
    "phrasing. Do NOT write any English — every prompt must be natural, "
    "idiomatic {lang}. Do NOT number the prompts or add commentary."
)

_LANGUAGES = [
    # key suffix,   display name,   script hint,          legacy key
    ("english",    "English",       "Latin script",        "lang_english"),
    ("spanish",    "Spanish",       "Latin script",        "lang_spanish"),
    ("french",     "French",        "Latin script",        "lang_french"),
    ("german",     "German",        "Latin script",        "lang_german"),
    ("portuguese", "Portuguese",    "Latin script",        "lang_portuguese"),
    ("hindi",      "Hindi",         "Devanagari script",   "lang_hindi"),
    ("chinese",    "Chinese",       "Simplified Chinese",  "lang_chinese"),
    ("japanese",   "Japanese",      "Japanese script",     "lang_japanese"),
    ("russian",    "Russian",       "Cyrillic script",     "lang_russian"),
    ("arabic",     "Arabic",        "Arabic script",       "lang_arabic"),
    # Extra / lower-resource languages to stress drafter generalisation.
    ("korean",     "Korean",        "Hangul script",       None),
    ("italian",    "Italian",       "Latin script",        None),
    ("turkish",    "Turkish",       "Latin script",        None),
    ("vietnamese", "Vietnamese",    "Latin script w/ diacritics", None),
    ("polish",     "Polish",        "Latin script",        None),
    ("swahili",    "Swahili",       "Latin script",        None),
]

# --------------------------------------------------------------------------- #
# Programming languages. Shared brief; the model tailors tasks to the language. #
# --------------------------------------------------------------------------- #
_CODE_INSTRUCTION = (
    "Generate realistic user prompts that ask an AI assistant to write, debug, "
    "explain, refactor, or reason about {lang} code. Mix difficulty: small "
    "functions, algorithms, data-structure work, idiomatic-{lang} requests, "
    "'explain this snippet', 'fix this bug', 'add tests', 'optimise this', and "
    "real-world tasks typical for {lang}. Where natural, embed a short {lang} "
    "snippet in the prompt. Write the prompts in English (the code is {lang}). "
    "Vary length and phrasing. Do NOT number the prompts or add commentary."
)

_CODE_LANGS = [
    ("python",     "Python",       "code_python"),
    ("javascript", "JavaScript",   "code_javascript"),
    ("sql",        "SQL",          "code_sql"),
    ("cpp",        "C++",          "code_cpp"),
    ("ruby",       "Ruby",         "code_ruby"),
    ("rust",       "Rust",         "code_rust"),
    ("go",         "Go",           "code_go"),
    ("bash",       "Bash",         "code_bash"),
    # Extra languages (Java explicitly requested) to widen coverage.
    ("java",       "Java",         None),
    ("typescript", "TypeScript",   None),
    ("c",          "C",            None),
    ("kotlin",     "Kotlin",       None),
    ("swift",      "Swift",        None),
    ("haskell",    "Haskell",      None),
    ("r",          "R",            None),
]

# --------------------------------------------------------------------------- #
# General task domains (English). These mirror the old task_* categories.       #
# --------------------------------------------------------------------------- #
_TASKS = [
    ("task_summarization", "Summarisation requests",
     "Generate user prompts that ask an assistant to summarise something: "
     "articles, meetings, papers, books, threads, documents. Include prompts "
     "that embed a short passage to be summarised, and prompts that name a "
     "topic to be summarised. Vary the requested length/format (bullet points, "
     "one sentence, TL;DR, for a specific audience).",
     "task_summarization"),
    ("task_question_answering", "General question answering",
     "Generate factual and explanatory questions a user would ask an assistant "
     "across science, history, geography, health, technology, everyday 'why/how' "
     "questions, and definitions. Mix simple lookups with 'explain like I'm five' "
     "and detailed-explanation requests.",
     "task_question_answering"),
    ("task_json_extraction", "Structured JSON extraction",
     "Generate prompts that ask the assistant to extract structured fields as "
     "JSON from a sentence or short passage embedded in the prompt (people, "
     "orders, events, invoices, sensor readings, bookings). Each prompt should "
     "contain the source text plus an instruction to return JSON.",
     "task_json_extraction"),
    ("task_translation", "Translation requests",
     "Generate prompts that ask the assistant to translate a given sentence or "
     "short passage from one language to another. Embed the source text and name "
     "the target language. Cover many language pairs.",
     "task_translation"),
    ("task_creative_writing", "Creative writing",
     "Generate prompts asking for creative writing: poems, short stories, "
     "flash fiction, limericks, haiku, dialogues, product descriptions, toasts, "
     "scene-setting, and playful pieces. Vary form, tone, and constraints.",
     "task_creative_writing"),
    ("task_email_writing", "Email & professional writing",
     "Generate prompts asking the assistant to draft emails and short "
     "professional messages: requests, apologies, follow-ups, invitations, "
     "announcements, negotiations, declines, thank-you notes. Vary tone and "
     "recipient.",
     "task_email_writing"),
    ("task_roleplay_chat", "Roleplay / persona chat",
     "Generate prompts that put the assistant into a role or persona and start "
     "an interaction (museum guide, tutor, dispatcher, coach, concierge, "
     "captain, mentor, etc.). Each prompt sets the scene and the user's opening.",
     "task_roleplay_chat"),
    ("task_math_reasoning", "Math word problems",
     "Generate math word problems and quantitative reasoning prompts: arithmetic, "
     "algebra, geometry, rates, percentages, probability, simple/compound "
     "interest. Ask for step-by-step solutions. Vary difficulty.",
     "task_math_reasoning"),
    ("task_logic_reasoning", "Logic puzzles",
     "Generate logic and reasoning puzzles: syllogisms, river-crossing, "
     "weighing puzzles, classic brain-teasers, deduction problems. Ask the "
     "assistant to reason step by step.",
     "task_logic_reasoning"),
    ("task_tabular_data", "Tabular reasoning",
     "Generate prompts that embed a small table (as text) and ask a question "
     "about it: totals, averages, comparisons, growth rates, best/worst, "
     "medians, filtering. Include the table inline in each prompt.",
     "task_tabular_data"),
    ("task_data_generation", "Synthetic data generation",
     "Generate prompts that ask the assistant to GENERATE synthetic/mock data: "
     "sample CSV rows, JSON records, fake user profiles, product catalogues, "
     "test fixtures, example datasets, seed data for a database, realistic-looking "
     "records for testing. Specify fields, counts, and formats in the prompts.",
     None),
]

# --------------------------------------------------------------------------- #
# Specialised / out-of-distribution domains — deliberately far from the         #
# drafter's likely training mix, to probe where it stops tracking the target.   #
# --------------------------------------------------------------------------- #
_OOD = [
    ("ood_medical", "Medical / clinical",
     "Generate prompts a user (patient, student, or clinician) might ask about "
     "medicine and health: symptoms and what they could mean, how conditions and "
     "treatments work, drug interactions, lab-value interpretation, anatomy and "
     "physiology, medical terminology, and 'explain this diagnosis'. Keep them "
     "informational (not a request for a personal diagnosis to act on). Use real "
     "clinical vocabulary."),
    ("ood_financial", "Finance / investing",
     "Generate prompts about finance, investing, accounting, and economics: "
     "explaining instruments and metrics, valuation, budgeting, taxes, interest "
     "and returns, reading financial statements, market concepts, and personal "
     "finance questions. Include some that embed small numbers/tables to reason "
     "over."),
    ("ood_legal", "Legal",
     "Generate prompts about law and legal topics: explaining legal concepts, "
     "contract clauses, rights and obligations, procedure, drafting simple "
     "clauses, summarising statutes or cases in plain language, and 'what does "
     "this legal term mean'. Keep them general/informational."),
    ("ood_customer_support", "Customer support",
     "Generate prompts in a customer-support setting: a user contacting support "
     "with a problem (billing issue, broken product, account access, refund, "
     "shipping delay, cancellation, how-to), AND prompts asking the assistant to "
     "draft support replies, troubleshooting steps, or apology/escalation "
     "messages. Cover many industries (SaaS, retail, telecom, banking, travel)."),
    ("ood_chemistry", "Chemistry / scientific notation",
     "Generate prompts about chemistry and scientific reasoning that involve "
     "formulas, equations, and notation: balancing reactions, stoichiometry, "
     "explaining mechanisms, unit conversions, interpreting chemical formulae, "
     "and problems whose answers use subscripts/superscripts or LaTeX-style math. "
     "Include some that ask for answers formatted in LaTeX."),
    ("ood_regex", "Regex & pattern matching",
     "Generate prompts asking the assistant to write, explain, debug, or test "
     "regular expressions: match emails/URLs/dates/phone numbers, extract groups, "
     "validate formats, 'what does this regex do', and 'fix this regex'. Embed "
     "the pattern or the target strings where natural."),
    ("ood_shell", "Shell one-liners / DevOps",
     "Generate prompts asking for shell/command-line one-liners and DevOps tasks: "
     "find/grep/awk/sed pipelines, git commands, docker/kubectl usage, file "
     "manipulation, process and disk inspection, cron, and 'explain this command'. "
     "Embed commands where natural."),
    ("ood_poetry", "Formal poetry",
     "Generate prompts asking for poetry under FORMAL constraints: sonnets, "
     "villanelles, sestinas, acrostics, specific rhyme schemes, syllable counts, "
     "meter, or poems where each line starts with a given letter. Vary theme and "
     "the exact constraint."),
    ("ood_ascii_art", "ASCII art & diagrams",
     "Generate prompts asking the assistant to produce ASCII art or "
     "text-based diagrams: simple pictures, banners, box diagrams, flowcharts, "
     "tables drawn with characters, and 'draw X in ASCII'. Vary the subject."),
]


def _lang_key(suffix: str) -> str:
    return f"lang_{suffix}"


def _code_key(suffix: str) -> str:
    return f"code_{suffix}"


def build_registry() -> "OrderedDict[str, dict]":
    """Return the full ordered domain registry."""
    reg: "OrderedDict[str, dict]" = OrderedDict()

    for suffix, name, script, legacy in _LANGUAGES:
        reg[_lang_key(suffix)] = {
            "group": "languages",
            "description": f"Natural language: {name}",
            "instruction": _LANG_INSTRUCTION.format(lang=name, script=script),
            "legacy_key": legacy,
        }

    for suffix, name, legacy in _CODE_LANGS:
        reg[_code_key(suffix)] = {
            "group": "coding",
            "description": f"Programming language: {name}",
            "instruction": _CODE_INSTRUCTION.format(lang=name),
            "legacy_key": legacy,
        }

    for key, desc, instruction, legacy in _TASKS:
        reg[key] = {
            "group": "tasks",
            "description": desc,
            "instruction": instruction,
            "legacy_key": legacy,
        }

    for key, desc, instruction in _OOD:
        reg[key] = {
            "group": "ood",
            "description": desc,
            "instruction": instruction,
            "legacy_key": None,
        }

    return reg


REGISTRY = build_registry()
GROUPS = ["languages", "coding", "tasks", "ood"]


if __name__ == "__main__":
    from collections import Counter
    counts = Counter(spec["group"] for spec in REGISTRY.values())
    print(f"{len(REGISTRY)} domains across {len(GROUPS)} groups: {dict(counts)}\n")
    for key, spec in REGISTRY.items():
        seed = f"  (seeds from ../prompts.py:{spec['legacy_key']})" if spec["legacy_key"] else ""
        print(f"  {key:28s} [{spec['group']:9s}] {spec['description']}{seed}")
