"""Extract numeric values and MCQ letters from generated text; never raise, return None on failure."""

import re

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_PAREN_LETTER = re.compile(r"\(([A-Za-z])\)")
_ANSWER_LETTER = re.compile(r"answer\s*(?:is|:)?\s*\(?([A-Za-z])\)?", re.I)
_DOTTED_LETTER = re.compile(r"\b([A-Za-z])\.")
_LONE_LETTER = re.compile(r"(?:^|\n|\s)([A-Za-z])(?:\b|$)")
_NUM_SANITY = 1e6   # materials properties don't legitimately exceed this

# Leak signatures (base-LM priors) scrubbed before extraction. _URL_RE also
# matches single-slash `http(s)/...` typos so truncated-markdown dates aren't
# parsed as numbers; _LEADING_BANG_RE strips `!` preamble off `!A) Yes`.
_MD_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_URL_RE = re.compile(r"https?[:/]+\S+")
_LEADING_BANG_RE = re.compile(r"^!+\s*")
_NULL_LITERAL_RE = re.compile(r"^\s*(nan|null|none|n/a)\s*(eV|GPa|Å|[a-zA-Z/%]*)?\s*$", re.I)


def _scrub(text):
    # Order matters: full markdown image, then URL fragment, then leading `!`.
    text = _MD_IMG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _LEADING_BANG_RE.sub("", text)
    return text


def detect_leak(text):
    """True iff the output contains a markdown image embed or http(s) URL (wrong-format fallback)."""
    if not text:
        return False
    if _MD_IMG_RE.search(text):
        return True
    if _URL_RE.search(text):
        return True
    return False


def extract_number(text):
    """First signed float in text (after scrub); None on null literals or out-of-range values."""
    if text is None:
        return None
    if _NULL_LITERAL_RE.match(text):
        return None
    text = _scrub(text)
    m = _NUM_RE.search(text)
    if not m:
        return None
    v = float(m.group(0))
    if not (v == v) or v == float("inf") or v == float("-inf") or abs(v) > _NUM_SANITY:
        return None
    return v


# Truncate at the first hallucinated follow-up turn before extracting; reading
# the last line of the untruncated output picks up runaway garbage.
_GSM_BOUNDARY_RE = re.compile(r"\n\s*(?:Question|Human|Problem|Q[:.])", re.I)
_GSM_ANSWER_TAIL_RE = re.compile(
    r"answer\s*(?:is|:)\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)", re.I)
_GSM_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_gsm8k_answer(text):
    """Final numeric answer from a GSM8K chain-of-thought generation."""
    if not text:
        return None
    seg = _GSM_BOUNDARY_RE.split(text, 1)[0]
    seg = _scrub(seg)
    m = list(_GSM_ANSWER_TAIL_RE.finditer(seg))
    if m:
        s = m[-1].group(1)
    else:
        nums = _GSM_NUM_RE.findall(seg)
        if not nums:
            return None
        s = nums[-1]
    try:
        v = float(s.replace(",", ""))
    except ValueError:
        return None
    if not (v == v) or abs(v) > _NUM_SANITY:
        return None
    return v


def extract_choice(text, choices=("A", "B", "C", "D")):
    if not text:
        return None
    text = _scrub(text)
    upper_choices = {c.upper() for c in choices}
    for pat in (_PAREN_LETTER, _ANSWER_LETTER, _DOTTED_LETTER):
        m = pat.search(text)
        if m:
            c = m.group(1).upper()
            if c in upper_choices:
                return c
    last = None
    for m in _LONE_LETTER.finditer(text):
        c = m.group(1).upper()
        if c in upper_choices:
            last = c
    return last


if __name__ == "__main__":
    cases_num = [
        ("the formation energy is -1.234 eV/atom", -1.234),
        ("approximately 2.5 GPa", 2.5),
        ("2.4e-3", 2.4e-3),
        ("no number here", None),
        ("", None),
        (None, None),
        # Leak cases: digit inside imgur hash / MP id must NOT be returned.
        ("![](https://i.imgur.com/6Z7Z7Z7.png)", None),
        ("![](https://i.imgur.com/6YJ6Z7K.png)", None),
        ("![](https://www.materialsproject.org/materials/101112)", None),
        # Leaked-but-genuine: number is extractable; caller drops via detect_leak.
        ("![](https://i.imgur.com/HASH.png) {\"density\": 4.6}", 4.6),
        ("![](https://www.materialsproject.org/materials/101112) {\"density\": 4.6}", 4.6),
        # Schema-corruption: leading "!" from arxiv-bucket bleed-through.
        ("!{\"k\": 1.5}", 1.5),
        ("!nan eV", None),
        ("nan eV", None),
        ("null", None),
        ("None", None),
        # Truncated single-slash markdown leak: must NOT return 2020.03.
        ("![](https/2020.03.03.045321/0000000000", None),
        ("![](https:/2020.03.03/foo", None),
        # Single-`!` prefix to a JSON answer still parses cleanly.
        ("!{\"band gap (eV)\": 3.5221}", 3.5221),
        ("!{\"band gap (eV)\": 0.0}", 0.0),
    ]
    n_pass = 0
    for text, want in cases_num:
        got = extract_number(text)
        ok = (got is None and want is None) or (got is not None and want is not None and abs(got - want) < 1e-9)
        n_pass += ok
        print(f"{'PASS' if ok else 'FAIL'} extract_number({text!r}) = {got}  want {want}")

    cases_choice = [
        ("(B)", "B"),
        ("Answer: D.", "D"),
        ("Answer is C", "C"),
        ("the answer is c", "C"),
        ("A.", "A"),
        ("...so the answer is\n\nB", "B"),
        ("AB", None),
        ("", None),
        (None, None),
        # Letters inside URLs must not bleed into the choice.
        ("![](https://i.imgur.com/6Z7Z7Z7.png)", None),
        # Genuine letter outside a leak block still matches.
        ("![](https://i.imgur.com/HASH.png)\nAnswer: B", "B"),
        # Leading `!` preamble before the real answer letter.
        ("!A) Yes", "A"),
        ("!B) No", "B"),
        ("!C) Orthorhombic", "C"),
        ("!! D) Hexagonal", "D"),  # multiple `!` should also strip
        ("! E) Trigonal", "E", ("A","B","C","D","E","F","G")),  # 7-class crystal_system
        # Bare "!" with no answer stays unparseable.
        ("!", None),
    ]
    for case in cases_choice:
        if len(case) == 3:
            text, want, choices = case
            got = extract_choice(text, choices=choices)
        else:
            text, want = case
            got = extract_choice(text)
        ok = got == want
        n_pass += ok
        print(f"{'PASS' if ok else 'FAIL'} extract_choice({text!r}) = {got}  want {want}")

    cases_leak = [
        ("![](https://i.imgur.com/6Z7Z7Z7.png)", True),
        ("![](https://www.materialsproject.org/materials/101112) {\"x\":1}", True),
        ("https://example.com/foo", True),
        ("plain old text with -1.234 eV", False),
        ("", False),
        (None, False),
        ("see https://arxiv.org/abs/1234.56789 for refs", True),  # any URL counts
        # Truncated-URL variants.
        ("![](https/2020.03.03.045321/0000000000", True),
        ("![](https:/2020.03.03/foo", True),
    ]
    for text, want in cases_leak:
        got = detect_leak(text)
        ok = got == want
        n_pass += ok
        print(f"{'PASS' if ok else 'FAIL'} detect_leak({text!r}) = {got}  want {want}")

    total = len(cases_num) + len(cases_choice) + len(cases_leak)
    print(f"\n{n_pass}/{total} tests passed")
    if n_pass < total:
        raise SystemExit(1)
