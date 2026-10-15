"""Shared parsing for exam files.

Every script reads the answer key out of the file itself, so no separate
answers argument is ever needed.
"""
import re

HEDGE = r"\b(typically|usually|roughly|approximately|often|generally|may|likely|somewhat|tends? to|can)\b"
ABSOL = r"\b(always|never|only|cannot|must|every|any |no |impossible|undefined|exactly|all )\b"


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def split_sections(s):
    """Return (questions_body, key_text, explanations_text)."""
    if "**Q1.**" not in s:
        raise SystemExit("FAIL: no '**Q1.**' marker found — is this an exam file?")
    body = "**Q1.**" + s.split("**Q1.**", 1)[1]
    qbody = body.split("# ANSWER KEY")[0]
    key = ""
    expl = ""
    if "# ANSWER KEY" in s:
        key = s.split("# ANSWER KEY", 1)[1].split("# EXPLANATIONS")[0]
    if "# EXPLANATIONS" in s:
        expl = s.split("# EXPLANATIONS", 1)[1]
    return qbody, key, expl


def parse_key(key_text):
    """Answer key as {qnum: letter}. Handles 1-, 2- and 3-group table rows."""
    out = {}
    for row in re.findall(r"^\|(.+)\|\s*$", key_text, flags=re.M):
        cells = [c.strip() for c in row.split("|")]
        for i in range(len(cells) - 1):
            if cells[i].isdigit() and re.fullmatch(r"[A-D]", cells[i + 1] or ""):
                out[int(cells[i])] = cells[i + 1]
    return out


def parse_errata(s):
    """Errata override the key: {qnum: letter}."""
    return {int(q): a for q, a in
            re.findall(r"\*\*Q(\d+)\s*→\s*([A-D])\*\*", s)}


def parse_questions(qbody):
    """List of (qnum, stem, {letter: option_text}). Raises on malformed blocks."""
    out = []
    problems = []
    for blk in re.split(r"(?=^\*\*Q\d+\.\*\*)", qbody, flags=re.M):
        m = re.match(r"^\*\*Q(\d+)\.\*\*", blk)
        if not m:
            continue
        q = int(m.group(1))
        sp = [(o.group(1), o.start()) for o in
              re.finditer(r"^([A-D])\.(?: |$)", blk, flags=re.M)]
        if len(sp) != 4 or [l for l, _ in sp] != list("ABCD"):
            problems.append((q, f"found {len(sp)} option markers {[l for l, _ in sp]}"))
            continue
        ends = [sp[i + 1][1] for i in range(3)]
        dm = blk.find("\n---", sp[3][1])
        ends.append(dm if dm != -1 else len(blk))
        stem = blk[:sp[0][1]]
        opts = {l: blk[p + 2:ends[i]].strip() for i, (l, p) in enumerate(sp)}
        out.append((q, stem, opts))
    return out, problems


def parse_explanations(expl_text):
    """{qnum: letter} from explanation headers."""
    return {int(q): a for q, a in
            re.findall(r"^\*\*Q(\d+)\s*—\s*([A-D])\.\*\*", expl_text, flags=re.M)}


def has_code(opts):
    return any(v.lstrip().startswith("```") for v in opts.values())


def words(t):
    return len(t.split())


def load(path):
    """Convenience: returns (raw, questions, key, explanations, problems)."""
    s = read(path)
    qbody, key_text, expl_text = split_sections(s)
    qs, problems = parse_questions(qbody)
    key = parse_key(key_text)
    key.update(parse_errata(s))
    expl = parse_explanations(expl_text)
    return s, qbody, qs, key, expl, problems
