#!/usr/bin/env python3
"""Check reading-load quotas.

    python scripts/load_check.py <file.md> [--block]

Without --block, checks against the full 90-question paper quotas.
With --block, checks a single block's proportionate share.
Exits non-zero if a quota is missed.
"""
import re
import sys
import statistics as st

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from examparse import read, split_sections, parse_questions  # noqa: E402

FULL = {
    "code questions": 35,
    "code lines": 165,
    "table questions": 17,
    "mean stem words": 47,
    "median stem words": 38,
    "stems over 80w": 17,
    "stems over 120w": 7,
    "total words": 11000,
}

# Per-topic code-question targets, used in --block mode (see references/subtopics.md).
# Topics not listed default to 5. Code lines scale from the paper-wide ratio.
BLOCK_CODE_Q = {1: 0, 2: 2, 3: 0, 6: 6, 7: 0, 11: 2}
DEFAULT_BLOCK_CODE_Q = 5
LINES_PER_CODE_Q = FULL["code lines"] / FULL["code questions"]


def block_topic(path):
    m = re.match(r"block(\d+)", path.rsplit("/", 1)[-1])
    return int(m.group(1)) if m else None


def main(path, block=False):
    s = read(path)
    qbody, _, _ = split_sections(s)
    qs, _ = parse_questions(qbody)
    if not qs:
        raise SystemExit("FAIL: no questions parsed")

    stems, codeq, lines, tables = [], 0, 0, 0
    for _, stem, _ in qs:
        stems.append(len(stem.split()))
        cl = re.findall(r"```.*?```", stem, flags=re.S)
        if cl:
            codeq += 1
            lines += sum(c.count("\n") - 1 for c in cl)
        if re.search(r"^\|.*\|\s*$", stem, flags=re.M):
            tables += 1

    n = len(qs)
    scale = (n / 90) if block else 1.0
    topic = block_topic(path) if block else None
    got = {
        "code questions": codeq,
        "code lines": lines,
        "table questions": tables,
        "mean stem words": st.mean(stems),
        "median stem words": st.median(stems),
        "stems over 80w": sum(x > 80 for x in stems),
        "stems over 120w": sum(x > 120 for x in stems),
        "total words": len(qbody.split()),
    }

    print("=" * 62)
    print(f"READING LOAD  {path.rsplit('/', 1)[-1]}   ({n} questions"
          f"{', block share' if block else ''})")
    print("=" * 62)
    print(f"{'metric':<22}{'actual':>10}{'required':>11}   status")
    print("-" * 62)

    fails = []
    for k, minimum in FULL.items():
        # averages are not scaled by block size; counts are
        req = minimum if k.startswith(("mean", "median")) else minimum * scale
        if block and topic is not None and k in ("code questions", "code lines"):
            cq = BLOCK_CODE_Q.get(topic, DEFAULT_BLOCK_CODE_Q)
            req = cq if k == "code questions" else cq * LINES_PER_CODE_Q
        a = got[k]
        ok = a >= req
        if not ok:
            fails.append(f"{k}: {a:.0f} < {req:.0f}")
        print(f"{k:<22}{a:>10.1f}{req:>11.1f}   {'ok' if ok else 'SHORT'}")

    print("-" * 62)
    if fails:
        for f in fails:
            print(f"  FAIL  {f}")
        print("\nRESULT: FAILED — lengthen stems or add code, do not add questions")
        return 1
    print("RESULT: PASSED")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        raise SystemExit("usage: load_check.py <file.md> [--block]")
    sys.exit(main(args[0], "--block" in sys.argv))
