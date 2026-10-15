#!/usr/bin/env python3
"""Adversarial audit: try to game the paper without domain knowledge.

    python scripts/audit.py <file.md>

Simulates the heuristics a candidate actually uses — pick the longest option,
pick the shortest, eliminate anything containing an absolute, pick the only
hedged option — and reports how often each one wins. Exits non-zero if any
heuristic beats chance by enough to be worth using.
"""
import re
import sys
import statistics as st
from collections import Counter

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from examparse import load, has_code, words, HEDGE, ABSOL  # noqa: E402


def main(path):
    s, qbody, qs, key, expl, problems = load(path)
    txt = [(q, key[q], o) for q, _, o in qs if q in key and not has_code(o)]
    n = len(txt)
    if n == 0:
        raise SystemExit("FAIL: no text questions parsed")

    print("=" * 62)
    print(f"ADVERSARIAL AUDIT  {path.rsplit('/', 1)[-1]}   ({n} text questions)")
    print("=" * 62)

    ul = us = 0
    cw, dw, spread, wide = [], [], [], []
    for q, ck, o in txt:
        w = {l: words(t) for l, t in o.items()}
        mx, mn = max(w.values()), min(w.values())
        if w[ck] == mx and list(w.values()).count(mx) == 1:
            ul += 1
        if w[ck] == mn and list(w.values()).count(mn) == 1:
            us += 1
        cw.append(w[ck])
        dw.append(st.mean(w[l] for l in o if l != ck))
        spread.append(mx - mn)
        if mx - mn > 5:
            wide.append(q)

    hed = [q for q, ck, o in txt
           if re.search(HEDGE, o[ck].lower())
           and not any(re.search(HEDGE, o[l].lower()) for l in o if l != ck)]
    iso = [q for q, ck, o in txt
           if [l for l, t in o.items() if not re.search(ABSOL, t.lower())] == [ck]]

    gap = st.mean(cw) - st.mean(dw)
    print(f"\nHEURISTIC SUCCESS RATES   (chance = 25%)")
    print(f"  pick the longest option    {ul:>3}/{n}  {ul/n*100:5.1f}%")
    print(f"  pick the shortest option   {us:>3}/{n}  {us/n*100:5.1f}%")
    print(f"  only-hedged option         {len(hed):>3}/{n}  {len(hed)/n*100:5.1f}%")
    print(f"  eliminate-absolutes        {len(iso):>3}/{n}  {len(iso)/n*100:5.1f}%")

    print(f"\nLENGTH")
    print(f"  mean correct               {st.mean(cw):5.1f} words")
    print(f"  mean distractor            {st.mean(dw):5.1f} words")
    print(f"  gap                        {gap:+5.2f}        target within +/-1.0")
    print(f"  mean within-question spread{st.mean(spread):5.1f}        target <= 3")
    print(f"  spread > 5 words           {len(wide):>3}        target <= 3  {wide}")

    letters = [key[q] for q, _, _ in qs if q in key]
    c = Counter(letters)
    import itertools
    runs = max((len(list(g)) for g in itertools.groupby(letters)), default=0)
    print(f"\nLETTERS")
    print(f"  distribution               {dict(sorted(c.items()))}")
    print(f"  longest identical run      {runs}        target <= 3")

    fails = []
    if not 20 <= ul / n * 100 <= 30:
        fails.append(f"'pick longest' scores {ul/n*100:.0f}%, outside 20-30%")
    if not 20 <= us / n * 100 <= 30:
        fails.append(f"'pick shortest' scores {us/n*100:.0f}%, outside 20-30%")
    if hed:
        fails.append(f"correct is the only hedged option on {hed}")
    if iso:
        fails.append(f"eliminating absolutes isolates the answer on {iso}")
    if abs(gap) > 1.0:
        fails.append(f"mean word gap {gap:+.2f} exceeds +/-1.0")
    if st.mean(spread) > 3:
        fails.append(f"mean spread {st.mean(spread):.1f} exceeds 3")
    if len(wide) > 3:
        fails.append(f"{len(wide)} questions with spread > 5 words (max 3)")
    if runs >= 4:
        fails.append(f"{runs} identical answer letters in a row")
    if len(letters) >= 40 and c and max(c.values()) - min(c.values()) > 4:
        fails.append(f"letter distribution uneven: {dict(sorted(c.items()))}")

    print("\n" + "-" * 62)
    if fails:
        for f in fails:
            print(f"  FAIL  {f}")
        print(f"\nRESULT: GAMEABLE ({len(fails)} issue"
              f"{'s' if len(fails) != 1 else ''})")
        return 1
    print("RESULT: NOT GAMEABLE — every heuristic sits at chance")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: audit.py <file.md>")
    sys.exit(main(sys.argv[1]))
