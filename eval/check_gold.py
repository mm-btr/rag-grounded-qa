"""Gold markup vs corpus: numeric key values of each expected answer are matched as digit
signatures against corpus_flat.json; a value whose carrier chunks are not covered by the
gold groups is reported as a CANDIDATE hole (heuristic — each hit is resolved by hand).
Files-only:

    python eval/check_gold.py
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
_YEAR = re.compile(r"^(19|20)\d{2}$")
_DATE = re.compile(r"\d{1,2}\.\d{1,2}\.(19|20)\d{2}")
_NUM = re.compile(r"\d[\d.,]*(?:\s\d[\d.,]*)*")


def signatures(text):
    """Digit signatures of numeric tokens ('1 768 424' -> '1768424'), 4+ digits;
    years and calendar dates excluded — period qualifiers, not values."""
    out = set()
    for m in _NUM.finditer(text or ""):
        if _DATE.search(m.group()):
            continue
        sig = re.sub(r"\D", "", m.group())
        if len(sig) >= 4 and not _YEAR.match(sig):
            out.add(sig)
    return out


def main():
    with open(os.path.join(HERE, "golden.json"), encoding="utf-8") as f:
        golden = json.load(f)
    with open(os.path.join(HERE, "corpus", "corpus_flat.json"), encoding="utf-8") as f:
        corpus = json.load(f)
    corpus_sigs = {cid: signatures(txt) for cid, txt in corpus.items()}

    questions = holes = 0
    for q in golden:
        if not q.get("answerable") or not q.get("gold_chunks"):
            continue
        questions += 1
        gold = {cid for group in q["gold_chunks"] for cid in group}
        facts = " ; ".join(q.get("expected_facts") or []) or q.get("reference_answer") or ""
        for sig in sorted(signatures(facts)):
            carriers = {cid for cid, sigs in corpus_sigs.items() if sig in sigs}
            if carriers and not (carriers & gold):
                holes += 1
                print(f"№{q['id']} [{q['type']}] ДЫРА? значение '{sig}' вне голда "
                      f"(gold: {sorted(gold)})")
                for cid in sorted(carriers)[:4]:
                    text = re.sub(r"\s+", " ", corpus[cid])
                    pos = max(re.sub(r"\D", " ", text).find(sig[:4]), 0)
                    print(f"    + {cid}: …{text[max(0, pos - 60):pos + 80]}…")
    print(f"\nпроверено вопросов: {questions}; кандидатов-дыр: {holes} "
          f"(каждый решается руками: промах разметки или корпусный дубликат вне контекста вопроса)")


if __name__ == "__main__":
    main()
