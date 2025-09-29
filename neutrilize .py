# Very simple lexicon-based substitutions; extend per-language
NEUTRAL_DICT = {
    # profanity
    "bloody": "very",
    "stupid": "unwise",
    "damn": "darn",
    # hate / threats examples
    "kill": "harm",
    "destroy": "defeat",
}

REPLACERS = [(re.compile(r"\b"+re.escape(k)+r"\b", flags=re.IGNORECASE), v) for k,v in NEUTRAL_DICT.items()]

def neutralize_text(txt: str) -> Tuple[str, List[Tuple[int,int]]]:
    """Return (neutralized_text, replaced_spans)
    replaced_spans = list of (start,end) of neutralized ranges after substitution.
    """
    spans = []
    out = txt
    for pat, repl in REPLACERS:
        for m in pat.finditer(out):
            spans.append((m.start(), m.end()))
        out = pat.sub(repl, out)
    return out, spans

def load_gold_spans(path: str) -> List[Dict]:
    return [json.loads(l) for l in open(path, 'r', encoding='utf-8')]
