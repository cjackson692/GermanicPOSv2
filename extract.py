import glob, json, os, re, sys
from collections import Counter
import numpy as np
import pandas as pd
import yaml
from tag_maps import PENN_BASE, PENN_PUNCT, GOTHIC, OHG_POS

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yaml")))
MASKED = set(CFG["masked_tags"])
_TOKEN = re.compile(r"\(|\)|[^\s()]+")
_TRACE = re.compile(r"^\*|^0$|^\{|<.*>")

def iter_leaf_lists(text):
    toks = _TOKEN.findall(text)
    depth, leaves, i, n = 0, [], 0, len(toks)
    while i < n:
        t = toks[i]
        if t == "(":
            if i + 3 < n and toks[i+1] not in "()" and toks[i+2] not in "()" and toks[i+3] == ")":
                leaves.append((toks[i+1], toks[i+2])); i += 4; continue
            depth += 1
        elif t == ")":
            depth -= 1
            if depth == 0 and leaves:
                yield leaves; leaves = []
        i += 1
    if leaves:
        yield leaves

def clean_leaves(leaves, form_lemma=False):
    words, tags = [], []
    for tag, word in leaves:
        if tag in ("CODE", "ID", "LB", "REF") or _TRACE.match(word) or "<" in word or ">" in word:
            continue
        if form_lemma and "-" in word:
            word = word.rsplit("-", 1)[0]
        if word:
            words.append(word); tags.append(tag)
    return words, tags

def norm_penn(tag):
    t = tag.split("^")[0].split("-")[0]
    t = t.split("+")[-1] if "+" in t else t
    return re.sub(r"[0-9]+$", "", t)

def penn_final(rt):
    b = norm_penn(rt)
    return "PUNCT" if (b in PENN_PUNCT or rt in PENN_PUNCT) else PENN_BASE.get(b, "XX")

def finalize(name, sents, tags, extra=None):
    out_s, out_t = [], []
    for ws, ts in zip(sents, tags):
        assert len(ws) == len(ts)
        if not ws or all(t in MASKED for t in ts):
            continue
        ws = ["start"] + [str(w).lower() for w in ws] + ["stop"]
        ts = ["start"] + [str(t) for t in ts] + ["stop"]
        out_s.append(ws); out_t.append(ts)
    d = CFG["paths"]["processed"]; os.makedirs(d, exist_ok=True)
    np.save(f"{d}/{name}_sents.npy", np.array(out_s, dtype=object), allow_pickle=True)
    np.save(f"{d}/{name}_tags.npy", np.array(out_t, dtype=object), allow_pickle=True)
    rep = {"dataset": name, "n_sentences": len(out_s),
           "n_tokens": int(sum(len(s) for s in out_s)),
           "final_tag_counts": dict(Counter(t for ts in out_t for t in ts).most_common())}
    if extra: rep.update(extra)
    json.dump(rep, open(f"{d}/{name}_extraction_report.json", "w"), indent=1, ensure_ascii=False)
    print(f"[{name}] {len(out_s)} sents, {rep['n_tokens']} tokens", flush=True)
    return out_s, out_t

def oe():
    sents, tags = [], []
    for path in sorted(glob.glob(os.path.join(CFG["paths"]["raw"]["ycoe_psd_dir"], "*.psd"))):
        for leaves in iter_leaf_lists(open(path, encoding="utf-8", errors="replace").read()):
            ws, raw = clean_leaves(leaves)
            if ws:
                sents.append(ws); tags.append([penn_final(r) for r in raw])
    finalize("OE", sents, tags)

def os_():
    sents, tags = [], []
    for leaves in iter_leaf_lists(open(CFG["paths"]["raw"]["helipad_psd"], encoding="utf-8", errors="replace").read()):
        ws, raw = clean_leaves(leaves, form_lemma=True)
        if ws:
            sents.append(ws); tags.append([penn_final(r) for r in raw])
    finalize("OS", sents, tags)

def oi():
    d = CFG["paths"]["raw"]["icepahc_tagged_dir"]
    sents, tags = [], []
    for f in CFG["extraction"]["oi_files"]:
        cur_w, cur_t = [], []
        for line in open(os.path.join(d, f), encoding="utf-8", errors="replace"):
            line = line.rstrip("\n")
            if not line.strip():
                if cur_w:
                    sents.append(cur_w); tags.append(cur_t); cur_w, cur_t = [], []
                continue
            parts = line.split("\t")
            if len(parts) < 2 or parts[0].startswith("*") or parts[0] == "0" or "<" in parts[0]:
                continue
            cur_w.append(parts[0]); cur_t.append(penn_final(parts[1]))
        if cur_w:
            sents.append(cur_w); tags.append(cur_t)
    finalize("OI", sents, tags)

def gothic():
    df = pd.read_csv(CFG["paths"]["raw"]["gothic_csv"], low_memory=False)
    sents, tags = [], []
    for _, seg in df.groupby("SegmentID", sort=True):
        seg = seg.sort_values("Position")
        ws, ts = [], []
        for _, row in seg.iterrows():
            w = str(row["Type"])
            if not w or w == "nan":
                continue
            ws.append(w); ts.append(GOTHIC.get(str(row["POS"]).split(",")[0].strip(), "XX"))
        if ws:
            sents.append(ws); tags.append(ts)
    finalize("GOTH", sents, tags)

def map_ohg(raw, split_def=False):
    s = str(raw).strip()
    if s in ("", "nan", "None") or "??" in s:
        return "XX"
    s2 = s.replace("?", "")
    alts = [a.strip() for a in s2.split("¦") if a]
    finals = {OHG_POS.get(a) for a in alts}
    if None in finals or len(finals) != 1:
        return "XX"
    f = finals.pop()
    if split_def and f == "D":
        base = alts[0]
        return "DDEF" if base.startswith("DD") else ("DINDEF" if base.startswith("DI") else "D")
    return f

def ohg():
    df = pd.read_csv(CFG["paths"]["raw"]["ohg_csv"], low_memory=False)
    df.loc[df["pos"].astype(str).str.startswith("$"), "lang"] = "PUNCT"
    for split_def, name in [(False, "OHG"), (True, "OHG_defcontrast")]:
        sents, tags, files = [], [], []
        cur_w, cur_t, cur_f, bad = [], [], [], False
        for w, pos, lang, fn in zip(df["text"], df["pos"], df["lang"], df["file"]):
            w = str(w)
            cur_w.append(w); cur_t.append(map_ohg(pos, split_def)); cur_f.append(str(fn))
            if str(lang) not in ("goh", "ogh", "PUNCT"):
                bad = True
            if w == ".":
                if cur_w and not bad:
                    sents.append(cur_w); tags.append(cur_t); files.append(cur_f[0])
                cur_w, cur_t, cur_f, bad = [], [], [], False
        if cur_w and not bad:
            sents.append(cur_w); tags.append(cur_t); files.append(cur_f[0])
        keep = [i for i, ts in enumerate(tags) if not all(t in MASKED for t in ts)]
        sents = [sents[i] for i in keep]; tags = [tags[i] for i in keep]; files = [files[i] for i in keep]
        out_s, out_t = finalize(name, sents, tags)
        if not split_def:
            d = CFG["paths"]["processed"]
            prefix = CFG["ohg_subsets"]["notker"]["files_prefix"]
            nk = [i for i, f in enumerate(files) if f.startswith(prefix)]
            for n in CFG["ohg_subsets"]["notker"]["n_sents"]:
                idx = nk[:n]
                np.save(f"{d}/OHG_notker{n}_sents.npy", np.array([out_s[i] for i in idx], dtype=object), allow_pickle=True)
                np.save(f"{d}/OHG_notker{n}_tags.npy", np.array([out_t[i] for i in idx], dtype=object), allow_pickle=True)
            n = CFG["ohg_subsets"]["heterogeneous"]["n_sents"]
            het = np.random.default_rng(0).choice(len(out_s), size=min(n, len(out_s)), replace=False)
            np.save(f"{d}/OHG_heterogeneous{n}_sents.npy", np.array([out_s[i] for i in het], dtype=object), allow_pickle=True)
            np.save(f"{d}/OHG_heterogeneous{n}_tags.npy", np.array([out_t[i] for i in het], dtype=object), allow_pickle=True)

def run_all():
    oe(); os_(); oi(); gothic(); ohg()

if __name__ == "__main__":
    run_all()
