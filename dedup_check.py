import json, os, sys
import numpy as np
from core import CFG, TagDataset, SplitEncoder, train, evaluate, load_lang

def run(lang):
    ds = load_lang(lang)
    out = []
    for seed in CFG["training"]["seeds"]:
        tr, dv, te = ds.split(seed)
        train_sents = {tuple(ds.sents[i]) for i in tr}
        te_dedup = [i for i in te if tuple(ds.sents[i]) not in train_sents]
        enc = SplitEncoder(ds, tr)
        model = train(ds, enc, tr, dv, seed)
        full = evaluate(model, ds, enc, te, n_examples=0)
        dedup = evaluate(model, ds, enc, te_dedup, n_examples=0)
        out.append({"seed": seed,
                    "test_sents": len(te), "dedup_test_sents": len(te_dedup),
                    "full_accuracy": full["overall_accuracy"],
                    "dedup_accuracy": dedup["overall_accuracy"],
                    "full_prf": full["prf"], "dedup_prf": dedup["prf"]})
        print(f"[{lang}] seed {seed}: full {full['overall_accuracy']:.4f} "
              f"dedup {dedup['overall_accuracy']:.4f} "
              f"({len(te)-len(te_dedup)} duplicate sentences removed)", flush=True)
    os.makedirs(CFG["paths"]["results"], exist_ok=True)
    json.dump(out, open(f"{CFG['paths']['results']}/dedup_{lang}.json", "w"), indent=1)

if __name__ == "__main__":
    for lang in (sys.argv[1:] or ["GOTH"]):
        run(lang)
