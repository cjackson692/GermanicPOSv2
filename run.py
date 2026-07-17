import csv, json, os, subprocess, sys

def _install():
    subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")],
                   check=True)

if __name__ == "__main__":
    _install()

from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats as sps
import extract
from core import CFG, TagDataset, SplitEncoder, train, evaluate, mft_baseline, load_lang

R = CFG["paths"]["results"]
os.makedirs(R, exist_ok=True)

def main_models():
    for lang in CFG["languages"]:
        ds = load_lang(lang)
        out = {"language": lang, "n_sentences": len(ds), "seeds": []}
        for seed in CFG["training"]["seeds"]:
            tr, dv, te = ds.split(seed)
            enc = SplitEncoder(ds, tr)
            model = train(ds, enc, tr, dv, seed)
            res = evaluate(model, ds, enc, te)
            res.update(seed=seed, baseline=mft_baseline(ds, tr, te),
                       oov_rate_test=enc.oov_rate(te), train_vocab_size=enc.vocab_size)
            out["seeds"].append(res)
            print(f"[{lang}] seed {seed} acc {res['overall_accuracy']:.4f}", flush=True)
        cats = sorted({c for s in out["seeds"] for c in s["prf"]})
        agg = {}
        for c in cats:
            agg[c] = {m: {"mean": float(np.mean([s["prf"][c][m] for s in out["seeds"] if c in s["prf"]])),
                          "sd": float(np.std([s["prf"][c][m] for s in out["seeds"] if c in s["prf"]], ddof=1))}
                      for m in ("precision", "recall", "f1")}
            agg[c]["support_mean"] = float(np.mean([s["prf"][c]["support"] for s in out["seeds"] if c in s["prf"]]))
        out["aggregate_prf"] = agg
        accs = [s["overall_accuracy"] for s in out["seeds"]]
        out["aggregate_overall"] = {"mean": float(np.mean(accs)), "sd": float(np.std(accs, ddof=1))}
        json.dump(out, open(f"{R}/main_{lang}.json", "w"), indent=1)

def ablation():
    for lang in CFG["languages"]:
        ds = load_lang(lang)
        sizes = [s for s in CFG["ablation"]["sizes"] if s <= len(ds)]
        if not sizes or sizes[-1] != len(ds):
            sizes.append(len(ds))
        path = f"{R}/ablation_{lang}.csv"
        new = not os.path.exists(path)
        f = open(path, "a", newline="")
        w = csv.writer(f)
        if new:
            w.writerow(["language", "tag", "sample_size", "run", "seed",
                        "recall", "precision", "f1", "support", "overall_accuracy"])
        for size in sizes:
            for r in range(CFG["ablation"]["runs_per_size"]):
                seed = CFG["ablation"]["seed_base"] + size * 1000 + r
                tr, dv, te = ds.split(seed, subsample=size)
                enc = SplitEncoder(ds, tr)
                model = train(ds, enc, tr, dv, seed)
                res = evaluate(model, ds, enc, te, n_examples=0)
                for tag, m in res["prf"].items():
                    w.writerow([lang, tag, size, r, seed, f"{m['recall']:.6f}",
                                f"{m['precision']:.6f}", f"{m['f1']:.6f}", m["support"],
                                f"{res['overall_accuracy']:.6f}"])
                f.flush()
                print(f"[{lang}] size {size} run {r+1} acc {res['overall_accuracy']:.4f}", flush=True)
        f.close()

def contrasts():
    p = CFG["paths"]["processed"]
    n_het = CFG["ohg_subsets"]["heterogeneous"]["n_sents"]
    subsets = [f"OHG_notker{n}" for n in CFG["ohg_subsets"]["notker"]["n_sents"]]
    subsets += [f"OHG_heterogeneous{n_het}", "OHG_defcontrast"]
    results = {}
    for name in subsets:
        ds = TagDataset(f"{p}/{name}_sents.npy", f"{p}/{name}_tags.npy")
        runs = []
        for seed in CFG["contrast_seeds"]:
            tr, dv, te = ds.split(seed)
            enc = SplitEncoder(ds, tr)
            model = train(ds, enc, tr, dv, seed)
            res = evaluate(model, ds, enc, te, n_examples=0)
            res.update(seed=seed, baseline=mft_baseline(ds, tr, te))
            runs.append(res)
            print(f"[{name}] seed {seed} acc {res['overall_accuracy']:.4f}", flush=True)
        results[name] = runs
    json.dump(results, open(f"{R}/ohg_contrasts.json", "w"), indent=1)

def holm(p):
    m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj

def statistics():
    dfs = {l: pd.read_csv(f"{R}/ablation_{l}.csv") for l in CFG["languages"]
           if os.path.exists(f"{R}/ablation_{l}.csv")}
    rows = []
    for metric in ("recall", "f1"):
        for tag, langs in CFG["comparable"].items():
            for l1, l2 in combinations([l for l in langs if l in dfs], 2):
                d1 = dfs[l1][dfs[l1].tag == tag]; d2 = dfs[l2][dfs[l2].tag == tag]
                for ss in sorted(set(d1.sample_size) & set(d2.sample_size)):
                    a = d1[d1.sample_size == ss][metric].values * 100
                    b = d2[d2.sample_size == ss][metric].values * 100
                    if len(a) < 5 or len(b) < 5:
                        continue
                    t, p_t = sps.ttest_ind(a, b, equal_var=False)
                    try:
                        _, p_u = sps.mannwhitneyu(a, b, alternative="two-sided")
                    except ValueError:
                        p_u = np.nan
                    na, nb = len(a), len(b)
                    sp = np.sqrt(((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1)) / max(na+nb-2, 1))
                    rows.append(dict(tag=tag, metric=metric, lang1=l1, lang2=l2, sample_size=ss,
                                     n1=na, n2=nb, mean1=a.mean(), mean2=b.mean(),
                                     sd1=a.std(ddof=1), sd2=b.std(ddof=1), welch_t=t,
                                     p_welch=p_t, p_mannwhitney=p_u,
                                     cohens_d=(a.mean()-b.mean())/sp if sp > 0 else np.nan))
    out = pd.DataFrame(rows)
    out["p_welch_holm"] = np.nan
    for _, g in out.groupby(["metric", "tag", "lang1", "lang2"]):
        out.loc[g.index, "p_welch_holm"] = holm(g.p_welch.values)
    out.to_csv(f"{R}/stats_all_pairs.csv", index=False)

def tables():
    rows, overall = [], []
    for lang in CFG["languages"]:
        path = f"{R}/main_{lang}.json"
        if not os.path.exists(path):
            continue
        res = json.load(open(path))
        for cat, m in res["aggregate_prf"].items():
            rows.append(dict(language=lang, category=cat,
                             recall_mean=m["recall"]["mean"], recall_sd=m["recall"]["sd"],
                             precision_mean=m["precision"]["mean"], precision_sd=m["precision"]["sd"],
                             f1_mean=m["f1"]["mean"], f1_sd=m["f1"]["sd"],
                             support_mean=m["support_mean"],
                             baseline_recall_mean=float(np.mean(
                                 [s["baseline"]["prf"].get(cat, {}).get("recall", np.nan)
                                  for s in res["seeds"]]))))
        overall.append(dict(language=lang, accuracy_mean=res["aggregate_overall"]["mean"],
                            accuracy_sd=res["aggregate_overall"]["sd"],
                            baseline_mean=float(np.mean([s["baseline"]["overall_accuracy"] for s in res["seeds"]])),
                            oov_rate_mean=float(np.mean([s["oov_rate_test"] for s in res["seeds"]]))))
        cats = sorted({c for s in res["seeds"] for c in s["confusion"]})
        mat = pd.DataFrame(0.0, index=cats, columns=cats)
        for s in res["seeds"]:
            for g, preds in s["confusion"].items():
                for p, n in preds.items():
                    if p in mat.columns:
                        mat.loc[g, p] += n / len(res["seeds"])
        mat.to_csv(f"{R}/confusion_{lang}.csv")
    pd.DataFrame(rows).to_csv(f"{R}/table1_revised.csv", index=False)
    pd.DataFrame(overall).to_csv(f"{R}/overall.csv", index=False)
    ctx, ent = [], []
    for lang in CFG["languages"]:
        path = f"{R}/main_{lang}.json"
        if not os.path.exists(path):
            continue
        res = json.load(open(path))
        for cat, m in res["aggregate_prf"].items():
            base = np.nanmean([s["baseline"]["prf"].get(cat, {}).get("recall", np.nan)
                               for s in res["seeds"]])
            ctx.append(dict(language=lang, category=cat,
                            model_recall=m["recall"]["mean"], baseline_recall=base,
                            context_contribution=m["recall"]["mean"] - base,
                            support_mean=m["support_mean"]))
        for s_ in res["seeds"]:
            for cat, h in s_["mean_softmax_entropy"].items():
                ent.append(dict(language=lang, category=cat, seed=s_["seed"], mean_entropy=h))
    pd.DataFrame(ctx).to_csv(f"{R}/context_contribution.csv", index=False)
    e = pd.DataFrame(ent).groupby(["language", "category"]).mean_entropy.agg(["mean", "std"]).reset_index()
    e.to_csv(f"{R}/softmax_entropy.csv", index=False)
    cpath = f"{R}/ohg_contrasts.json"
    if os.path.exists(cpath):
        rows2 = []
        for name, runs in json.load(open(cpath)).items():
            cats = sorted({c for r in runs for c in r["prf"]})
            for cat in cats:
                vals = [r["prf"][cat]["recall"] for r in runs if cat in r["prf"]]
                rows2.append(dict(subset=name, category=cat,
                                  recall_mean=float(np.mean(vals)),
                                  recall_sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                                  support_mean=float(np.mean([r["prf"][cat]["support"] for r in runs if cat in r["prf"]]))))
        pd.DataFrame(rows2).to_csv(f"{R}/ohg_contrast_table.csv", index=False)

def plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(f"{R}/figures", exist_ok=True)
    dfs = {l: pd.read_csv(f"{R}/ablation_{l}.csv") for l in CFG["languages"]
           if os.path.exists(f"{R}/ablation_{l}.csv")}
    for tag in ["ADJ", "D", "C", "CONJ", "N", "VB"]:
        for metric in ("recall", "f1"):
            plt.figure(figsize=(8, 5))
            for lang in ["OE", "OI", "OHG"]:
                if lang not in dfs:
                    continue
                d = dfs[lang][(dfs[lang].tag == tag) & (dfs[lang].sample_size <= 10000)]
                if d.empty:
                    continue
                g = d.groupby("sample_size")[metric]
                m, se = g.mean() * 100, (g.std() / g.count() ** 0.5) * 100
                plt.plot(m.index, m.values, marker="o", label=lang)
                plt.fill_between(m.index, (m - se).values, (m + se).values, alpha=0.2)
            plt.xlabel("Training sample size (sentences)"); plt.ylabel(f"{tag} {metric} (%)")
            plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(f"{R}/figures/ablation_{tag}_{metric}.png", dpi=200); plt.close()
    cpath = f"{R}/context_contribution.csv"
    if os.path.exists(cpath):
        c = pd.read_csv(cpath)
        cats = ["N", "VB", "AP", "PRO", "D", "ADV", "CONJ", "C", "ADJ", "TO"]
        c = c[c.category.isin(cats) & (c.support_mean >= 25)]
        langs = [l for l in CFG["languages"] if l in set(c.language)]
        x = np.arange(len(cats)); w = 0.8 / max(len(langs), 1)
        plt.figure(figsize=(9, 4.5))
        for i, lang in enumerate(langs):
            vals = [100 * c[(c.language == lang) & (c.category == t)].context_contribution.mean()
                    if not c[(c.language == lang) & (c.category == t)].empty else np.nan
                    for t in cats]
            plt.bar(x + i * w, vals, w, label=lang)
        plt.axhline(0, color="k", lw=0.8)
        plt.xticks(x + 0.4 - w / 2, cats)
        plt.ylabel("Model recall - baseline recall (pp)")
        plt.legend(); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
        plt.savefig(f"{R}/figures/context_contribution.png", dpi=200); plt.close()

if __name__ == "__main__":
    extract.run_all()
    main_models()
    ablation()
    contrasts()
    statistics()
    tables()
    plots()
