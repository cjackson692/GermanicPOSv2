import os
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
import yaml

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yaml")))
PAD_ID, UNK_ID = 0, 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TagDataset:
    def __init__(self, sents_path, tags_path):
        self.sents = [list(map(str, s)) for s in np.load(sents_path, allow_pickle=True)]
        self.tags = [list(map(str, t)) for t in np.load(tags_path, allow_pickle=True)]
        self.masked_tags = set(CFG["masked_tags"])
        tagset = sorted({t for ts in self.tags for t in ts})
        self.tag2id = {t: i + 1 for i, t in enumerate(tagset)}
        self.id2tag = {i: t for t, i in self.tag2id.items()}
        self.masked_ids = {self.tag2id[t] for t in self.masked_tags if t in self.tag2id}

    def __len__(self):
        return len(self.sents)

    def split(self, seed, subsample=None):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self))
        if subsample is not None and subsample < len(idx):
            idx = idx[:subsample]
        f = CFG["training"]["split"]
        c1, c2 = int(f[0] * len(idx)), int((f[0] + f[1]) * len(idx))
        return idx[:c1], idx[c1:c2], idx[c2:]

class SplitEncoder:
    def __init__(self, ds, train_idx):
        self.ds = ds
        self.p = CFG["oov"]["unk_dropout"]
        freq = Counter(w for i in train_idx for w in ds.sents[i])
        self.word2id = {w: i + 2 for i, w in enumerate(sorted(freq))}
        self.singletons = {w for w, c in freq.items() if c == 1}
        self.vocab_size = len(self.word2id) + 2

    def encode(self, idx, rng=None):
        ds = self.ds
        ws = []
        for w in ds.sents[idx]:
            wid = self.word2id.get(w, UNK_ID)
            if rng is not None and wid != UNK_ID and w in self.singletons and rng.random() < self.p:
                wid = UNK_ID
            ws.append(wid)
        ts = [ds.tag2id[t] for t in ds.tags[idx]]
        return ws, ts, [0 if t in ds.masked_ids else 1 for t in ts]

    def oov_rate(self, idx_set):
        tot = oov = 0
        for i in idx_set:
            for w, t in zip(self.ds.sents[i], self.ds.tags[i]):
                if t not in self.ds.masked_tags:
                    tot += 1; oov += w not in self.word2id
        return oov / max(tot, 1)

class Tagger(nn.Module):
    def __init__(self, vocab, tagset):
        super().__init__()
        m = CFG["model"]
        self.embedding = nn.Embedding(vocab, m["embedding_dim"], padding_idx=PAD_ID, scale_grad_by_freq=True)
        self.lstm = nn.LSTM(m["embedding_dim"], m["hidden_size"], num_layers=m["num_layers"],
                            bidirectional=True, batch_first=True,
                            dropout=m["dropout"] if m["num_layers"] > 1 else 0.0)
        self.dropout = nn.Dropout(m["dropout"])
        self.out = nn.Linear(m["hidden_size"] * 2, tagset)

    def forward(self, x):
        h, _ = self.lstm(self.embedding(x))
        return self.out(self.dropout(h))

def pad(seqs):
    L = max(len(s) for s in seqs)
    return np.array([s + [PAD_ID] * (L - len(s)) for s in seqs], dtype=np.int64)

def batches(indices, bs, rng=None):
    idx = np.array(indices)
    if rng is not None:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]

def _pass(model, enc, indices, bs, lossfn, opt=None, rng=None):
    model.train() if opt else model.eval()
    tl, tm = 0.0, 0
    with torch.enable_grad() if opt else torch.no_grad():
        for b in batches(indices, bs, rng):
            eb = [enc.encode(i, rng=rng if opt else None) for i in b]
            xb = torch.from_numpy(pad([e[0] for e in eb])).to(DEVICE)
            yb = torch.from_numpy(pad([e[1] for e in eb])).to(DEVICE)
            mb = torch.from_numpy(pad([e[2] for e in eb])).float().to(DEVICE)
            raw = lossfn(model(xb).transpose(1, 2), yb)
            loss = (raw * mb).sum() / mb.sum().clamp(min=1)
            if opt:
                opt.zero_grad(); loss.backward(); opt.step()
            tl += (raw * mb).sum().item(); tm += mb.sum().item()
    return tl / max(tm, 1)

def train(ds, enc, tr, dv, seed, verbose=False):
    torch.manual_seed(seed)
    t = CFG["training"]
    model = Tagger(enc.vocab_size, len(ds.tag2id) + 1).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
    lossfn = nn.CrossEntropyLoss(reduction="none", ignore_index=PAD_ID)
    best, best_state, patience = float("inf"), None, 0
    for epoch in range(t["max_epochs"]):
        rng = np.random.default_rng(seed * 1000 + epoch)
        _pass(model, enc, tr, t["batch_size"], lossfn, opt, rng)
        dv_loss = _pass(model, enc, dv, t["batch_size"] * 4, lossfn)
        if verbose:
            print(f"    epoch {epoch+1} dev {dv_loss:.4f}", flush=True)
        if dv_loss < best:
            best, patience = dv_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= t["patience"]:
                break
    model.load_state_dict(best_state)
    return model

def prf(conf):
    cats = sorted(set(conf) | {p for c in conf.values() for p in c})
    out = {}
    for c in cats:
        tp = conf[c][c] if c in conf else 0
        fn = (sum(conf[c].values()) - tp) if c in conf else 0
        fp = sum(conf[g][c] for g in conf if g != c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out[c] = {"precision": pr, "recall": rc,
                  "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0, "support": tp + fn}
    return out

def acc(conf):
    return sum(conf[c][c] for c in conf) / max(sum(sum(c.values()) for c in conf.values()), 1)

def evaluate(model, ds, enc, te, n_examples=5):
    model.eval()
    conf = defaultdict(Counter); ent_s, ent_n = Counter(), Counter()
    ex = defaultdict(list)
    with torch.no_grad():
        for b in batches(te, 256):
            eb = [enc.encode(i) for i in b]
            xb = torch.from_numpy(pad([e[0] for e in eb])).to(DEVICE)
            probs = torch.softmax(model(xb), dim=-1).cpu().numpy()
            preds = probs.argmax(-1)
            H = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(-1)
            for row, i in enumerate(b):
                _, tags, mask = eb[row]
                for j, (t, mf) in enumerate(zip(tags, mask)):
                    if not mf:
                        continue
                    g, p = ds.id2tag[t], ds.id2tag.get(int(preds[row, j]), "PAD")
                    conf[g][p] += 1
                    ent_s[g] += float(H[row, j]); ent_n[g] += 1
                    if g != p and len(ex[(g, p)]) < n_examples:
                        ex[(g, p)].append({"sentence": " ".join(ds.sents[i]),
                                           "token": ds.sents[i][j], "gold": g, "pred": p})
    return {"confusion": {g: dict(c) for g, c in conf.items()}, "prf": prf(conf),
            "overall_accuracy": acc(conf),
            "mean_softmax_entropy": {g: ent_s[g] / ent_n[g] for g in ent_s},
            "error_examples": {f"{g}->{p}": v for (g, p), v in ex.items()}}

def mft_baseline(ds, tr, te):
    wt = defaultdict(Counter); overall = Counter()
    for i in tr:
        for w, t in zip(ds.sents[i], ds.tags[i]):
            if t not in ds.masked_tags:
                wt[w][t] += 1; overall[t] += 1
    default = overall.most_common(1)[0][0]
    conf = defaultdict(Counter)
    for i in te:
        for w, t in zip(ds.sents[i], ds.tags[i]):
            if t not in ds.masked_tags:
                conf[t][wt[w].most_common(1)[0][0] if w in wt else default] += 1
    return {"prf": prf(conf), "overall_accuracy": acc(conf),
            "confusion": {g: dict(c) for g, c in conf.items()}}

def load_lang(lang):
    p = CFG["paths"]["processed"]
    spec = CFG["languages"][lang]
    return TagDataset(os.path.join(p, spec["sents"]), os.path.join(p, spec["tags"]))
