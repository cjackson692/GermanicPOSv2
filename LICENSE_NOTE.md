# Data licensing

The extraction scripts require the source corpora in `data/raw/` (paths in
`config.yaml`). The corpora are NOT distributed here and must be obtained from
their maintainers:

- **YCOE** (Old English): Oxford Text Archive license — no redistribution.
  `data/processed/OE_sents.npy` and `OE_tags.npy` are therefore placeholders;
  run `extract.py` against your licensed YCOE copy to regenerate them
  (the extraction report documents the expected output exactly).
- **HeliPaD** (Old Saxon): CC BY-NC-SA — processed derivatives shared here
  non-commercially with attribution, same license.
- **IcePaHC** (Old Icelandic): LGPL — processed derivatives shared here.
- **Project Wulfila** (Gothic): research use — processed derivatives shared
  for research replication; contact the project before further reuse.
- **Referenzkorpus Altdeutsch** (OHG): CC BY-NC-SA (via LAUDATIO) — processed
  derivatives shared here non-commercially with attribution.
