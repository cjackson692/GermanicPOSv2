# GermanicPOS
Neural network error analysis for understanding language change: training Bi-LSTM POS taggers on five early Germanic languages (Old English, Old Saxon, Old Icelandic, Old High German, Gothic) and reading their error patterns as signals of syntactic stability and change. This is the complete, configuration-driven pipeline for the revised study (v2); it supersedes the notebook-based v1 repository.

### Structure
Run everything with:
  - `python run.py`
    - Installs requirements, then runs end to end: extraction -> 5-seed main models -> 100-run ablation grid -> OHG contrast experiments -> statistics -> tables -> figures. All outputs land in results/. All settings (hyperparameters, seeds, splits, sample sizes, tag policies) live in config.yaml and nowhere else.
  - `python dedup_check.py GOTH`
    - Optional duplicate-sentence sensitivity check; deterministic seeds reproduce the main-run models.

Code:
  - `extract.py` — parses each source corpus (Penn .psd, IcePaHC .tagged, Wulfila and ReA token tables) into aligned sentence-tag arrays, applying the form-level tag conversions in `tag_maps.py`. Every extraction writes a report listing each source tag -> final tag mapping with token counts (these reports regenerate the paper's Appendix A).
  - `core.py` — dataset handling (train-only vocabulary with a trained UNK embedding, 80/10/10 splits, masking), the Bi-LSTM tagger, training with early stopping on development loss, evaluation (precision/recall/F1, confusion matrices, softmax entropy, error examples), and the most-frequent-tag baseline.
  - `run.py` — experiment orchestration and analysis (Welch + Mann-Whitney tests with Holm correction over the ablation grid, on both recall and F1).

Datasets:
  - `data/processed/` contains the extracted aligned sentence-tag arrays for OS, OIce, OHG (including the Notker/heterogeneous and definiteness-contrast subsets), and Gothic, plus extraction reports for all five languages.
  - The Old English arrays are placeholders: the YCOE license does not permit redistribution. Obtain the YCOE, place its psd/ folder at `data/raw/ycoe/psd`, and `extract.py` regenerates the arrays exactly (the OE extraction report documents the expected output).
  - Raw corpora are not distributed; see LICENSE_NOTE.md for where each dataset comes from and what may be shared.

Results:
  - `results/` is the complete results set from the study: overall and per-category tables (recall/precision/F1 with baselines and OOV rates), the full ablation grids (100 runs per sample size per language), confusion matrices, statistical tests (2,186 comparisons), softmax entropies, context-contribution table, OHG homogeneity and definiteness contrasts, the Gothic deduplication sensitivity check, and the code-generated figures.

### Dataset Citation Info
De Herdt, T. (1997). Project Wulfila : an electronic edition of the Gothic Bible. Universitaire Faculteiten Sint-Ignatius Antwerpen.

Taylor, A., A. Warner, S. Pintzuk, and F. Beths. (2003). The York-Toronto-Helsinki Parsed Corpus of Old English Prose. Oxford Text Archive.

Walkden, G. (2016). The HeliPaD: a parsed corpus of Old Saxon. International journal of corpus linguistics, 21(4), (pp. 559-571).

Wallenberg, J. C., Ingason A. K., Sigurðsson, E. F., and Rögnvaldsson E. (2011). Icelandic Parsed Historical Corpus (IcePaHC). Version 0.9. http://www.linguist.is/icelandic_treebank

Zeige, L. E. , Schnelle, G., Klotz, M., Donhauser, K., Gippert, J., Lühr, R. (2024). Deutsch Diachron Digital - Referenzkorpus Altdeutsch. Version 1.2. Humboldt-Universität zu Berlin.
