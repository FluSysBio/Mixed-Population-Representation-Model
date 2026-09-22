# MPAT

**Mixed Population Antigenic Titer prediction model**

Conventional antigenic prediction represents each virus by a single consensus sequence,
which assumes viral populations are genetically homogeneous. MPAT represents each
observation as a virus, an antiserum, and a **population composition**: the coexisting
variants at a chosen residue together with their relative frequencies. Populations sharing
the same variants but differing in abundance therefore receive different predicted titers.

The framework is general in the focal residue, the number and identity of variants, and
the serological assay. The shipped configuration is the case reported in the manuscript:
influenza A(H3N2) HA populations defined by K, T, and I at residue 160 (H3 numbering),
assayed by microneutralization.

---

## Quickstart

```bash
git clone https://github.com/<user>/mpat.git && cd mpat
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=src            # Windows: set PYTHONPATH=src

# 1. pretrain the sequence encoder (once)
python -m mpat pretrain --gisaid_fasta data/Public_HA_Sequence_Data.fasta --pdb data/HA_structure_template.pdb \
                        --out_encoder artefacts/encoder.pt

# 2. train and evaluate
python -m mpat train --data data/MN_Titers_Data.csv \
                     --virus_fasta data/Virus.fasta \
                     --gisaid_fasta data/Public_HA_Sequence_Data.fasta --pdb data/HA_structure_template.pdb \
                     --pretrained_encoder artefacts/encoder.pt \
                     --output_dir results --n_seeds 5

# 3. predict a composition that was never assayed
python -m mpat predict --bundle results/model_bundle.pt \
                       --virus A/HK/19 --serum A/Kan/17 --ratio 40:35:25
```

A GPU is optional; add `--cpu` to run on CPU, where pretraining takes 30 to 60 minutes and
training a few minutes.

---

## Inputs

| File | Description |
|---|---|
| Titer table (CSV) | One row per virus x serum x composition |
| Virus FASTA | HA protein sequences of the viral backgrounds |
| Background FASTA | Large public HA protein sequence dataset for pretraining and the PSSM prior |
| Structure (PDB) | Builds the residue contact graph |

**Titer table columns:** `strain`, `serum`, one fraction column per variant
(`K_fraction`, `T_fraction`, `I_fraction`), `log2_titer`, `censor_status`. 

`serum` is read as a **categorical panel identifier**: each distinct value becomes one
index in the antiserum panel and is learned as an embedding.

The public sequence set used in the manuscript came from GISAID EpiFlu and the structure is obtained from the RCSB
PDB entry 4WE4. See `data/README.md`.

---


## Usage

### Training and evaluation

```bash
python -m mpat train --data data/MN_Titers_Data.csv \
                     --virus_fasta data/Virus_Sequence.fasta \
                     --gisaid_fasta data/Public_HA_Sequence_Data.fasta --pdb data/HA_structure_template.pdb \
                     --pretrained_encoder artefacts/encoder.pt \
                     --output_dir results --n_seeds 5 --cv leave_one_ratio
```

`--cv leave_one_ratio` runs the leave-one-composition-out protocol reported as the main
generalization result; `--cv all` runs every protocol; `--cv_only` skips the final fit.

### Prediction

```bash
# single query
python -m mpat predict --bundle results/model_bundle.pt \
                       --virus A/HK/19 --serum A/Kan/17 --ratio 40:35:25

# full composition simplex for one virus x serum pair
python -m mpat predict --bundle results/model_bundle.pt \
                       --virus A/HK/19 --serum A/Kan/17 --simplex --grid_step 0.05

# batch queries
python -m mpat predict --bundle results/model_bundle.pt \
                       --query_csv queries.csv --out_csv preds.csv
```

The bundle carries the model ensemble, feature bank, fitted scaler, and encoder weights,
so prediction needs no FASTA or PDB unless a query names an unseen virus. Queries naming an
antiserum outside the training panel are rejected.

### As a library

```python
from mpat.inference import MPATPredictor   # requires src/ on PYTHONPATH

p = MPATPredictor("results/model_bundle.pt")
p.predict("A/HK/19", "A/Kan/17", (0.4, 0.35, 0.25))
# {'log2_titer': 5.12, 'titer_2fold': 32.0, 'titer_continuous': 34.8, ...}
```

All commands are run from the repository root with `src` on `PYTHONPATH`. Set it once per
shell, as in the Quickstart, or prefix each command:

```bash
PYTHONPATH=src python -m mpat train ...
```

---

## Options

| Option | Default | Effect |
|---|---|---|
| `--n_seeds` | 5 | Ensemble size; predictions averaged across seeds |
| `--epochs` | 400 | Maximum epochs, early stopping on validation MAE |
| `--lr` | 3e-4 | Peak learning rate for the one-cycle schedule |
| `--batch_size` | 16 | Minibatch size |
| `--label_noise_std` | 0.05 | Gaussian noise on training targets only |
| `--serum_mode` | index | `index` reads the antiserum as a panel identifier; `sequence` uses HA sequences and requires `--serum_fasta` |
| `--split_mode` | stratified | `ratio_holdout` and `pair_holdout` are stricter |
| `--cv` | none | `leave_one_ratio`, `leave_one_pair`, `random_grouped`, `all` |
| `--ablation` | none | `no_mixture`, `sequence_only`, `mixture_only`, `no_sequence_embeddings`, `no_pair_features` |
| `--cpu` | off | Force CPU execution |

`python -m mpat train --help` lists the rest.

---

## Outputs

Written to `--output_dir`:

| File | Contents |
|---|---|
| `metrics_all.json` | Train, validation, and test metrics with bootstrap CIs |
| `predictions_*.csv` | Per-observation observed and predicted titers |
| `split_{train,val,test}.csv` | Exact rows in each partition |
| `model_bundle.pt` | Self-contained inference bundle |
| `training_history.json` | Per-epoch loss and validation traces |
| `mixture_response_grid.csv` | Predicted titer surface over the composition simplex, with the additive expectation and the non-additive residual |
| `cross_validation/cv_summary.{csv,json}` | Pooled and per-fold CV results |
| `cross_validation/*_predictions.csv` | Per-observation held-out predictions per protocol |

### Figures

`notebooks/evaluation_figures.ipynb` reproduces the evaluation figures from these outputs.
It needs no model inference: point the `RESULTS` path at your `--output_dir` and run.

```bash
pip install matplotlib jupyter
jupyter lab notebooks/evaluation_figures.ipynb
```

---

## Adapting to another system

Configuration lives in `src/mpat/constants.py`.

| Change | What to do |
|---|---|
| Different focal residue or window | Set `FOCUS_POSITION` and `LOCAL_WINDOW`. Nothing else changes. |
| Different variants, same count | Replace `MIX_AAS`; add matching `MIX_AA_PROPS` entries of length `MIX_PROP_DIM`. |
| Different number of variants | Set `MIX_AAS`; the embedding table and DeepSets size themselves. Also edit the fraction column names in `data.py` and generalize `ratio_geometry_features()` in `features.py`, which currently writes the quadratic, pairwise, and triple terms explicitly. |
| Different assay | No code change. Supply log₂ titers and censoring flags; set `ASSAY_FLOOR_TITER` to the detection limit. |
| Different subtype or pathogen | Supply the matching structure and background FASTA, and replace `H3_EPITOPES`. The rest is sequence-agnostic. |

---

## Design notes

**Censoring.** Titers below the first assay dilution are left-censored, so the loss
penalizes over-prediction only for those observations via a one-sided Tobit-style Huber
term. Treating them as exact values at the substituted floor would bias predictions
downward at the detection limit.

**Replicates.** Replicates are collapsed to their geometric mean titer, with replicate
agreement supplying a measurement-quality weight, rather than being treated as independent
observations. `--use_replicates` reinstates the replicate-level form as label-noise
augmentation, keeping replicates within a single split.

**Leakage control.** Feature scaling, imbalance weights, and all context statistics are
computed on the training split alone and applied unchanged to validation and test.

**Evaluation protocol.** The stratified 80/10/10 split is an optimistic bound, since
related compositions appear on both sides. `leave_one_ratio` withholds an entire
composition per fold and is the protocol matching the extrapolation claim; `leave_one_pair`
withholds an entire virus x serum pair. Reporting the range across protocols is the
intended usage. The final row of each `*_metrics.csv` gives the pooled result plus the
across-fold mean, standard deviation, minimum, and maximum.

---

## Reproducing the manuscript results

```bash
python -m mpat train --data data/mn_titers_collapsed_ordered.csv \
                     --virus_fasta data/Virus.fasta \
                     --gisaid_fasta data/Gisaid_HA_Seq.fasta --pdb data/4WE4.pdb \
                     --pretrained_encoder artefacts/encoder.pt \
                     --output_dir results --n_seeds 5 --cv all --seed 42
```

Results are seed-dependent at the third decimal place.

## Citation
```bibtex
@article{prasai2026MPATmodel,
  title={Representing heterogeneous viral populations for machine learning–based antigenic prediction},
  author={Prasai K, Yang Z, Long Y, Wan X-F},
  journal={To be added},
  year={2026}
}
```

## License

## License

This project is distributed under the Academic Research License.
The software may be used for academic and research purposes only.  
See the [LICENSE](LICENSE) file for the complete license terms.
