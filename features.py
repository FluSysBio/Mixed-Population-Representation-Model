"""Feature construction for virus, serum, and mixture composition."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from scipy import stats

from .constants import AA_LIST, AA_PROPS_4, EPITOPE_NAMES, FOCUS_POSITION, H3_EPITOPES, LOCAL_WINDOW, MIX_AAS, MIX_AA_PROPS, MIX_PROP_DIM
from .encoder import HAGraphEncoder
from .utils import find_sequence, normalize_name, seq_to_indices


def encode_sequence_full(seq: str, encoder: HAGraphEncoder, adj: torch.Tensor,
                         graph_len: int, device: torch.device,
                         focus: int = FOCUS_POSITION) -> dict[str, np.ndarray]:
    x = torch.tensor(seq_to_indices(seq, graph_len), dtype=torch.long, device=device)[None, :]
    with torch.no_grad():
        h = encoder.encode(x, adj.to(device)).squeeze(0).cpu().numpy()
    mean_emb = h.mean(axis=0)
    fi = focus - 1
    focus_emb = h[fi] if 0 <= fi < h.shape[0] else mean_emb
    lo, hi = max(0, LOCAL_WINDOW[0] - 1), min(graph_len, LOCAL_WINDOW[1])
    win = h[lo:hi] if hi > lo else h
    return {
        "mean": mean_emb.astype(np.float32),
        "focus": focus_emb.astype(np.float32),
        "local_mean": win.mean(axis=0).astype(np.float32),
        "local_max": win.max(axis=0).astype(np.float32),
    }


def pssm_seq_features(seq: str, logodds: np.ndarray, entropy: np.ndarray,
                      graph_len: int) -> np.ndarray:
    """42-dim: per-epitope PSSM summaries + global stats + residue-160 profile."""
    arr = seq_to_indices(seq, graph_len)
    scores = np.zeros(graph_len, dtype=np.float32)
    for i in range(graph_len):
        if arr[i] < len(AA_LIST):
            scores[i] = logodds[i, arr[i]]

    feats = []
    for site in EPITOPE_NAMES:
        idx = [p - 1 for p in H3_EPITOPES[site] if 0 <= p - 1 < graph_len]
        if idx:
            ss = scores[idx]
            feats += [ss.mean(), ss.std(), ss.max(), entropy[idx].mean()]
        else:
            feats += [0.0] * 4
    feats += [float(scores.mean()), float(scores.std())]
    fi = FOCUS_POSITION - 1
    feats += logodds[fi, :].tolist() if 0 <= fi < graph_len else [0.0] * len(AA_LIST)
    return np.asarray(feats, dtype=np.float32)


def pssm_pair_features(vseq: str, sseq: str, logodds: np.ndarray, graph_len: int) -> np.ndarray:
    """13-dim PSSM divergence between a virus HA and the serum's vaccine HA."""
    varr, sarr = seq_to_indices(vseq, graph_len), seq_to_indices(sseq, graph_len)
    vs = np.zeros(graph_len, np.float32)
    ss = np.zeros(graph_len, np.float32)
    for i in range(graph_len):
        if varr[i] < len(AA_LIST):
            vs[i] = logodds[i, varr[i]]
        if sarr[i] < len(AA_LIST):
            ss[i] = logodds[i, sarr[i]]
    diff_mask = (varr != sarr) & (varr < len(AA_LIST)) & (sarr < len(AA_LIST))
    div = np.abs(vs - ss)

    feats = []
    for site in EPITOPE_NAMES:
        idx = np.asarray([p - 1 for p in H3_EPITOPES[site] if 0 <= p - 1 < graph_len])
        if len(idx):
            m = diff_mask[idx]
            feats += [float(div[idx][m].mean()) if m.any() else 0.0, float(m.mean())]
        else:
            feats += [0.0, 0.0]
    feats += [float(div[diff_mask].mean()) if diff_mask.any() else 0.0, float(div.max())]
    fi = FOCUS_POSITION - 1
    feats += ([float(vs[fi]), float(ss[fi]), float(div[fi])] if 0 <= fi < graph_len else [0.0] * 3)
    return np.asarray(feats, dtype=np.float32)


def glycosylation_features(seq: str, graph_len: int, structure_info: dict) -> np.ndarray:
    """6-dim N-linked sequon descriptors relative to residue 160."""
    s = str(seq).replace("-", "").upper()[:graph_len].ljust(graph_len, "X")
    motifs = [i + 1 for i in range(graph_len - 2)
              if s[i] == "N" and s[i + 1] != "P" and s[i + 2] in ("S", "T")]
    if motifs:
        d = np.abs(np.asarray(motifs, np.float32) - FOCUS_POSITION)
        min_pos, near10, near20 = float(d.min()), float((d <= 10).any()), float((d <= 20).any())
        idx = [p - 1 for p in motifs if 0 <= p - 1 < graph_len]
        min_struct = float(structure_info["dist_to_focus"][idx].min()) if idx else 999.0
        n_contact = float(structure_info["contact_to_focus"][idx].sum()) if idx else 0.0
    else:
        min_pos, near10, near20, min_struct, n_contact = 999.0, 0.0, 0.0, 999.0, 0.0
    return np.asarray([len(motifs), near10, near20, min_pos, min_struct, n_contact], np.float32)


def epitope_pair_features(vseq: str, sseq: str, graph_len: int,
                          structure_info: dict) -> tuple[np.ndarray, list[str]]:
    """Epitope mismatch counts, structure-weighted mismatch, biochemical distance."""
    v = str(vseq).replace("-", "").upper()[:graph_len].ljust(graph_len, "X")
    s = str(sseq).replace("-", "").upper()[:graph_len].ljust(graph_len, "X")
    diff = np.asarray([v[i] != s[i] for i in range(graph_len)], dtype=bool)

    feat = {"ha_mismatch_count": float(diff.sum()), "ha_mismatch_fraction": float(diff.mean())}
    for site in EPITOPE_NAMES:
        idx = [p - 1 for p in H3_EPITOPES[site] if 0 <= p - 1 < graph_len]
        sd = diff[idx] if idx else np.zeros(1, bool)
        feat[f"epitope_{site}_mismatch_count"] = float(sd.sum())
        feat[f"epitope_{site}_mismatch_fraction"] = float(sd.mean())

    mi = np.where(diff)[0]
    if len(mi):
        feat["struct_mismatch_mean_degree"] = float(structure_info["degree"][mi].mean())
        feat["struct_mismatch_max_degree"] = float(structure_info["degree"][mi].max())
        feat["struct_mismatch_min_dist_160"] = float(structure_info["dist_to_focus"][mi].min())
        feat["struct_mismatch_mean_dist_160"] = float(structure_info["dist_to_focus"][mi].mean())
        feat["struct_mismatch_contact160"] = float(structure_info["contact_to_focus"][mi].sum())
    else:
        feat.update({"struct_mismatch_mean_degree": 0.0, "struct_mismatch_max_degree": 0.0,
                     "struct_mismatch_min_dist_160": 999.0, "struct_mismatch_mean_dist_160": 999.0,
                     "struct_mismatch_contact160": 0.0})

    # per-epitope biochemical distance (conservative vs disruptive substitutions)
    for site in EPITOPE_NAMES:
        idx = [p - 1 for p in H3_EPITOPES[site] if 0 <= p - 1 < graph_len]
        if not idx:
            for k in range(10):
                feat[f"biochem_{site}_{k}"] = 0.0
            continue
        d_all = np.asarray([np.abs(AA_PROPS_4.get(v[i], np.zeros(4)) -
                                   AA_PROPS_4.get(s[i], np.zeros(4))) for i in idx])
        d_mis = np.asarray([np.abs(AA_PROPS_4.get(v[i], np.zeros(4)) -
                                   AA_PROPS_4.get(s[i], np.zeros(4))) for i in idx if v[i] != s[i]])
        if len(d_mis) == 0:
            d_mis = np.zeros((1, 4), np.float32)
        vals = d_all.mean(0).tolist() + d_mis.mean(0).tolist() + \
               [float(d_all.sum()), float(np.linalg.norm(d_all, axis=1).max())]
        for k, val in enumerate(vals):
            feat[f"biochem_{site}_{k}"] = float(val)

    return np.asarray(list(feat.values()), np.float32), list(feat.keys())


RATIO_FEATURE_NAMES = (
    ["w_K", "w_T", "w_I", "w_K2", "w_T2", "w_I2", "w_KT", "w_KI", "w_TI", "w_KTI",
     "w_max", "w_min", "w_gap", "shannon_evenness", "simpson", "effective_n",
     "is_pure", "has_zero"]
    + [f"mix_prop_mean_{i}" for i in range(MIX_PROP_DIM)]
    + [f"mix_prop_var_{i}" for i in range(MIX_PROP_DIM)]
)


def ratio_geometry_features(weights: np.ndarray) -> np.ndarray:
    """
    Simplex geometry + polynomial mixture terms for [K, T, I].

    The quadratic / cross / triple-product terms are the explicit non-additive
    (epistatic) basis: a purely additive mixture-to-titer map needs only w_K,
    w_T, w_I, so any weight the model puts on the higher-order terms is direct
    evidence of interaction between co-circulating variants.
    """
    w = np.asarray(weights, dtype=np.float32)
    eps = 1e-8
    sw = np.sort(w)[::-1]
    feats = [
        w[0], w[1], w[2],
        w[0] ** 2, w[1] ** 2, w[2] ** 2,
        w[0] * w[1], w[0] * w[2], w[1] * w[2],
        w[0] * w[1] * w[2],
        float(w.max()), float(w.min()), float(sw[0] - sw[1]),
        float(-(w * np.log(w + eps)).sum() / np.log(len(w))),   # evenness
        float(1.0 - (w ** 2).sum()),                            # Simpson
        float(1.0 / ((w ** 2).sum() + eps)),                    # effective richness
        float(w.max() >= 0.999), float(w.min() <= 1e-6),
    ]
    props = np.asarray([MIX_AA_PROPS[a] for a in MIX_AAS], dtype=np.float32)
    mean_p = (w[:, None] * props).sum(0)
    var_p = (w[:, None] * props ** 2).sum(0) - mean_p ** 2
    feats += mean_p.tolist() + var_p.tolist()
    return np.asarray(feats, dtype=np.float32)


def train_only_context_features(train_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Context features computed during the TRAINING
    distance of a query mixture to observed training mixtures, and per-serum,
    per-virus titer summaries.
    """
    ratio_cols = ["K_fraction", "T_fraction", "I_fraction"]
    ref_ratios = train_df[ratio_cols].to_numpy(np.float32)
    # drop any previously attached context columns so repeated calls stay idempotent
    tgt = target_df.drop(columns=[c for c in target_df.columns if c.startswith("ctx_")],
                         errors="ignore").copy()

    d = np.abs(tgt[ratio_cols].to_numpy(np.float32)[:, None, :] - ref_ratios[None]).sum(-1)
    tgt["ctx_ratio_min_dist"] = d.min(1)
    tgt["ctx_ratio_mean_dist"] = d.mean(1)
    tgt["ctx_ratio_seen"] = (d.min(1) < 1e-6).astype(np.float32)

    for key, col in (("serum_strain", "ctx_serum"), ("virus_strain", "ctx_virus")):
        g = train_df.groupby(key)["log2_titer"]
        stats_df = pd.DataFrame({
            f"{col}_mean": g.mean(), f"{col}_median": g.median(),
            f"{col}_max": g.max(), f"{col}_std": g.std().fillna(0.0),
            f"{col}_breadth": g.apply(lambda s: float((s >= 5.0).mean())),
        })
        tgt = tgt.merge(stats_df, left_on=key, right_index=True, how="left")

    ctx_cols = [c for c in tgt.columns if c.startswith("ctx_")]
    global_mean = float(train_df["log2_titer"].mean())
    for c in ctx_cols:
        tgt[c] = tgt[c].fillna(global_mean if ("mean" in c or "median" in c or "max" in c) else 0.0)
    return tgt


CTX_COLS = ["ctx_ratio_min_dist", "ctx_ratio_mean_dist", "ctx_ratio_seen",
            "ctx_serum_mean", "ctx_serum_median", "ctx_serum_max", "ctx_serum_std",
            "ctx_serum_breadth", "ctx_virus_mean", "ctx_virus_median", "ctx_virus_max",
            "ctx_virus_std", "ctx_virus_breadth"]


@dataclass
class FeatureBank:
    """Everything needed to turn (virus, serum, ratio) into model inputs."""
    seq_cache: dict            # strain key -> embedding dict (+ pssm)
    pair_cache: dict           # pair key -> static feature vector
    pair_names: list
    graph_len: int
    seq_dim: int
    pssm_dim: int
    static_pair_dim: int
    serum_mode: str = "sequence"
    serum_vocab: dict | None = None   # normalized serum name -> index

    @property
    def n_sera(self) -> int:
        return len(self.serum_vocab) if self.serum_vocab else 0

    def pair_key(self, virus: str, serum: str) -> str:
        if self.serum_mode == "index":
            return normalize_name(virus)
        return normalize_name(virus) + "__" + normalize_name(serum)

    def serum_index(self, serum: str) -> int:
        """Index of an antiserum in the panel; -1 for one unseen at training time."""
        return (self.serum_vocab or {}).get(normalize_name(serum), -1)


def build_feature_bank(df: pd.DataFrame, vseqs: dict, sseqs: dict, encoder: HAGraphEncoder,
                       adj: torch.Tensor, structure_info: dict, graph_len: int,
                       pssm_logodds: np.ndarray, pssm_entropy: np.ndarray,
                       device: torch.device, use_pssm: bool = True,
                       serum_mode: str = "sequence") -> FeatureBank:
    """
    serum_mode:
        'sequence'  antisera are represented by the HA sequence of the immunizing
                    strain, giving virus-vs-serum comparison features.
        'index'     antisera are represented by a panel index read from the titer
                    table; no serum FASTA is required and all virus-vs-serum
                    sequence comparisons are dropped.
    """
    seq_cache, pair_cache, pair_names = {}, {}, None
    index_mode = serum_mode == "index"

    def _get(name, primary, secondary):
        s = find_sequence(name, primary) or find_sequence(name, secondary)
        if s is None:
            raise KeyError(f"No HA sequence found for '{name}'")
        return s

    for v in sorted(df["virus_strain"].astype(str).unique()):
        seq = _get(v, vseqs, sseqs)
        key = "V_" + normalize_name(v)
        seq_cache[key] = encode_sequence_full(seq, encoder, adj, graph_len, device)
        seq_cache[key]["seq"] = seq
        if use_pssm:
            seq_cache[key]["pssm"] = pssm_seq_features(seq, pssm_logodds, pssm_entropy, graph_len)

    for s in ([] if index_mode else sorted(df["serum_strain"].astype(str).unique())):
        seq = _get(s, sseqs, vseqs)
        key = "S_" + normalize_name(s)
        seq_cache[key] = encode_sequence_full(seq, encoder, adj, graph_len, device)
        seq_cache[key]["seq"] = seq
        if use_pssm:
            seq_cache[key]["pssm"] = pssm_seq_features(seq, pssm_logodds, pssm_entropy, graph_len)

    if index_mode:
        for v in sorted(df["virus_strain"].astype(str).unique()):
            vk = "V_" + normalize_name(v)
            vseq = seq_cache[vk]["seq"]
            parts = [glycosylation_features(vseq, graph_len, structure_info),
                     seq_cache[vk]["local_mean"],
                     seq_cache[vk]["local_max"]]
            pnames = ([f"glyc_v_{i}" for i in range(6)]
                      + [f"win_mean_{i}" for i in range(len(parts[1]))]
                      + [f"win_max_{i}" for i in range(len(parts[2]))])
            if use_pssm:
                parts.append(seq_cache[vk]["pssm"])
                pnames += [f"pssm_v_{i}" for i in range(len(parts[-1]))]
            pair_cache[normalize_name(v)] = np.concatenate(parts).astype(np.float32)
            pair_names = pnames

    for v, s in ([] if index_mode else
                 df[["virus_strain", "serum_strain"]].drop_duplicates().itertuples(index=False)):
        vk, sk = "V_" + normalize_name(v), "S_" + normalize_name(s)
        vseq, sseq = seq_cache[vk]["seq"], seq_cache[sk]["seq"]
        ep, names = epitope_pair_features(vseq, sseq, graph_len, structure_info)
        parts = [ep,
                 glycosylation_features(vseq, graph_len, structure_info),
                 glycosylation_features(sseq, graph_len, structure_info),
                 np.abs(seq_cache[vk]["local_mean"] - seq_cache[sk]["local_mean"]),
                 (seq_cache[vk]["local_mean"] + seq_cache[sk]["local_mean"]) / 2.0,
                 np.abs(seq_cache[vk]["local_max"] - seq_cache[sk]["local_max"])]
        pnames = (list(names)
                  + [f"glyc_v_{i}" for i in range(6)] + [f"glyc_s_{i}" for i in range(6)]
                  + [f"win_absdiff_{i}" for i in range(len(parts[3]))]
                  + [f"win_mean_{i}" for i in range(len(parts[4]))]
                  + [f"win_maxdiff_{i}" for i in range(len(parts[5]))])
        if use_pssm:
            pp = pssm_pair_features(vseq, sseq, pssm_logodds, graph_len)
            parts.append(pp)
            pnames += [f"pssm_pair_{i}" for i in range(len(pp))]
        pair_cache[normalize_name(v) + "__" + normalize_name(s)] = np.concatenate(parts).astype(np.float32)
        pair_names = pnames

    vocab = ({normalize_name(s): i
              for i, s in enumerate(sorted(df["serum_strain"].astype(str).unique()))}
             if index_mode else None)

    any_seq = next(iter(seq_cache.values()))
    return FeatureBank(
        seq_cache=seq_cache, pair_cache=pair_cache, pair_names=pair_names,
        graph_len=graph_len, seq_dim=int(any_seq["mean"].shape[0]),
        pssm_dim=int(any_seq["pssm"].shape[0]) if use_pssm else 0,
        static_pair_dim=int(next(iter(pair_cache.values())).shape[0]),
        serum_mode=serum_mode, serum_vocab=vocab,
    )


def build_row_features(df: pd.DataFrame, bank: FeatureBank) -> np.ndarray:
    """Per-row engineered vector = static pair features + ratio geometry + context."""
    out = []
    ctx_present = [c for c in CTX_COLS if c in df.columns]   # fixed, canonical order
    for _, row in df.iterrows():
        w = np.asarray([row["K_fraction"], row["T_fraction"], row["I_fraction"]], np.float32)
        parts = [bank.pair_cache[bank.pair_key(row["virus_strain"], row["serum_strain"])],
                 ratio_geometry_features(w)]
        if ctx_present:
            parts.append(row[ctx_present].to_numpy(np.float32))
        out.append(np.concatenate(parts))
    return np.asarray(out, dtype=np.float32)
