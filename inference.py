"""Self-contained inference bundle and prediction entry points."""

from __future__ import annotations

import re
from dataclasses import field
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from .constants import ASSAY_FLOOR_TITER, FOCUS_POSITION, MIX_AAS, MIX_AA_PROPS
from .encoder import HAGraphEncoder
from .features import CTX_COLS, encode_sequence_full, epitope_pair_features, glycosylation_features, pssm_pair_features, pssm_seq_features, ratio_geometry_features, train_only_context_features
from .model import MPATModel
from .utils import find_sequence, json_safe, nearest_twofold_titer, normalize_name, read_fasta


def save_bundle(path: Path, models, bank, scaler, encoder_state, encoder_cfg,
                adj, structure_info, pssm_logodds, pssm_entropy, train_reference,
                args, metrics, feature_names) -> None:
    bundle = {
        "format_version": "1.0",
        "model_states": [{k: v.detach().cpu() for k, v in m.state_dict().items()} for m in models],
        "model_config": {
            "seq_dim": bank.seq_dim, "pair_dim": int(scaler.mean_.shape[0]),
            "mix_hidden": args.mix_hidden_dim, "token_dim": args.token_dim,
            "dropout": args.dropout, "pssm_dim": bank.pssm_dim,
            "lstm_hidden": args.lstm_hidden, "lstm_layers": args.lstm_layers,
            "ablation": args.ablation,
        },
        "encoder_state": encoder_state,
        "encoder_config": encoder_cfg,
        "adjacency": adj.cpu(),
        "structure_info": structure_info,
        "pssm_logodds": pssm_logodds, "pssm_entropy": pssm_entropy,
        "use_pssm": bank.pssm_dim > 0,
        "graph_len": bank.graph_len,
        "seq_cache": {k: {kk: vv for kk, vv in v.items()} for k, v in bank.seq_cache.items()},
        "pair_cache": bank.pair_cache,
        "pair_names": bank.pair_names,
        "feature_names": feature_names,
        "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
        "train_reference": train_reference.to_dict(orient="list"),
        "mix_aas": MIX_AAS, "focus_position": FOCUS_POSITION,
        "serum_mode": bank.serum_mode, "serum_vocab": bank.serum_vocab,
        "args": vars(args), "metrics": json_safe(metrics),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)


class MPATPredictor:
    """
    Load a trained bundle and predict MN titers for any
    (virus, serum, K:T:I ratio) query - including ratios never assayed.

        p = MPATPredictor("results/model_bundle.pt")
        p.predict("A/HK/19", "A/Kan/17", (0.4, 0.35, 0.25))
        -> {'log2_titer': 5.12, 'titer_2fold': 32.0, 'titer_continuous': 34.8,
            'log2_sd_ensemble': 0.21, ...}
    """

    def __init__(self, bundle_path: str, device: str | None = None,
                 virus_fasta: str | None = None, serum_fasta: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        b = torch.load(bundle_path, map_location="cpu", weights_only=False)
        self.b = b
        self.graph_len = b["graph_len"]
        self.adj = b["adjacency"].to(self.device)
        self.structure_info = b["structure_info"]
        self.pssm_logodds, self.pssm_entropy = b["pssm_logodds"], b["pssm_entropy"]
        self.use_pssm = b["use_pssm"]
        self.seq_cache = dict(b["seq_cache"])
        self.pair_cache = dict(b["pair_cache"])
        self.train_reference = pd.DataFrame(b["train_reference"])
        self.scaler_mean = b["scaler_mean"]
        self.scaler_scale = b["scaler_scale"]

        ecfg = b["encoder_config"]
        self.encoder = HAGraphEncoder(ecfg.get("graph_len", self.graph_len),
                                      ecfg.get("embed_dim", 128), ecfg.get("layers", 3),
                                      ecfg.get("dropout", 0.1)).to(self.device)
        self.encoder.load_state_dict(b["encoder_state"], strict=False)
        self.encoder.eval()

        self.serum_mode = b.get("serum_mode", "sequence")
        self.serum_vocab = b.get("serum_vocab")

        mc = b["model_config"]
        self.models = []
        for state in b["model_states"]:
            m = MPATModel(seq_dim=mc["seq_dim"], pair_dim=mc["pair_dim"],
                                  mix_hidden=mc["mix_hidden"], token_dim=mc["token_dim"],
                                  dropout=mc["dropout"], pssm_dim=mc["pssm_dim"],
                                  lstm_hidden=mc["lstm_hidden"], lstm_layers=mc["lstm_layers"],
                                  ablation=mc.get("ablation", "full"),
                                  serum_mode=self.serum_mode,
                                  n_sera=len(self.serum_vocab or {})).to(self.device)
            m.load_state_dict(state)
            m.eval()
            self.models.append(m)

        self.extra_virus = read_fasta(virus_fasta) if virus_fasta else {}
        self.extra_serum = read_fasta(serum_fasta) if serum_fasta else {}

    # -- internals ---------------------------------------------------------
    def _seq_entry(self, name: str, role: str):
        key = ("V_" if role == "virus" else "S_") + normalize_name(name)
        if key in self.seq_cache:
            return self.seq_cache[key]
        pool = {**self.extra_virus, **self.extra_serum}
        seq = find_sequence(name, pool)
        if seq is None:
            raise KeyError(f"'{name}' is not in the bundle; pass --virus_fasta/--serum_fasta "
                           f"containing its HA sequence.")
        entry = encode_sequence_full(seq, self.encoder, self.adj, self.graph_len, self.device)
        entry["seq"] = seq
        if self.use_pssm:
            entry["pssm"] = pssm_seq_features(seq, self.pssm_logodds, self.pssm_entropy,
                                              self.graph_len)
        self.seq_cache[key] = entry
        return entry

    def serum_index(self, serum: str) -> int:
        idx = (self.serum_vocab or {}).get(normalize_name(serum))
        if idx is None:
            raise KeyError(
                f"'{serum}' is not in the antiserum panel this model was trained on. "
                f"Known antisera: {sorted(self.serum_vocab or {})}")
        return idx

    def _pair_vec(self, virus: str, serum: str) -> np.ndarray:
        if self.serum_mode == "index":
            key = normalize_name(virus)
            if key not in self.pair_cache:
                raise KeyError(f"'{virus}' is not in the bundle; pass --virus_fasta")
            return self.pair_cache[key]

        key = normalize_name(virus) + "__" + normalize_name(serum)
        if key in self.pair_cache:
            return self.pair_cache[key]
        ve, se = self._seq_entry(virus, "virus"), self._seq_entry(serum, "serum")
        ep, _ = epitope_pair_features(ve["seq"], se["seq"], self.graph_len, self.structure_info)
        parts = [ep,
                 glycosylation_features(ve["seq"], self.graph_len, self.structure_info),
                 glycosylation_features(se["seq"], self.graph_len, self.structure_info),
                 np.abs(ve["local_mean"] - se["local_mean"]),
                 (ve["local_mean"] + se["local_mean"]) / 2.0,
                 np.abs(ve["local_max"] - se["local_max"])]
        if self.use_pssm:
            parts.append(pssm_pair_features(ve["seq"], se["seq"], self.pssm_logodds, self.graph_len))
        vec = np.concatenate(parts).astype(np.float32)
        self.pair_cache[key] = vec
        return vec

    def _batch_inputs(self, queries: list[tuple[str, str, tuple]]):
        q = pd.DataFrame([{"virus_strain": v, "serum_strain": s,
                           "K_fraction": float(r[0]), "T_fraction": float(r[1]),
                           "I_fraction": float(r[2])} for v, s, r in queries])
        tot = q[["K_fraction", "T_fraction", "I_fraction"]].sum(axis=1)
        if (tot > 1.5).any():
            q[["K_fraction", "T_fraction", "I_fraction"]] /= 100.0
            tot = q[["K_fraction", "T_fraction", "I_fraction"]].sum(axis=1)
        for c in ("K_fraction", "T_fraction", "I_fraction"):
            q[c] = q[c] / tot
        q = train_only_context_features(self.train_reference, q)

        rows = []
        for _, r in q.iterrows():
            w = np.asarray([r["K_fraction"], r["T_fraction"], r["I_fraction"]], np.float32)
            rows.append(np.concatenate([
                self._pair_vec(r["virus_strain"], r["serum_strain"]),
                ratio_geometry_features(w),
                r[[c for c in CTX_COLS if c in q.columns]].to_numpy(np.float32)]))
        X = (np.asarray(rows, np.float32) - self.scaler_mean) / self.scaler_scale
        return q, X.astype(np.float32)

    # -- public API --------------------------------------------------------
    @torch.no_grad()
    def predict_many(self, queries: list[tuple[str, str, tuple]]) -> pd.DataFrame:
        q, X = self._batch_inputs(queries)
        aa_ids = torch.tensor(np.arange(len(MIX_AAS)))[None].repeat(len(q), 1).to(self.device)
        aa_props = torch.tensor(np.asarray([MIX_AA_PROPS[a] for a in MIX_AAS], np.float32))[None] \
            .repeat(len(q), 1, 1).to(self.device)
        aa_w = torch.tensor(q[["K_fraction", "T_fraction", "I_fraction"]]
                            .to_numpy(np.float32)).to(self.device)
        Xt = torch.tensor(X).to(self.device)

        def emb(name, role, field):
            e = self._seq_entry(name, role)
            return torch.tensor(np.stack([e[field]]))

        vm = torch.cat([emb(r["virus_strain"], "virus", "mean") for _, r in q.iterrows()]).to(self.device)
        vf = torch.cat([emb(r["virus_strain"], "virus", "focus") for _, r in q.iterrows()]).to(self.device)
        vp = sp = sm = sf = si = None
        if self.use_pssm:
            vp = torch.cat([emb(r["virus_strain"], "virus", "pssm") for _, r in q.iterrows()]).to(self.device)

        if self.serum_mode == "index":
            si = torch.tensor([self.serum_index(r["serum_strain"]) for _, r in q.iterrows()],
                              dtype=torch.long).to(self.device)
        else:
            sm = torch.cat([emb(r["serum_strain"], "serum", "mean") for _, r in q.iterrows()]).to(self.device)
            sf = torch.cat([emb(r["serum_strain"], "serum", "focus") for _, r in q.iterrows()]).to(self.device)
            if self.use_pssm:
                sp = torch.cat([emb(r["serum_strain"], "serum", "pssm") for _, r in q.iterrows()]).to(self.device)

        preds = np.stack([m(vm, vf, Xt, aa_ids, aa_props, aa_w, serum_idx=si,
                            serum_mean=sm, serum_focus=sf,
                            virus_pssm=vp, serum_pssm=sp).cpu().numpy()
                          for m in self.models])
        mean, sd = preds.mean(0), preds.std(0)
        out = q[["virus_strain", "serum_strain", "K_fraction", "T_fraction", "I_fraction"]].copy()
        out["log2_titer"] = mean
        out["log2_sd_ensemble"] = sd
        out["titer_continuous"] = np.power(2.0, mean)
        out["titer_2fold"] = nearest_twofold_titer(mean)
        out["log2_ci_lower"] = mean - 1.96 * sd
        out["log2_ci_upper"] = mean + 1.96 * sd
        out["below_assay_floor"] = out["titer_2fold"] < ASSAY_FLOOR_TITER
        return out

    def predict(self, virus: str, serum: str, ratio) -> dict:
        return self.predict_many([(virus, serum, tuple(ratio))]).iloc[0].to_dict()

    def predict_simplex(self, virus: str, serum: str, step: float = 0.05) -> pd.DataFrame:
        grid = []
        for k in np.round(np.arange(0, 1 + 1e-9, step), 4):
            for t in np.round(np.arange(0, 1 - k + 1e-9, step), 4):
                grid.append((virus, serum, (float(k), float(t), float(max(1 - k - t, 0.0)))))
        return self.predict_many(grid)


def parse_ratio(text: str) -> tuple[float, float, float]:
    """'40:35:25', '0.4,0.35,0.25' or '100:0:0' -> normalised (K, T, I)."""
    parts = [float(x) for x in re.split(r"[:,/\s]+", str(text).strip().strip("'")) if x != ""]
    if len(parts) != 3:
        raise ValueError(f"Ratio must have three values (K:T:I), got: {text}")
    a = np.asarray(parts, dtype=float)
    if a.sum() <= 0:
        raise ValueError("Ratio values sum to zero")
    return tuple((a / a.sum()).tolist())
