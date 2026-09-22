"""Token-fusion BiLSTM model with attention pooling."""

from __future__ import annotations

import torch
import torch.nn as nn

from .constants import MIX_AAS, MIX_PROP_DIM, SERUM_EMBED_DIM


class MixtureEncoder(nn.Module):
    """
    Permutation-invariant DeepSets encoder over the variant set {K, T, I}:
        z = rho( sum_a  f_a * psi(a) )
    The model therefore sees a population, not an ordered triple, so the same
    composition expressed in any variant order yields the same representation.
    """

    def __init__(self, prop_dim: int = MIX_PROP_DIM, embed_dim: int = 32,
                 hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.aa_emb = nn.Embedding(len(MIX_AAS), embed_dim)
        self.psi = nn.Sequential(
            nn.Linear(embed_dim + prop_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU())
        self.rho = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))

    def forward(self, aa_ids, aa_props, aa_weights):
        h = self.psi(torch.cat([self.aa_emb(aa_ids), aa_props], dim=-1))
        return self.rho((h * aa_weights[:, :, None]).sum(dim=1))


class FeatureAssembler(nn.Module):
    """
    Assembles biology into a short token sequence.

    serum_mode='index' (7 tokens, 8 with PSSM):
        0 virus_mean   1 serum embedding   2 virus_focus   3 mixture z
        4 virus x z    5 serum x z         6 engineered features
        7 virus PSSM

    serum_mode='sequence' (11 tokens, 13 with PSSM):
        0 virus_mean      1 serum_mean       2 virus_focus     3 serum_focus
        4 |v-s| at focus  5 (v+s)/2          6 mixture z       7 v x z
        8 s x z           9 |v-s| x z       10 engineered features
       11 virus PSSM     12 serum PSSM
    """

    def __init__(self, seq_dim: int, pair_dim: int, mix_hidden: int = 64,
                 token_dim: int = 128, dropout: float = 0.1, pssm_dim: int = 0,
                 serum_mode: str = "index", n_sera: int = 0):
        super().__init__()
        self.use_pssm = pssm_dim > 0
        self.serum_mode = serum_mode
        self.mixture_enc = MixtureEncoder(MIX_PROP_DIM, 32, mix_hidden, dropout)
        mk = lambda d: nn.Sequential(nn.Linear(d, token_dim), nn.LayerNorm(token_dim))
        self.feat_proj = mk(pair_dim)
        self.mix_proj = mk(mix_hidden)
        self.int_up = nn.Linear(mix_hidden, token_dim)
        self.v_int = nn.Linear(seq_dim, mix_hidden)
        if self.use_pssm:
            self.pssm_proj = mk(pssm_dim)

        if serum_mode == "index":
            if n_sera < 1:
                raise ValueError("serum_mode='index' requires n_sera >= 1")
            self.serum_emb = nn.Embedding(n_sera, SERUM_EMBED_DIM)
            self.seq_proj = mk(seq_dim)
            self.serum_proj = mk(SERUM_EMBED_DIM)
            self.s_int = nn.Linear(SERUM_EMBED_DIM, mix_hidden)
            self.vs_int = nn.Linear(SERUM_EMBED_DIM, mix_hidden)
            self.n_tokens = 8 if self.use_pssm else 7
        else:
            self.seq_proj, self.pair_proj = mk(seq_dim), mk(seq_dim)
            self.centroid_proj = mk(seq_dim)
            self.s_int = nn.Linear(seq_dim, mix_hidden)
            self.p_int = nn.Linear(seq_dim, mix_hidden)
            self.n_tokens = 13 if self.use_pssm else 11

    def forward(self, virus_mean, virus_focus, pair_features, aa_ids, aa_props,
                aa_weights, serum_idx=None, serum_mean=None, serum_focus=None,
                virus_pssm=None, serum_pssm=None):
        z = self.mixture_enc(aa_ids, aa_props, aa_weights)

        if self.serum_mode == "index":
            e_s = self.serum_emb(serum_idx.clamp(min=0))
            tokens = [
                self.seq_proj(virus_mean), self.serum_proj(e_s),
                self.seq_proj(virus_focus), self.mix_proj(z),
                self.int_up(self.v_int(virus_focus) * z),
                self.int_up(self.s_int(e_s) * z),
                self.feat_proj(pair_features),
            ]
            if self.use_pssm and virus_pssm is not None:
                tokens.append(self.pssm_proj(virus_pssm))
            return torch.stack(tokens, dim=1)

        diff = torch.abs(virus_focus - serum_focus)
        cent = (virus_focus + serum_focus) * 0.5
        tokens = [
            self.seq_proj(virus_mean), self.seq_proj(serum_mean),
            self.seq_proj(virus_focus), self.seq_proj(serum_focus),
            self.pair_proj(diff), self.centroid_proj(cent), self.mix_proj(z),
            self.int_up(self.v_int(virus_focus) * z),
            self.int_up(self.s_int(serum_focus) * z),
            self.int_up(self.p_int(diff) * z),
            self.feat_proj(pair_features),
        ]
        if self.use_pssm and virus_pssm is not None:
            tokens += [self.pssm_proj(virus_pssm), self.pssm_proj(serum_pssm)]
        return torch.stack(tokens, dim=1)


class BiLSTMHead(nn.Module):
    """BiLSTM over the token sequence with additive-attention pooling."""

    def __init__(self, token_dim: int = 128, hidden: int = 128, layers: int = 2,
                 dropout: float = 0.15):
        super().__init__()
        self.bilstm = nn.LSTM(token_dim, hidden, layers, batch_first=True,
                              bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        d = hidden * 2
        self.attn = nn.Sequential(nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, 1))
        self.out = nn.Sequential(
            nn.Linear(d, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 1))

    def forward(self, tokens, return_attention: bool = False):
        self.bilstm.flatten_parameters()
        h, _ = self.bilstm(tokens)
        a = torch.softmax(self.attn(h), dim=1)
        ctx = (h * a).sum(dim=1)
        y = self.out(ctx)
        return (y, a.squeeze(-1)) if return_attention else y


class MPATModel(nn.Module):
    """Full predictor: HA embeddings + mixture DeepSets + BiLSTM token head."""

    def __init__(self, seq_dim: int, pair_dim: int, mix_hidden: int = 64,
                 token_dim: int = 128, dropout: float = 0.15, pssm_dim: int = 0,
                 lstm_hidden: int = 128, lstm_layers: int = 2, ablation: str = "full",
                 serum_mode: str = "index", n_sera: int = 0):
        super().__init__()
        self.ablation = ablation
        self.use_pssm = pssm_dim > 0
        self.serum_mode = serum_mode
        self.assembler = FeatureAssembler(seq_dim, pair_dim, mix_hidden, token_dim,
                                          dropout, pssm_dim, serum_mode, n_sera)
        self.head = BiLSTMHead(token_dim, lstm_hidden, lstm_layers, dropout)

    def _ablate(self, vm, sm, vf, sf, pf, aw):
        z = torch.zeros_like
        zo = lambda t: z(t) if t is not None else None
        if self.ablation in ("no_mixture", "sequence_only"):
            aw = z(aw)
        elif self.ablation == "mixture_only":
            vm, sm, vf, sf, pf = z(vm), zo(sm), z(vf), zo(sf), z(pf)
        elif self.ablation == "no_sequence_embeddings":
            vm, sm, vf, sf = z(vm), zo(sm), z(vf), zo(sf)
        elif self.ablation == "no_pair_features":
            pf = z(pf)
        elif self.ablation != "full":
            raise ValueError(f"Unknown ablation: {self.ablation}")
        return vm, sm, vf, sf, pf, aw

    def forward(self, virus_mean, virus_focus, pair_features, aa_ids, aa_props,
                aa_weights, serum_idx=None, serum_mean=None, serum_focus=None,
                virus_pssm=None, serum_pssm=None, return_attention: bool = False):
        (virus_mean, serum_mean, virus_focus, serum_focus,
         pair_features, aa_weights) = self._ablate(
            virus_mean, serum_mean, virus_focus, serum_focus, pair_features, aa_weights)
        if self.use_pssm and self.ablation in ("no_sequence_embeddings", "mixture_only"):
            virus_pssm = torch.zeros_like(virus_pssm) if virus_pssm is not None else None
            serum_pssm = torch.zeros_like(serum_pssm) if serum_pssm is not None else None

        tokens = self.assembler(virus_mean, virus_focus, pair_features, aa_ids,
                                aa_props, aa_weights, serum_idx, serum_mean,
                                serum_focus, virus_pssm, serum_pssm)
        if return_attention:
            y, a = self.head(tokens, return_attention=True)
            return y.squeeze(-1), a
        return self.head(tokens).squeeze(-1)
