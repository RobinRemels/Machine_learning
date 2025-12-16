#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ----------------------------------------------------------------------
# Model definition (same as in training script)
# ----------------------------------------------------------------------
class CardMatchNet(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_level,
        id_emb_dim=32,
        lvl_emb_dim=8,
        hidden_dims=(128, 64),
        dropout=0.2,
    ):
        super().__init__()

        # Card ID embedding
        self.id_embed = nn.Embedding(vocab_size, id_emb_dim, padding_idx=None)
        # Level embedding: levels are 0..max_level
        self.lvl_embed = nn.Embedding(max_level + 1, lvl_emb_dim)

        per_card_emb = id_emb_dim + lvl_emb_dim

        # Player representation: mean & max pooling across 8 cards
        # -> per_card_emb * 2 dimensions (mean and max)
        player_vec_dim = per_card_emb * 2

        # Match representation: concat player1 + player2
        mlp_input = player_vec_dim * 2

        layers = []
        inp = mlp_input
        for h in hidden_dims:
            layers.append(nn.Linear(inp, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            inp = h
        # Final output: single logit
        layers.append(nn.Linear(inp, 1))

        self.mlp = nn.Sequential(*layers)

    def _player_repr(self, ids, lvls):
        """
        ids, lvls: (B, 8)
        Returns: (B, per_card_emb * 2)
        """
        id_e = self.id_embed(ids)  # (B, 8, id_emb_dim)
        lvl_e = self.lvl_embed(lvls)  # (B, 8, lvl_emb_dim)
        card_e = torch.cat([id_e, lvl_e], dim=-1)  # (B, 8, per_card_emb)

        avg = card_e.mean(dim=1)              # (B, per_card_emb)
        mx = card_e.max(dim=1).values         # (B, per_card_emb)
        return torch.cat([avg, mx], dim=-1)   # (B, per_card_emb * 2)

    def forward(self, p1_ids, p1_lvls, p2_ids, p2_lvls):
        p1_repr = self._player_repr(p1_ids, p1_lvls)
        p2_repr = self._player_repr(p2_ids, p2_lvls)
        x = torch.cat([p1_repr, p2_repr], dim=1)  # (B, 2 * player_vec_dim)
        logit = self.mlp(x).squeeze(1)           # (B,)
        return logit


# ----------------------------------------------------------------------
# Helper: batch prediction
# ----------------------------------------------------------------------
def predict_batch(p1_ids, p1_lvls, p2_ids, p2_lvls, model, device):
    """
    p?_ids, p?_lvls: numpy arrays of shape (B, 8)
    Returns: numpy array of shape (B,) with P(player1 wins)
    """
    model.eval()
    with torch.no_grad():
        p1_ids_t = torch.tensor(p1_ids, dtype=torch.long, device=device)
        p1_lvls_t = torch.tensor(p1_lvls, dtype=torch.long, device=device)
        p2_ids_t = torch.tensor(p2_ids, dtype=torch.long, device=device)
        p2_lvls_t = torch.tensor(p2_lvls, dtype=torch.long, device=device)

        logits = model(p1_ids_t, p1_lvls_t, p2_ids_t, p2_lvls_t)  # (B,)
        probs = torch.sigmoid(logits).cpu().numpy()
        return probs


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Test CardMatchNet on random matches.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="pytorch_card_model",
        help="Directory containing best_model.pt",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="matches_clean_randomized.csv",
        help="Cleaned matches CSV (with player1/player2 and winner_player columns)",
    )
    parser.add_argument(
        "--num-matches",
        type=int,
        default=10,
        help="Number of random matches to test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling matches",
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    ckpt_path = os.path.join(model_dir, "best_model.pt")
    csv_path = args.csv_path
    num_samples = args.num_matches

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ----------------------------
    # 1. Load cleaned dataset
    # ----------------------------
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    p1_id_cols = [f"player1.card{i}.id" for i in range(1, 9)]
    p1_lvl_cols = [f"player1.card{i}.level" for i in range(1, 9)]
    p2_id_cols = [f"player2.card{i}.id" for i in range(1, 9)]
    p2_lvl_cols = [f"player2.card{i}.level" for i in range(1, 9)]

    missing = [
        c
        for c in (p1_id_cols + p1_lvl_cols + p2_id_cols + p2_lvl_cols + ["winner_player"])
        if c not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    X_p1_ids = df[p1_id_cols].fillna(-1).astype(int).values  # (N, 8)
    X_p1_lvls = df[p1_lvl_cols].fillna(0).astype(int).values
    X_p2_ids = df[p2_id_cols].fillna(-1).astype(int).values
    X_p2_lvls = df[p2_lvl_cols].fillna(0).astype(int).values
    y = (df["winner_player"] == "player1").astype(int).values  # (N,)

    N = len(y)
    print(f"N matches: {N}")

    # ----------------------------
    # 2. Load checkpoint & model
    # ----------------------------
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"Loaded checkpoint from {ckpt_path}")

    vocab_size = ckpt["vocab_size"]
    max_level = ckpt["max_level"]
    id_emb_dim = ckpt.get("id_emb_dim", 32)
    lvl_emb_dim = ckpt.get("lvl_emb_dim", 8)
    hidden_dims = ckpt.get("hidden_dims", [128, 64])

    model = CardMatchNet(
        vocab_size=vocab_size,
        max_level=max_level,
        id_emb_dim=id_emb_dim,
        lvl_emb_dim=lvl_emb_dim,
        hidden_dims=hidden_dims,
        dropout=0.2,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("Model loaded:")
    print(model)

    # ----------------------------
    # 3. Sample random matches
    # ----------------------------
    rng = np.random.default_rng(seed=args.seed)
    k = min(num_samples, N)
    indices = rng.choice(N, size=k, replace=False)

    probs = predict_batch(
        X_p1_ids[indices],
        X_p1_lvls[indices],
        X_p2_ids[indices],
        X_p2_lvls[indices],
        model,
        device,
    )

    print(f"\n=== Predictions on {k} random matches ===")
    correct = 0
    for i, (idx, p) in enumerate(zip(indices, probs), start=1):
        true_label = int(y[idx])
        pred_label = int(p >= 0.5)
        if pred_label == true_label:
            correct += 1
        print(
            f"Match {i:02d} (row {idx}): "
            f"P(player1 wins) = {p:.3f}, "
            f"pred = {pred_label}, "
            f"true = {true_label}"
        )

    acc = correct / k * 100.0
    print(f"\nCorrect on {correct}/{k} sampled matches -> accuracy = {acc:.2f}%")

if __name__ == "__main__":
    main()
