# %%
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json

ID_EMBED_DIM = 32
LVL_EMBED_DIM = 8
HIDDEN_DIMS = [128, 64]
DROPOUT = 0.2
BATCH_SIZE = 1024
EPOCHS = 15
LR = 1e-3
MODEL_DIR = "pytorch_card_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
os.makedirs(MODEL_DIR, exist_ok=True)

# %%
df = pd.read_csv("BattlesStaging_01012021_WL_tagged.csv")

# %%
# -------------------------------------------------------------------
# 2. Identify card ID and level columns
# -------------------------------------------------------------------
winner_id_cols   = [f"winner.card{i}.id" for i in range(1, 9)]
winner_lvl_cols  = [f"winner.card{i}.level" for i in range(1, 9)]
loser_id_cols    = [f"loser.card{i}.id" for i in range(1, 9)]
loser_lvl_cols   = [f"loser.card{i}.level" for i in range(1, 9)]

all_card_id_cols = winner_id_cols + loser_id_cols

# -------------------------------------------------------------------
# 3. Extract unique card IDs across all matches and remap to integers
# -------------------------------------------------------------------
unique_card_ids = pd.unique(df[all_card_id_cols].values.ravel())

id_to_compact = {orig_id: idx for idx, orig_id in enumerate(unique_card_ids)}

# Save mapping
with open("card_id_mapping.json", "w") as f:
    json.dump({str(k): v for k, v in id_to_compact.items()}, f)

# Apply mapping
for col in all_card_id_cols:
    df[col] = df[col].map(id_to_compact)

# -------------------------------------------------------------------
# 4. Subset dataset to only card id + card level + winner/loser tags
# -------------------------------------------------------------------
df_small = df[winner_id_cols + winner_lvl_cols + loser_id_cols + loser_lvl_cols + ["winner.tag", "loser.tag"]].copy()

# -------------------------------------------------------------------
# 5. Randomize who is player1 and player2
# -------------------------------------------------------------------
flip = np.random.randint(0, 2, size=len(df_small))  # 0 = swap, 1 = normal

player1_ids   = np.where(flip[:, None] == 1, df_small[winner_id_cols].values,  df_small[loser_id_cols].values)
player1_lvls  = np.where(flip[:, None] == 1, df_small[winner_lvl_cols].values, df_small[loser_lvl_cols].values)

player2_ids   = np.where(flip[:, None] == 1, df_small[loser_id_cols].values,  df_small[winner_id_cols].values)
player2_lvls  = np.where(flip[:, None] == 1, df_small[loser_lvl_cols].values, df_small[winner_lvl_cols].values)

# Determine winner based on flip
winner_player = np.where(flip == 1, "player1", "player2")

# -------------------------------------------------------------------
# 6. Build final cleaned DataFrame
# -------------------------------------------------------------------
final_data = {}

# Player 1
for i in range(1, 9):
    final_data[f"player1.card{i}.id"]    = player1_ids[:, i-1]
    final_data[f"player1.card{i}.level"] = player1_lvls[:, i-1]

# Player 2
for i in range(1, 9):
    final_data[f"player2.card{i}.id"]    = player2_ids[:, i-1]
    final_data[f"player2.card{i}.level"] = player2_lvls[:, i-1]

# Winner label
final_data["winner_player"] = winner_player

# Optional: keep original player tags
final_data["winner.tag"] = df_small["winner.tag"].values
final_data["loser.tag"]  = df_small["loser.tag"].values

df_clean = pd.DataFrame(final_data)

# -------------------------------------------------------------------
# 7. Save final dataset
# -------------------------------------------------------------------
df_clean.to_csv("matches_clean_randomized.csv", index=False)

df_clean.head()


# %%
# Cell 2 - Load CSV & inspect
CSV_PATH = "matches_clean_randomized.csv"  # replace with your file path if different
df = pd.read_csv(CSV_PATH)
print("rows,cols:", df.shape)
df.head()


# %%
# Cell 3 - Extract card id & level arrays and labels

p1_id_cols = [f"player1.card{i}.id" for i in range(1,9)]
p1_lvl_cols = [f"player1.card{i}.level" for i in range(1,9)]
p2_id_cols = [f"player2.card{i}.id" for i in range(1,9)]
p2_lvl_cols = [f"player2.card{i}.level" for i in range(1,9)]

# Safety check: ensure columns exist
missing = [c for c in (p1_id_cols+p1_lvl_cols+p2_id_cols+p2_lvl_cols+["winner_player"]) if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# X arrays
X_p1_ids  = df[p1_id_cols].fillna(-1).astype(int).values  # shape (N,8)
X_p1_lvls = df[p1_lvl_cols].fillna(0).astype(int).values
X_p2_ids  = df[p2_id_cols].fillna(-1).astype(int).values
X_p2_lvls = df[p2_lvl_cols].fillna(0).astype(int).values

# Label: 1 if player1 won else 0
y = (df["winner_player"] == "player1").astype(int).values

print("N examples:", len(y))
print("Example ids p1:", X_p1_ids[0])
print("Example lvls p1:", X_p1_lvls[0])

# Determine vocab sizes
max_card_id = int(np.max(np.concatenate([X_p1_ids.ravel(), X_p2_ids.ravel()])))
if max_card_id < 0:
    raise ValueError("All card ids are negative or empty.")
vocab_size = max_card_id + 1  # assuming ids start at 0
max_level = int(np.max(np.concatenate([X_p1_lvls.ravel(), X_p2_lvls.ravel()])))
print("vocab_size (max id+1) =", vocab_size, "max_level =", max_level)


# %%
# Cell 4 - Train/Val split and class weights
X = {
    "p1_ids": X_p1_ids,
    "p1_lvls": X_p1_lvls,
    "p2_ids": X_p2_ids,
    "p2_lvls": X_p2_lvls
}

Xw_train, Xw_val, Xl_train, Xl_val, y_train, y_val = train_test_split(
    X["p1_ids"], X["p2_ids"], y, test_size=0.1, stratify=y, random_state=42
)
# But we split p1 ids and p2 ids together via indexes; better create indices:
# We'll create index split explicitly to keep lvls aligned

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
train_idx, val_idx = next(sss.split(X["p1_ids"], y))
Xw_train_ids  = X["p1_ids"][train_idx]
Xw_train_lvls = X["p1_lvls"][train_idx]
Xl_train_ids  = X["p2_ids"][train_idx]
Xl_train_lvls = X["p2_lvls"][train_idx]
y_train = y[train_idx]

Xw_val_ids  = X["p1_ids"][val_idx]
Xw_val_lvls = X["p1_lvls"][val_idx]
Xl_val_ids  = X["p2_ids"][val_idx]
Xl_val_lvls = X["p2_lvls"][val_idx]
y_val = y[val_idx]

print("Train/Val sizes:", len(y_train), len(y_val))

# class weights for loss (optional)
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
print("class_weights:", class_weights.cpu().numpy())


# %%
# Cell 5 - Dataset and DataLoader
class MatchDataset(Dataset):
    def __init__(self, p1_ids, p1_lvls, p2_ids, p2_lvls, labels):
        self.p1_ids = torch.tensor(p1_ids, dtype=torch.long)
        self.p1_lvls = torch.tensor(p1_lvls, dtype=torch.long)
        self.p2_ids = torch.tensor(p2_ids, dtype=torch.long)
        self.p2_lvls = torch.tensor(p2_lvls, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.p1_ids[idx], self.p1_lvls[idx],
                self.p2_ids[idx], self.p2_lvls[idx],
                self.labels[idx])

train_ds = MatchDataset(Xw_train_ids, Xw_train_lvls, Xl_train_ids, Xl_train_lvls, y_train)
val_ds   = MatchDataset(Xw_val_ids, Xw_val_lvls, Xl_val_ids, Xl_val_lvls, y_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Batches train:", len(train_loader), "val:", len(val_loader))


# %%
# Cell 6 - Model definition

class CardMatchNet(nn.Module):
    def __init__(self, vocab_size, max_level, id_emb_dim=32, lvl_emb_dim=8, hidden_dims=[128,64], dropout=0.2):
        super().__init__()
        self.id_embed = nn.Embedding(vocab_size, id_emb_dim, padding_idx=None)
        # level values range from 0..max_level, embed (add +1 if you want to reserve idx 0)
        self.lvl_embed = nn.Embedding(max_level + 1, lvl_emb_dim)
        per_card_emb = id_emb_dim + lvl_emb_dim

        # We'll pool across the 8 cards with avg and max pooling
        # So player vector size = per_card_emb * 2 (avg+max)
        player_vec_dim = per_card_emb * 2

        # final input: concat player1 + player2 => 2 * player_vec_dim
        mlp_input = player_vec_dim * 2

        mlp_layers = []
        inp = mlp_input
        for h in hidden_dims:
            mlp_layers.append(nn.Linear(inp, h))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            inp = h
        mlp_layers.append(nn.Linear(inp, 1))  # output logit
        self.mlp = nn.Sequential(*mlp_layers)

    def _player_repr(self, ids, lvls):
        # ids: (B,8), lvls: (B,8)
        id_e = self.id_embed(ids)    # (B,8,id_emb)
        lvl_e = self.lvl_embed(lvls) # (B,8,lvl_emb)
        card_e = torch.cat([id_e, lvl_e], dim=-1)  # (B,8,per_card)
        # pooling
        avg = card_e.mean(dim=1)     # (B, per_card)
        mx  = card_e.max(dim=1).values  # (B, per_card)
        return torch.cat([avg, mx], dim=-1)  # (B, per_card*2)

    def forward(self, p1_ids, p1_lvls, p2_ids, p2_lvls):
        # ensure long tensors
        p1_repr = self._player_repr(p1_ids, p1_lvls)
        p2_repr = self._player_repr(p2_ids, p2_lvls)
        x = torch.cat([p1_repr, p2_repr], dim=1)
        logit = self.mlp(x).squeeze(1)  # (B,)
        return logit

# instantiate
model = CardMatchNet(vocab_size=vocab_size, max_level=max_level,
                     id_emb_dim=ID_EMBED_DIM, lvl_emb_dim=LVL_EMBED_DIM,
                     hidden_dims=HIDDEN_DIMS, dropout=DROPOUT).to(device)
print(model)


# %%
# Cell 7 - Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# Use BCEWithLogitsLoss and pass pos_weight for imbalance (pos_weight = weight_for_positive_class)
# compute pos_weight = class_weights[1] / class_weights[0] if using sklearn balanced weights
# but easier: compute class counts
pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
pos_weight = torch.tensor([neg / (pos + 1e-9)], dtype=torch.float32, device=device)  # >1 -> upweight positives
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

best_val_auc = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []
    for batch in train_loader:
        p1_ids_b, p1_lvls_b, p2_ids_b, p2_lvls_b, y_b = batch
        p1_ids_b = p1_ids_b.to(device)
        p1_lvls_b = p1_lvls_b.to(device)
        p2_ids_b = p2_ids_b.to(device)
        p2_lvls_b = p2_lvls_b.to(device)
        y_b = y_b.to(device)

        optimizer.zero_grad()
        logits = model(p1_ids_b, p1_lvls_b, p2_ids_b, p2_lvls_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # Validation
    model.eval()
    val_logits_all = []
    val_y_all = []
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            p1_ids_b, p1_lvls_b, p2_ids_b, p2_lvls_b, y_b = batch
            p1_ids_b = p1_ids_b.to(device)
            p1_lvls_b = p1_lvls_b.to(device)
            p2_ids_b = p2_ids_b.to(device)
            p2_lvls_b = p2_lvls_b.to(device)
            y_b = y_b.to(device)

            logits = model(p1_ids_b, p1_lvls_b, p2_ids_b, p2_lvls_b)
            loss = criterion(logits, y_b)
            val_losses.append(loss.item())

            val_logits_all.append(logits.cpu().numpy())
            val_y_all.append(y_b.cpu().numpy())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    val_logits_all = np.concatenate(val_logits_all)
    val_y_all = np.concatenate(val_y_all)
    val_probs = 1 / (1 + np.exp(-val_logits_all))  # sigmoid
    val_auc = roc_auc_score(val_y_all, val_probs)
    val_pred = (val_probs >= 0.5).astype(int)
    val_acc = accuracy_score(val_y_all, val_pred)

    print(f"Epoch {epoch:02d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_auc={val_auc:.4f}  val_acc={val_acc:.4f}")

    # Save best model by AUC
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save({
            "model_state": model.state_dict(),
            "vocab_size": vocab_size,
            "max_level": max_level,
            "id_emb_dim": ID_EMBED_DIM,
            "lvl_emb_dim": LVL_EMBED_DIM,
            "hidden_dims": HIDDEN_DIMS
        }, os.path.join(MODEL_DIR, "best_model.pt"))
        print("  Saved best model (AUC improved).")


# %%
# Cell 8 - Load model and prediction helper

ckpt = torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=device)
# instantiate a model with same hyperparams (from our variables)
loaded_model = CardMatchNet(vocab_size=ckpt.get("vocab_size", vocab_size),
                            max_level=ckpt.get("max_level", max_level),
                            id_emb_dim=ID_EMBED_DIM,
                            lvl_emb_dim=LVL_EMBED_DIM,
                            hidden_dims=HIDDEN_DIMS, dropout=DROPOUT).to(device)
loaded_model.load_state_dict(ckpt["model_state"])
loaded_model.eval()

def predict_match(p1_ids, p1_lvls, p2_ids, p2_lvls, model=loaded_model):
    """
    p?_ids and p?_lvls are sequences/lists/arrays length 8 each (ints).
    Returns probability that player1 wins (float in [0,1]).
    """
    model.eval()
    with torch.no_grad():
        p1_ids_t = torch.tensor([p1_ids], dtype=torch.long, device=device)
        p1_lvls_t = torch.tensor([p1_lvls], dtype=torch.long, device=device)
        p2_ids_t = torch.tensor([p2_ids], dtype=torch.long, device=device)
        p2_lvls_t = torch.tensor([p2_lvls], dtype=torch.long, device=device)
        logit = model(p1_ids_t, p1_lvls_t, p2_ids_t, p2_lvls_t)
        prob = torch.sigmoid(logit).item()
    return prob

# quick test on a validation example
example_idx = 0
prob = predict_match(Xw_val_ids[example_idx], Xw_val_lvls[example_idx],
                     Xl_val_ids[example_idx], Xl_val_lvls[example_idx])
print("Pred prob player1 wins (example):", prob, "true label:", int(y_val[example_idx]))



