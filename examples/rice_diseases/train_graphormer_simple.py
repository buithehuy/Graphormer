"""
Mini Graphormer - Standalone Training Script for Rice Diseases Dataset
=======================================================================
Chạy trực tiếp trên dữ liệu .pt đã được xử lý.
Không cần fairseq, chỉ cần PyTorch + PyG.

Data format (từ rice_image_to_graph.py):
  - x          : (num_nodes, 3)   - RGB node features (float)
  - edge_index : (2, num_edges)   - edge connectivity (long)
  - edge_attr  : (num_edges,)     - edge type bin 0..9 (long)
  - y          : (1,)             - class label 0..3 (long)

Classes: BrownSpot=0, Healthy=1, Hispa=2, LeafBlast=3

Usage (Colab / terminal):
  python train_graphormer_simple.py --data_dir /content/Graphormer/examples/rice_diseases/rice_diseases_graphs/processed
  python train_graphormer_simple.py --data_dir ... --epochs 50 --batch_size 32 --lr 3e-4
"""

import os
import glob
import json
import random
import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data, Batch

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Dataset wrapper
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast']


class PTGraphDataset(Dataset):
    """Loads pre-processed .pt graph files from a directory."""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("data_*.pt"),
                            key=lambda p: int(p.stem.split('_')[1]))
        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No data_*.pt files found in: {data_dir}\n"
                "Make sure the processed directory is correct."
            )
        print(f"[Dataset] Found {len(self.files)} graph files in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        return data


def collate_fn(batch):
    """Collate a list of PyG Data objects into a padded batch for Graphormer."""
    return Batch.from_data_list(batch)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Graphormer Building Blocks
# ─────────────────────────────────────────────────────────────────────────────

class GraphormerAttention(nn.Module):
    """Multi-head attention with spatial bias from edge information."""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, attn_bias=None, key_padding_mask=None):
        """
        x              : (B, T, C)
        attn_bias      : (B, num_heads, T, T)  – spatial / edge bias
        key_padding_mask: (B, T) bool, True = padding
        """
        B, T, C = x.shape
        H = self.num_heads
        D = self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)   # (B, H, T, D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / self.scale           # (B, H, T, T)

        if attn_bias is not None:
            attn = attn + attn_bias

        if key_padding_mask is not None:
            # mask shape (B, T) -> (B, 1, 1, T)
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class GraphormerLayer(nn.Module):
    """Single Graphormer encoder layer."""

    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.attn = GraphormerAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_bias=None, key_padding_mask=None):
        # Pre-LN (more stable training)
        x = x + self.attn(self.norm1(x), attn_bias, key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class MiniGraphormer(nn.Module):
    """
    Lightweight Graphormer for graph-level classification.

    Key simplifications vs. the full model:
    • Node features projected via linear layer (not learnable embeddings)
    • Edge bias computed from edge_attr embeddings, scattered into attention
    • Virtual "graph token" at position 0 used for readout
    """

    def __init__(
        self,
        node_in_dim=3,        # RGB
        edge_vocab=10,        # bin 0..9
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        ffn_dim=256,
        num_classes=4,
        max_nodes=200,        # expected max superpixels (~75 + slack)
        dropout=0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_nodes = max_nodes

        # Node encoding
        self.node_encoder = nn.Linear(node_in_dim, embed_dim)

        # Virtual graph token (CLS-style)
        self.graph_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Edge bias (per head): edge_attr -> bias added to attention
        self.edge_bias = nn.Embedding(edge_vocab + 1, num_heads, padding_idx=0)

        # Degree bias for centrality encoding
        self.in_degree_encoder  = nn.Embedding(512, embed_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(512, embed_dim, padding_idx=0)

        # Position bias for the virtual node (single scalar per head)
        self.vnode_bias = nn.Parameter(torch.randn(num_heads))

        # Transformer layers
        self.layers = nn.ModuleList([
            GraphormerLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def build_attn_bias(self, batch, batch_size, max_len, device):
        """
        Construct attention bias tensor (B, H, T+1, T+1) where T = max_len (real nodes).
        +1 for the virtual graph token at index 0.
        """
        H = self.layers[0].attn.num_heads
        seq_len = max_len + 1   # +1 for vnode
        bias = torch.zeros(batch_size, H, seq_len, seq_len, device=device)

        # Virtual node bias (constant bias from/to vnode positions)
        bias[:, :, 0, :] = self.vnode_bias.view(1, H, 1)
        bias[:, :, :, 0] = self.vnode_bias.view(1, H, 1)

        # Edge bias: scatter edge embeddings into attention matrix
        # Each graph in the batch has its own node offset
        batch_idx   = batch.batch          # (total_nodes,)
        edge_index  = batch.edge_index     # (2, total_edges)
        edge_attr   = batch.edge_attr      # (total_edges,)

        if edge_attr is not None and edge_attr.numel() > 0:
            # Clamp edge_attr to valid range
            edge_attr_clamped = edge_attr.clamp(0, 9) + 1  # shift: 0 is padding

            # edge_bias: (total_edges, H)
            eb = self.edge_bias(edge_attr_clamped)   # (E, H)

            # Compute per-graph node offsets
            # batch_ptr[g] = first node index of graph g in the batch
            batch_ptr = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
            counts = torch.bincount(batch_idx, minlength=batch_size)
            batch_ptr[1:] = counts.cumsum(0)

            # For each edge, figure out which graph it belongs to
            src_global = edge_index[0]  # (E,)
            dst_global = edge_index[1]
            graph_id   = batch_idx[src_global]  # (E,)

            # Local node index within its graph (+1 for vnode offset)
            src_local = src_global - batch_ptr[graph_id] + 1   # (E,)
            dst_local = dst_global - batch_ptr[graph_id] + 1

            # Scatter into bias tensor
            # We iterate but limit to feasible nodes
            valid = (src_local < seq_len) & (dst_local < seq_len)
            g  = graph_id[valid]
            sl = src_local[valid]
            dl = dst_local[valid]
            ev = eb[valid]   # (valid_E, H)

            # Vectorised scatter_add over (B, T, T) for each head
            flat_idx = g * seq_len * seq_len + sl * seq_len + dl  # (valid_E,)
            for h in range(H):
                b_flat = bias[:, h, :, :].reshape(-1)
                b_flat.scatter_add_(0, flat_idx, ev[:, h])
                bias[:, h, :, :] = b_flat.view(batch_size, seq_len, seq_len)

        return bias

    def forward(self, batch):
        device = batch.x.device

        # ── 1. Pad graphs to same number of nodes ──────────────────────────
        batch_ids   = batch.batch       # (total_nodes,)
        batch_size  = batch_ids.max().item() + 1
        node_feats  = batch.x           # (total_nodes, 3)
        in_deg  = torch.zeros(node_feats.size(0), dtype=torch.long, device=device)
        out_deg = torch.zeros(node_feats.size(0), dtype=torch.long, device=device)

        if batch.edge_index is not None and batch.edge_index.numel() > 0:
            src = batch.edge_index[0]
            dst = batch.edge_index[1]
            in_deg.scatter_add_(0, dst, torch.ones_like(dst))
            out_deg.scatter_add_(0, src, torch.ones_like(src))
        in_deg  = in_deg.clamp(0, 511)
        out_deg = out_deg.clamp(0, 511)

        # Count nodes per graph
        counts = torch.bincount(batch_ids, minlength=batch_size)  # (B,)
        max_len = counts.max().item()

        # Build padded node tensor (B, max_len, C)
        x_enc = self.node_encoder(node_feats)                           # (N, C)
        x_enc = x_enc + self.in_degree_encoder(in_deg)
        x_enc = x_enc + self.out_degree_encoder(out_deg)

        x_pad = torch.zeros(batch_size, max_len, self.embed_dim, device=device)
        pad_mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)  # True = pad

        offset = 0
        for g in range(batch_size):
            n = counts[g].item()
            x_pad[g, :n, :] = x_enc[offset:offset + n]
            pad_mask[g, :n] = False
            offset += n

        # ── 2. Prepend virtual graph token ─────────────────────────────────
        vnode = self.graph_token.expand(batch_size, -1, -1)  # (B, 1, C)
        x_full = torch.cat([vnode, x_pad], dim=1)            # (B, 1+max_len, C)

        # Padding mask for full sequence (vnode is never masked)
        vnode_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        full_mask = torch.cat([vnode_mask, pad_mask], dim=1)  # (B, 1+max_len)

        # ── 3. Attention bias from edge features ───────────────────────────
        attn_bias = self.build_attn_bias(batch, batch_size, max_len, device)

        # ── 4. Transformer layers ──────────────────────────────────────────
        x = x_full
        for layer in self.layers:
            x = layer(x, attn_bias=attn_bias, key_padding_mask=full_mask)
        x = self.norm(x)

        # ── 5. Readout from virtual graph token ────────────────────────────
        graph_repr = x[:, 0, :]   # (B, C)
        logits = self.classifier(graph_repr)

        return logits


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Training & Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    total_loss = total_correct = total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            labels = batch.y.view(-1)
            logits = model(batch)
            loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(dim=-1)
            total_loss    += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total         += labels.size(0)
    return total_loss / total, total_correct / total


def train_one_epoch(model, loader, optimizer, device, scheduler=None):
    model.train()
    total_loss = total_correct = total = 0
    for batch in loader:
        batch = batch.to(device)
        labels = batch.y.view(-1)
        logits = model(batch)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        preds = logits.argmax(dim=-1)
        total_loss    += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total         += labels.size(0)
    return total_loss / total, total_correct / total


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Mini Graphormer - Rice Diseases")
    p.add_argument("--data_dir", type=str,
                   default="/content/Graphormer/examples/rice_diseases/rice_diseases_graphs/processed",
                   help="Directory with processed data_*.pt files")
    p.add_argument("--split_file", type=str, default=None,
                   help="Path to split_indices.pt (auto-detected if in data_dir)")
    # Model
    p.add_argument("--embed_dim",  type=int, default=128)
    p.add_argument("--num_heads",  type=int, default=4)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--ffn_dim",    type=int, default=256)
    p.add_argument("--dropout",    type=float, default=0.1)
    # Training
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--weight_decay",type=float, default=1e-2)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--num_workers", type=int,   default=2)
    p.add_argument("--save_dir",    type=str,   default="./ckpts_mini_graphormer")
    p.add_argument("--no_save",     action="store_true", help="Don't save checkpoints")
    p.add_argument("--inspect_only",action="store_true", help="Just inspect data, don't train")
    return p.parse_args()


def inspect_data(data_dir):
    """Quick inspection of the first few graphs."""
    print("\n" + "=" * 60)
    print("DATA INSPECTION")
    print("=" * 60)

    files = sorted(Path(data_dir).glob("data_*.pt"),
                   key=lambda p: int(p.stem.split('_')[1]))
    print(f"Total .pt files : {len(files)}")

    label_counts = [0] * 4
    for i, f in enumerate(files[:5]):
        data = torch.load(f, weights_only=False)
        label = data.y.item() if data.y is not None else -1
        print(f"\n  [{f.name}]")
        print(f"    x shape      : {data.x.shape}")
        print(f"    edge_index   : {data.edge_index.shape}")
        print(f"    edge_attr    : {data.edge_attr.shape}  range=[{data.edge_attr.min()}, {data.edge_attr.max()}]")
        print(f"    label (y)    : {label} ({CLASS_NAMES[label] if 0 <= label < 4 else '?'})")

    # Count labels
    print("\nCounting labels across all files ...")
    for f in files:
        data = torch.load(f, weights_only=False)
        if data.y is not None:
            label_counts[data.y.item()] += 1
    print("\nClass distribution:")
    for i, (name, count) in enumerate(zip(CLASS_NAMES, label_counts)):
        print(f"  {i} {name:12s}: {count:5d}")
    print("=" * 60)

    # Check splits
    split_path = Path(data_dir) / "split_indices.pt"
    if split_path.exists():
        splits = torch.load(split_path, weights_only=False)
        print(f"\nSplit file found: {split_path}")
        for k, v in splits.items():
            print(f"  {k}: {len(v)} samples")
    else:
        print("\nNo split_indices.pt found — will use random 70/15/15 split.")

    # Check metadata
    meta_path = Path(data_dir) / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"\nMetadata: {meta_path}")
        for k, v in meta.items():
            if k not in ('labels', 'image_paths'):
                print(f"  {k}: {v}")


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Mini Graphormer — Rice Diseases")
    print(f"{'='*60}")
    print(f"  Device     : {device}")
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Embed dim  : {args.embed_dim}")
    print(f"  Layers     : {args.num_layers}")
    print(f"  Heads      : {args.num_heads}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"{'='*60}\n")

    # ── Data Inspection ────────────────────────────────────────────────────
    inspect_data(args.data_dir)

    if args.inspect_only:
        print("\n[--inspect_only] Exiting.")
        return

    # ── Dataset & Splits ───────────────────────────────────────────────────
    full_dataset = PTGraphDataset(args.data_dir)
    n = len(full_dataset)

    split_path = args.split_file or str(Path(args.data_dir) / "split_indices.pt")
    if os.path.exists(split_path):
        print(f"\nUsing existing splits from: {split_path}")
        splits = torch.load(split_path, weights_only=False)
        train_idx = splits['train_idx'].tolist()
        val_idx   = splits['val_idx'].tolist()
        test_idx  = splits['test_idx'].tolist()

        from torch.utils.data import Subset
        train_set = Subset(full_dataset, train_idx)
        val_set   = Subset(full_dataset, val_idx)
        test_set  = Subset(full_dataset, test_idx)
    else:
        print("\nCreating random 70/15/15 split ...")
        n_train = int(0.70 * n)
        n_val   = int(0.15 * n)
        n_test  = n - n_train - n_val
        train_set, val_set, test_set = random_split(
            full_dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(args.seed)
        )

    print(f"  Train: {len(train_set):5d} | Val: {len(val_set):4d} | Test: {len(test_set):4d}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn)

    # ── Model ──────────────────────────────────────────────────────────────
    model = MiniGraphormer(
        node_in_dim=3,
        edge_vocab=10,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        num_classes=4,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters : {n_params:,}")

    # ── Optimiser & Scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(1000, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=warmup_steps / total_steps, anneal_strategy='cos'
    )

    # ── Training Loop ──────────────────────────────────────────────────────
    if not args.no_save:
        os.makedirs(args.save_dir, exist_ok=True)

    best_val_acc = 0.0
    print(f"\n{'─'*60}")
    print(f"  {'Epoch':>5}  {'TrainLoss':>9}  {'TrainAcc':>8}  {'ValLoss':>8}  {'ValAcc':>7}  {'Time':>6}")
    print(f"{'─'*60}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, scheduler)
        val_loss, val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        print(f"  {epoch:5d}  {tr_loss:9.4f}  {tr_acc*100:7.2f}%  {val_loss:8.4f}  {val_acc*100:6.2f}%  {elapsed:5.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if not args.no_save:
                ckpt = dict(epoch=epoch, model=model.state_dict(),
                            val_acc=val_acc, args=vars(args))
                torch.save(ckpt, os.path.join(args.save_dir, "best_model.pt"))
                print(f"         ✓ Saved best model (val_acc={val_acc*100:.2f}%)")

    # ── Test Evaluation ────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Final Test Evaluation")
    print(f"{'─'*60}")

    # Load best checkpoint if available
    best_path = os.path.join(args.save_dir, "best_model.pt")
    if not args.no_save and os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        print(f"  Loaded best model from epoch {ckpt['epoch']}")

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"  Test Loss : {test_loss:.4f}")
    print(f"  Test Acc  : {test_acc*100:.2f}%")
    print(f"  Best Val  : {best_val_acc*100:.2f}%")

    # Per-class accuracy
    model.eval()
    class_correct = [0] * 4
    class_total   = [0] * 4
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            labels = batch.y.view(-1)
            preds  = model(batch).argmax(dim=-1)
            for c in range(4):
                mask = labels == c
                class_correct[c] += (preds[mask] == c).sum().item()
                class_total[c]   += mask.sum().item()

    print(f"\n  Per-class accuracy:")
    for c, name in enumerate(CLASS_NAMES):
        if class_total[c] > 0:
            acc = class_correct[c] / class_total[c] * 100
            print(f"    {c} {name:12s}: {acc:6.2f}%  ({class_correct[c]}/{class_total[c]})")

    print(f"\n{'='*60}")
    print("  Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
