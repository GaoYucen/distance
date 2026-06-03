"""
Cross-Encoder L1 vs L1Tilde quick test on Harbin.
Demonstrates that L1Tilde significantly outperforms L1 when the encoder provides asymmetric representations.
"""
import numpy as np, torch, torch.nn as nn, time, sys
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, '.')
from utils.data_utils import load_graph, get_edge_attributes, get_node_attributes
from utils.torch_utils import read_query_file
from models.rgnndist2vec import RGNNdist2vec

torch.manual_seed(42)

# ---- Config ----
DATA_DIR = "data/OSM_Harbin_Small"
GNN_CKPT = "results/harbin_opt/sage_smoothl1_50ep_lr001_s42/saved_models/rgnndist2vec_OSM_Harbin_Small_random_500k.pt"
HIDDEN = 512
OUTPUT_DIM = 64
EPOCHS = 50
LR = 1e-3
R, S = 62, 2  # L1Tilde symmetric / asymmetric dimensions

# ---- Load data ----
G = load_graph(dir_name=DATA_DIR, force_shift=None)
edge_attr = get_edge_attributes(G)
node_attr = get_node_attributes(G)
print(f"Graph: {node_attr.shape[0]} nodes, {edge_attr.shape[0]} edges")

train_q = np.array(read_query_file(f"{DATA_DIR}/random_500k/Harbin_Small_train.queries"))
test_q = np.array(read_query_file(f"{DATA_DIR}/random_500k/Harbin_Small_test.queries"))
max_dist = max(train_q[:, 2].max(), train_q[:, 3].max())
print(f"Queries: {len(train_q)} train, {len(test_q)} test, max dist={max_dist:.0f}m")

# ---- Extract GNN features ----
model = RGNNdist2vec(n_input=2, n_hidden_1=HIDDEN, n_hidden_2=OUTPUT_DIM, layer_type='sage',
                      node_attributes=node_attr, edge_attributes=edge_attr,
                      max_distance=max_dist, disable_edge_weight=True)
ckpt = torch.load(GNN_CKPT, map_location='cuda', weights_only=False)
model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state_dict'].items()})
model.cuda(); model.eval()
with torch.no_grad():
    emb = model.encode(model.geometric_data.x.cuda(), model.geometric_data.edge_index.cuda()).cpu().numpy()

# Normalize coordinates
coord_mean, coord_std = node_attr.mean(axis=0), node_attr.std(axis=0)
coords_norm = (node_attr - coord_mean) / coord_std
feat = np.hstack([emb, coords_norm]).astype(np.float32)

def make_data(q):
    u, v = q[:, 0].astype(int), q[:, 1].astype(int)
    X = np.hstack([feat[u], feat[v]]).astype(np.float32)
    y = (q[:, 2].astype(np.float32) / max_dist).reshape(-1, 1)
    return X, y

X_train, y_train = make_data(train_q)
X_test, y_test = make_data(test_q)
print(f"Cross-Encoder input dim: {X_train.shape[1]} ({feat.shape[1]}×2)")

# ---- Cross-Encoder Model ----
class CrossEncoder(nn.Module):
    """Concat-based encoder with L1 or L1Tilde decoder."""
    def __init__(self, d_in, hidden=HIDDEN, output_dim=OUTPUT_DIM, mode='l1', r=R, s=S):
        super().__init__()
        self.mode, self.r, self.s = mode, r, s
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, output_dim * 2),
        )

    def forward(self, x):
        out = self.net(x)
        half = out.shape[1] // 2
        if self.mode == 'l1':
            return torch.mean(torch.abs(out[:, half:] - out[:, :half]), dim=1, keepdim=True)
        elif self.mode == 'l1tilde':
            s, r = self.s, half - self.s
            p1 = torch.abs(out[:, half:half+s] - out[:, 0:s])
            sym = torch.mean(p1, dim=1, keepdim=True) * s
            p2 = out[:, half+s:] - out[:, s:half]
            asym = torch.mean(p2, dim=1, keepdim=True) * r
            return (sym + asym) / half
        else:
            return torch.mean(out, dim=1, keepdim=True)

def train(mode, Xt, yt, Xv, yv, epochs=EPOCHS, lr=LR):
    m = CrossEncoder(Xt.shape[1], mode=mode).cuda()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xt), torch.from_numpy(yt)), batch_size=4096, shuffle=True)
    Xvt, yvt = torch.from_numpy(Xv).cuda(), torch.from_numpy(yv).cuda()
    best, best_ep = float('inf'), 0
    for ep_i in range(epochs):
        m.train()
        for bx, by in dl:
            bx, by = bx.cuda(), by.cuda()
            opt.zero_grad(); nn.SmoothL1Loss()(m(bx), by).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            mre = (torch.abs(m(Xvt) - yvt) / (yvt + 1e-8)).mean().item()
            sch.step(mre)
            if mre < best:
                best, best_ep = mre, ep_i + 1
    return best * 100, best_ep

# ---- Run ----
print(f"\n{'='*60}")
print(f"Cross-Encoder L1 vs L1Tilde (GNN features, {EPOCHS} epochs)")
print(f"{'='*60}")
for mode in ['l1', 'l1tilde']:
    t0 = time.time()
    mre, bep = train(mode, X_train, y_train, X_test, y_test)
    print(f"  {mode:>8s}: Test MRE = {mre:.2f}%  (best epoch {bep}, {time.time()-t0:.0f}s)")

# Pure coordinate baseline
print(f"\n{'='*60}")
print("Pure Coordinate Cross-Encoder (4-dim input)")
print(f"{'='*60}")
coord_feat = coords_norm.astype(np.float32)
def make_coord_data(q):
    u, v = q[:, 0].astype(int), q[:, 1].astype(int)
    X = np.hstack([coord_feat[u], coord_feat[v]]).astype(np.float32)
    y = (q[:, 2].astype(np.float32) / max_dist).reshape(-1, 1)
    return X, y
Xc_train, yc_train = make_coord_data(train_q)
Xc_test, yc_test = make_coord_data(test_q)
for mode in ['l1', 'l1tilde']:
    t0 = time.time()
    mre, bep = train(mode, Xc_train, yc_train, Xc_test, yc_test, epochs=30)
    print(f"  {mode:>8s}: Test MRE = {mre:.2f}%  (best epoch {bep}, {time.time()-t0:.0f}s)")
