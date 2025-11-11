import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DATA_PATH = os.path.join("data", "chengdu_directed_shortest_distance_matrix.npy")
NODE_PATH = os.path.join("data", "chengdu_node-mod.txt")
EDGE_PATH = os.path.join("data", "chengdu_link-mod.txt")

def load_graph(n):
    # 构建邻接矩阵（有向图）
    adj = np.zeros((n, n), dtype=np.float32)
    with open(EDGE_PATH, encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            arr = line.strip().split(",")
            if len(arr) != 3: continue
            u, v, _ = arr
            u, v = int(u), int(v)
            adj[u, v] = 1.0
    return adj

def get_node_features(adj):
    # 节点特征：度数（入度、出度），可扩展
    in_deg = adj.sum(axis=0, keepdims=True).T  # [n,1]
    out_deg = adj.sum(axis=1, keepdims=True)   # [n,1]
    features = np.concatenate([in_deg, out_deg], axis=1)  # [n,2]
    return features.astype(np.float32)

def get_batch_indices(n, batch_size, shuffle=True):
    idxs = [(i, j) for i in range(n) for j in range(n) if i != j]
    idxs = np.array(idxs)
    if shuffle:
        np.random.shuffle(idxs)
    for i in range(0, len(idxs), batch_size):
        yield idxs[i:i+batch_size]

def batch_to_tensor(batch, n, sdm_norm, node_feats, device):
    # batch: [B,2]，node_feats: [n,feat_dim]
    u_idx = batch[:, 0]
    v_idx = batch[:, 1]
    # 节点one-hot
    x1 = np.eye(n)[u_idx]
    x2 = np.eye(n)[v_idx]
    # 节点特征
    f1 = node_feats[u_idx]
    f2 = node_feats[v_idx]
    # 拼接one-hot和节点特征
    node1 = np.concatenate([x1, f1], axis=1)
    node2 = np.concatenate([x2, f2], axis=1)
    X = np.stack([node1, node2], axis=1)  # [B,2,feat]
    y = sdm_norm[u_idx, v_idx].reshape(-1, 1).astype(np.float32)
    X = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(y).float().to(device)
    return X, y

class SARNNet(nn.Module):
    def __init__(self, n, node_feat_dim, emb_dim=64, attn_heads=4):
        super().__init__()
        self.n = n
        self.input_dim = n + node_feat_dim
        self.emb_dim = emb_dim
        self.node_emb = nn.Linear(self.input_dim, emb_dim)
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=attn_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(2 * emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x, adj=None):
        # x: [B,2,input_dim]
        x = self.node_emb(x)  # [B,2,emb_dim]
        # 自注意力
        attn_out, _ = self.attn(x, x, x)  # [B,2,emb_dim]
        out = torch.cat([attn_out[:, 0, :], attn_out[:, 1, :]], dim=1)  # [B,2*emb_dim]
        out = self.mlp(out)
        return out

def build_and_run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sdm = np.load(DATA_PATH)
    max_len = np.max(sdm)
    if max_len == 0:
        raise ValueError("Distance matrix max is zero.")
    sdm_norm = sdm / max_len
    n = sdm.shape[0]

    # 构建邻接矩阵和节点特征
    adj = load_graph(n)
    node_feats = get_node_features(adj)  # shape: [n,2]
    node_feat_dim = node_feats.shape[1]

    model = SARNNet(n, node_feat_dim=node_feat_dim, emb_dim=64, attn_heads=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    batch_size = 1024
    epochs = 20

    # 构建所有合法节点对索引
    all_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    all_pairs = np.array(all_pairs)
    np.random.seed(42)
    perm = np.random.permutation(len(all_pairs))
    train_size = int(len(all_pairs) * 0.9)
    train_pairs = all_pairs[perm[:train_size]]

    # 早停参数
    early_stop_patience = 5
    best_loss = float('inf')
    bad_counter = 0

    # 训练
    for epoch in range(epochs):
        avg_loss = 0.0
        total = 0
        for batch in get_batch_indices_from_list(train_pairs, batch_size, shuffle=True):
            Xb, yb = batch_to_tensor(batch, n, sdm_norm, node_feats, device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * Xb.size(0)
            total += Xb.size(0)
        avg_loss /= total
        print(f"Epoch {epoch+1:03d} cost={avg_loss:.9f}")

        # 早停判断
        if avg_loss < best_loss:
            best_loss = avg_loss
            bad_counter = 0
        else:
            bad_counter += 1
            if bad_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    print("Optimization Finished!")

    # 保存和加载模型
    model_path = "param/baseline/SARN_model.pth"
    torch.save(model.state_dict(), model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    # 评估：从全集随机采样
    eval_sample_size = min(10000, len(all_pairs))
    np.random.seed(123)
    sample_idxs = np.random.choice(len(all_pairs), eval_sample_size, replace=False)
    sampled_eval_pairs = all_pairs[sample_idxs]

    preds = []
    reals = []
    with torch.no_grad():
        for batch in get_batch_indices_from_list(sampled_eval_pairs, batch_size, shuffle=False):
            Xb, yb = batch_to_tensor(batch, n, sdm_norm, node_feats, device)
            pred = model(Xb).cpu().numpy().flatten() * max_len
            real = yb.cpu().numpy().flatten() * max_len
            preds.append(pred)
            reals.append(real)
    preds = np.concatenate(preds)
    reals = np.concatenate(reals)

    abe = np.abs(reals - preds)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(reals > 0, abe / reals, 0.0)

    mse = np.mean((abe ** 2))
    mabe = np.mean(abe)
    maxae = np.max(abe) if abe.size else 0.0
    minae = np.min(abe) if abe.size else 0.0
    mre = np.mean(rel)
    maxre = np.max(rel) if rel.size else 0.0
    minre = np.min(rel) if rel.size else 0.0

    print(f"Eval sample size: {eval_sample_size}")
    print("mean square error:", mse)
    print("mean absolute error:", mabe)
    print("max absolute error:", maxae)
    print("min absolute error:", minae)
    print("mean relative error:", mre)
    print("max relative error:", maxre)
    print("min relative error:", minre)

    # 存储结果到log/SARN_sample.txt
    result_file = "log/SARN_sample.txt"
    with open(result_file, 'w') as f:
        f.write(f"Eval sample size: {eval_sample_size}\n")
        f.write(f"mean square error: {mse:.2f}\n")
        f.write(f"mean absolute error: {mabe:.5f}\n")
        f.write(f"max absolute error: {maxae:.4f}\n")
        f.write(f"min absolute error: {minae:.1f}\n")
        f.write(f"mean relative error: {mre:.8f}\n")
        f.write(f"max relative error: {maxre:.8f}\n")
        f.write(f"min relative error: {minre:.1f}\n")

# 新增：支持从指定索引列表分批
def get_batch_indices_from_list(idxs, batch_size, shuffle=True):
    idxs = np.array(idxs)
    if shuffle:
        np.random.shuffle(idxs)
    for i in range(0, len(idxs), batch_size):
        yield idxs[i:i+batch_size]

if __name__ == "__main__":
    build_and_run()
