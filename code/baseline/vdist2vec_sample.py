import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DATA_PATH = os.path.join("data", "chengdu_directed_shortest_distance_matrix.npy")

def get_batch_indices_from_list(idxs, batch_size, shuffle=True):
    idxs = np.array(idxs)
    if shuffle:
        np.random.shuffle(idxs)
    for i in range(0, len(idxs), batch_size):
        yield idxs[i:i+batch_size]

def batch_to_tensor(batch, sdm_norm, device):
    n = sdm_norm.shape[0]
    x1 = np.eye(n)[batch[:, 0]]
    x2 = np.eye(n)[batch[:, 1]]
    X = np.concatenate([x1, x2], axis=1)
    y = sdm_norm[batch[:, 0], batch[:, 1]].reshape(-1, 1).astype(np.float32)
    X = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(y).float().to(device)
    return X, y

def build_and_run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sdm = np.load(DATA_PATH)
    max_len = np.max(sdm)
    if max_len == 0:
        raise ValueError("Distance matrix max is zero.")
    sdm_norm = sdm / max_len
    n = sdm.shape[0]
    # 构建所有合法节点对索引
    all_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    all_pairs = np.array(all_pairs)
    np.random.seed(42)
    perm = np.random.permutation(len(all_pairs))
    train_size = int(len(all_pairs) * 0.9)
    train_pairs = all_pairs[perm[:train_size]]
    # test_pairs = all_pairs[perm[train_size:]]  # 不再用于评估

    # 定义模型
    class VDist2Vec(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.fc1 = nn.Linear(2 * n, max(1, int(n * 0.2)))
            self.fc2 = nn.Linear(max(1, int(n * 0.2)), 100)
            self.fc3 = nn.Linear(100, 20)
            self.out = nn.Linear(20, 1)
            self.act = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            x = self.act(self.fc1(x))
            x = self.act(self.fc2(x))
            x = self.act(self.fc3(x))
            x = self.sigmoid(self.out(x))
            return x

    model = VDist2Vec(n).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    batch_size = 1024
    epochs = 20

    # 训练
    for epoch in range(epochs):
        avg_loss = 0.0
        total = 0
        for batch in get_batch_indices_from_list(train_pairs, batch_size, shuffle=True):
            Xb, yb = batch_to_tensor(batch, sdm_norm, device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * Xb.size(0)
            total += Xb.size(0)
        avg_loss /= total
        print(f"Epoch {epoch+1:03d} cost={avg_loss:.9f}")
    print("Optimization Finished!")

    # 保存和加载模型
    model_path = "param/baseline/vdist2vec_model.pth"
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
            Xb, yb = batch_to_tensor(batch, sdm_norm, device)
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

    # 存储结果到log/vdist2vec_sample.txt
    result_file = "log/vdist2vec_results_sample.txt"
    with open(result_file, "w") as f:
        f.write(f"Eval sample size: {eval_sample_size}\n")
        f.write(f"mean square error: {mse:.2f}\n")
        f.write(f"mean absolute error: {mabe:.5f}\n")
        f.write(f"max absolute error: {maxae:.4f}\n")
        f.write(f"min absolute error: {minae:.1f}\n")
        f.write(f"mean relative error: {mre:.8f}\n")
        f.write(f"max relative error: {maxre:.8f}\n")
        f.write(f"min relative error: {minre:.1f}\n")

if __name__ == "__main__":
    build_and_run()