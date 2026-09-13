"""OD-grouped validation and label-preserving landmark oversampling."""
import numpy as np
import torch


def unordered_pair_keys(queries):
    q = np.asarray(queries)
    if q.ndim != 2 or q.shape[1] < 3:
        raise ValueError('Expected rows u,v,d_uv[,d_vu]')
    raw_ids = np.asarray(q[:,:2], dtype=np.float64)
    if not np.all(np.isfinite(raw_ids)) or not np.all(raw_ids == np.floor(raw_ids)) or np.any(raw_ids < 0):
        raise ValueError('Node IDs must be finite nonnegative integers')
    ids = raw_ids.astype(np.int64)
    return np.sort(ids,axis=1)


def _structured_keys(q):
    keys = np.ascontiguousarray(unordered_pair_keys(q))
    return keys.view(np.dtype([('u',np.int64),('v',np.int64)])).ravel()


def assert_disjoint_od(*parts):
    unique = [np.unique(_structured_keys(q)) for q in parts]
    for i in range(len(unique)):
        for j in range(i+1,len(unique)):
            overlap = np.intersect1d(unique[i],unique[j])
            if len(overlap):
                raise ValueError(f'OD leakage: splits {i}/{j} share {len(overlap)} unordered pairs')


def split_training_pairs(queries, validation_fraction=.1, seed=42):
    """Split TRAIN only; opposite orientations and duplicates remain grouped."""
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between 0 and 1')
    q = np.asarray(queries)
    keys, groups = np.unique(_structured_keys(q), return_inverse=True)
    if len(keys)<2:
        raise ValueError('At least two unordered pairs are needed')
    order = np.random.default_rng(seed).permutation(len(keys))
    nval = min(len(keys)-1,max(1,int(round(validation_fraction*len(keys)))))
    is_val_group = np.zeros(len(keys),dtype=bool)
    is_val_group[order[:nval]] = True
    mask = is_val_group[groups]
    train,val = q[~mask].copy(),q[mask].copy()
    assert_disjoint_od(train,val)
    return train,val


def resample_landmark_rows(i,j,y,landmarks,target_ratio):
    """Oversample EXISTING valid triples within a batch; never relabel endpoints.
    If a batch has no landmark row it is left unchanged. Thus target_ratio is
    a within-batch target, not a claim of global coverage or exact 60% sampling.
    Inputs are not modified, and every output triple is copied from an input row.
    """
    if not 0 <= target_ratio <= 1:
        raise ValueError('target_ratio must be in [0,1]')
    if len(i)!=len(j) or len(i)!=len(y):
        raise ValueError('Mismatched endpoint/label lengths')
    if not landmarks or len(i)==0:
        return i,j,y
    lm = torch.as_tensor(sorted(landmarks),device=i.device,dtype=i.dtype)
    mask = torch.isin(i,lm) | torch.isin(j,lm)
    selected = torch.where(mask)[0]
    normal = torch.where(~mask)[0]
    deficit = max(0,int(round(len(i)*target_ratio))-int(mask.sum()))
    if deficit==0 or len(selected)==0 or len(normal)==0:
        return i,j,y
    count = min(deficit,len(normal))
    dst = normal[torch.randperm(len(normal),device=i.device)[:count]]
    src = selected[torch.randint(len(selected),(count,),device=i.device)]
    order = torch.arange(len(i),device=i.device)
    order[dst] = src
    return i[order],j[order],y[order]
