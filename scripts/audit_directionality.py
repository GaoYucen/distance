"""Directional structure audit, not a learned-model accuracy benchmark.
The legacy Chengdu matrix is independently checked against integer-edge Dijkstra.
Both directions of each unordered OD pair are kept in the same split.
"""
from pathlib import Path
import argparse, hashlib, json, platform, subprocess, time
import numpy as np
import networkx as nx
from scipy import sparse
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.sparse.linalg import lsqr


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fit_potential(pairs, n_nodes):
    """Fit A_uv=(d_uv-d_vu)/2 by h[v]-h[u] on THESE pairs only.
    The minimum-norm LSQR solution fixes the additive gauge per component.
    Disconnected cross-component evaluation must be excluded, not interpreted.
    """
    pairs = np.asarray(pairs, dtype=np.float64)
    u, v = pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)
    a = (pairs[:, 2] - pairs[:, 3]) / 2
    rows = np.repeat(np.arange(len(u)), 2)
    cols = np.column_stack([u, v]).ravel()
    incidence = sparse.csr_matrix((np.tile([-1., 1.], len(u)), (rows, cols)),
                                  shape=(len(u), n_nodes))
    sol = lsqr(incidence, a, atol=1e-10, btol=1e-10, iter_lim=5000)
    adjacency = sparse.csr_matrix((np.ones(len(u)), (u, v)), shape=(n_nodes, n_nodes))
    _, components = connected_components(adjacency, directed=False)
    meta = dict(stop_code=int(sol[1]), iterations=int(sol[2]), residual_norm=float(sol[3]),
                normal_residual_norm=float(sol[7]), converged=int(sol[1]) in (0, 1, 2, 4, 5))
    return sol[0], components, meta


def report_slice(pairs, h):
    if len(pairs) == 0:
        return {'count': 0}
    a, b = pairs[:, 2], pairs[:, 3]
    sym, asym = (a+b)/2, (a-b)/2
    pred_asym = h[pairs[:, 1].astype(int)] - h[pairs[:, 0].astype(int)]
    resid = pred_asym - asym
    energy = float(np.mean(asym**2))
    resid_energy = float(np.mean(resid**2))
    pa, pb = sym+pred_asym, sym-pred_asym
    out = {
        'count': int(len(pairs)),
        'asymmetry_abs_diff_over_mean_quantiles': np.quantile(np.abs(a-b)/sym, [0, .5, .9, .95, .99, 1]).tolist(),
        'asymmetric_energy': energy,
        'potential_residual_energy': resid_energy,
        'potential_explained_energy_ratio_vs_zero': None if energy == 0 else 1-resid_energy/energy,
        'asymmetry_rmse': float(np.sqrt(energy)),
        'potential_residual_rmse': float(np.sqrt(resid_energy)),
        'oracle_symmetric_midpoint_mre_percent': float(100*np.mean(.5*(np.abs(sym-a)/a + np.abs(sym-b)/b))),
        'best_possible_symmetric_pairwise_mre_percent': float(100*np.mean(np.abs(a-b)/(2*np.maximum(a,b)))),
        'oracle_S_plus_train_fitted_potential_mre_percent': float(100*np.mean(.5*(np.abs(pa-a)/a + np.abs(pb-b)/b))),
        'oracle_S_plus_potential_negative_prediction_fraction': float(np.mean(np.concatenate([pa, pb]) < 0)),
    }
    return out


def diagnostic(pairs_train, pairs_val, pairs_test, n_nodes):
    h, comp, solver = fit_potential(pairs_train, n_nodes)
    if not solver['converged']:
        raise RuntimeError(f'Potential solver did not converge: {solver}')
    report = {'solver': solver, 'fit_scope': 'training unordered pairs only',
              'asymmetry_definition': 'abs(d_uv-d_vu) / ((d_uv+d_vu)/2)',
              'warning': 'Oracle symmetric component S uses ground truth: these are structural diagnostics, NOT learned-model MRE.',
              'energy_warning': 'Explained ratio measures squared directional difference, not fraction of queries and not fraction of total model error.',
              'split_metrics': {}}
    short_threshold = float(np.quantile((pairs_train[:,2]+pairs_train[:,3])/2, .25))
    report['short_threshold_train_q25_distance_units'] = short_threshold
    for name, p in [('train', pairs_train), ('validation', pairs_val), ('test', pairs_test)]:
        seen = comp[p[:,0].astype(int)] == comp[p[:,1].astype(int)]
        q = p[seen]
        sym = (q[:,2]+q[:,3])/2
        alpha = np.abs(q[:,2]-q[:,3])/sym
        report['split_metrics'][name] = {
            'input_count': int(len(p)), 'identifiable_pair_count': int(seen.sum()),
            'all': report_slice(q, h),
            'short_train_q25': report_slice(q[sym <= short_threshold], h),
            'high_asymmetry_ge_20pct': report_slice(q[alpha >= .2], h),
            'asymmetry_lt_1pct_fraction': float(np.mean(alpha < .01)),
            'asymmetry_ge_20pct_fraction': float(np.mean(alpha >= .2)),
        }
    return report, h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/audit')
    parser.add_argument('--output-dir', default='reports/audit-r1-20260913')
    parser.add_argument('--seed', type=int, default=20260913)
    parser.add_argument('--max-pairs', type=int, default=200000)
    args = parser.parse_args()
    start = time.perf_counter()
    data, out = Path(args.data_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    D = np.load(data/'chengdu_directed_shortest_distance_matrix.npy')
    n = len(D)
    coords = np.loadtxt(data/'chengdu_node-mod.txt', delimiter=',', skiprows=1)
    links = np.loadtxt(data/'chengdu_link-mod.txt', delimiter=',', skiprows=1)
    assert D.shape == (n,n) and np.array_equal(coords[:,0], np.arange(n))
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    for u,v,w in links:
        # Exactly the historical generator's convention: int(length), last duplicate wins.
        graph.add_edge(int(u), int(v), weight=int(w))
    matrix = nx.to_scipy_sparse_array(graph, nodelist=range(n), weight='weight', format='csr', dtype=np.float64)
    matrix = sparse.csr_matrix((matrix.data, matrix.indices.astype(np.int32), matrix.indptr.astype(np.int32)), shape=matrix.shape)
    exact = dijkstra(matrix, directed=True)
    match_finite = np.array_equal(np.isfinite(exact), np.isfinite(D))
    max_error = float(np.max(np.abs(exact[np.isfinite(exact)]-D[np.isfinite(exact)])))
    check = {'source_commit': 'e9fb84f261b75af490f9a11c833692f1af01c839',
             'nodes': n, 'directed_edges': graph.number_of_edges(),
             'strongly_connected': bool(nx.is_strongly_connected(graph)),
             'weight_convention': 'int(length) per legacy generate.py, last duplicate wins',
             'all_pairs_finite_mask_matches': match_finite,
             'all_pairs_max_abs_difference': max_error,
             'passed': bool(match_finite and max_error < 1e-8)}
    (out/'independent_label_check.json').write_text(json.dumps(check, indent=2)+'\n')
    print('LABEL_CHECK', json.dumps(check), flush=True)
    if not check['passed']:
        raise RuntimeError('Legacy labels failed independent graph audit; stopping before any model experiment')
    u,v = np.triu_indices(n, 1)
    good = np.isfinite(D[u,v]) & np.isfinite(D[v,u]) & (D[u,v]>0) & (D[v,u]>0)
    u,v = u[good],v[good]
    rng = np.random.default_rng(args.seed)
    take = rng.choice(len(u), min(args.max_pairs,len(u)), replace=False)
    u,v = u[take],v[take]
    q = np.column_stack([u,v,D[u,v],D[v,u]])
    t1,t2 = int(.8*len(q)), int(.9*len(q))
    train,val,test = q[:t1],q[t1:t2],q[t2:]
    report,h = diagnostic(train,val,test,n)
    # Diagnostic triads use true distances, distinct nodes, not a neural test score.
    tri = rng.integers(0,n,size=(50000,3))
    tri = tri[(tri[:,0]!=tri[:,1]) & (tri[:,1]!=tri[:,2]) & (tri[:,0]!=tri[:,2])]
    i,j,k = tri.T
    circulation = .5*(D[i,j]-D[j,i]+D[j,k]-D[k,j]+D[k,i]-D[i,k])
    report['cycle_diagnostic'] = {'count':len(tri), 'nonzero_fraction_at_1e_minus_8':float(np.mean(np.abs(circulation)>1e-8)),
                                  'absolute_circulation_quantiles':np.quantile(np.abs(circulation),[.5,.9,.99,1]).tolist()}
    report['dataset'] = {'name':'legacy_Chengdu_1901', 'nodes':n, 'query_pairs':len(q),
                         'not_a_reproduction_of': ['OSM_Harbin_Small', 'OSM_Beijing'],
                         'provenance':check, 'input_sha256':{p.name:sha256(p) for p in data.iterdir() if p.is_file()}}
    report['protocol'] = {'seed':args.seed, 'grouping':'unordered OD pairs; both directions stay together',
                           'split_counts':{'train':len(train),'validation':len(val),'test':len(test)},
                           'stage':'exploratory diagnostic, not confirmatory benchmark'}
    report['code_commit_at_run'] = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    report['runtime_seconds'] = time.perf_counter()-start
    (data/'protocol_r1').mkdir(exist_ok=True)
    np.savez_compressed(data/'protocol_r1/splits.npz', train=train, validation=val, test=test,
                        coordinates=coords[:,1:3].astype(np.float32), edges=links[:,:2].astype(np.int64), potential=h)
    report['split_artifact_sha256'] = sha256(data/'protocol_r1/splits.npz')
    (out/'diagnostic_chengdu.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for split in ('train','validation','test'):
        print('DIRECTION_DIAGNOSTIC', split, json.dumps(report['split_metrics'][split]), flush=True)
    print('DIRECTION_DIAGNOSTIC_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
