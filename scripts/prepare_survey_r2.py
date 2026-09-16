"""R2: independently audit survey labels and retain native road direction.
Never interpret a stored orientation of an undirected .edges row as a one-way road.
Native CSV directions come from the primary source s41597-023-02589-y.
The Jinan paired derivative is NOT the original survey benchmark: reverse pairs
are grouped, both directions are labeled anew, and native decimal lengths retained.
"""
from pathlib import Path
import hashlib, json, sys, time
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_directionality import diagnostic
from utils.audit_protocol import assert_disjoint_od
SURVEY=ROOT/'data/survey_dcaa89d'
RAW=ROOT/'data/figshare_native_20260913'
OUT=ROOT/'reports/audit-r2-20260913'
DATA=ROOT/'data/protocol_r2'
SEED=20260913


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def as_csr(graph):
    a=nx.to_scipy_sparse_array(graph,nodelist=range(len(graph)),weight='weight',format='csr',dtype=np.float64)
    # Keep explicit zero-weight edges: eliminating zeros would change distances.
    return sparse.csr_matrix((a.data,a.indices.astype(np.int32),a.indptr.astype(np.int32)),shape=a.shape)


def pair_labels(graph,ids):
    """Batched source Dijkstra, O(batch*N) temporary memory, both directions."""
    ids=np.asarray(ids,dtype=np.int64)
    if len(ids)==0:raise ValueError('Empty query set')
    a=as_csr(graph);u,v=ids.T
    order_u=np.argsort(u);order_v=np.argsort(v)
    us=u[order_u];vs=v[order_v]
    sources=np.unique(ids)
    labels=np.full((len(ids),2),np.inf,dtype=np.float64)
    for begin in range(0,len(sources),32):
        ss=sources[begin:begin+32]
        ds=dijkstra(a,directed=graph.is_directed(),indices=ss)
        for row,source in enumerate(ss):
            ix=order_u[np.searchsorted(us,source,'left'):np.searchsorted(us,source,'right')]
            labels[ix,0]=ds[row,v[ix]]
            ix=order_v[np.searchsorted(vs,source,'left'):np.searchsorted(vs,source,'right')]
            labels[ix,1]=ds[row,u[ix]]
    return np.column_stack([ids,labels])


def nx_crosscheck(graph,q,seed,count=64):
    ids=np.random.default_rng(seed).choice(len(q),min(count,len(q)),replace=False)
    errors=[]
    for i in ids:
        u,v=map(int,q[i,:2])
        errors.extend([abs(nx.shortest_path_length(graph,u,v,weight='weight')-q[i,2]),
                       abs(nx.shortest_path_length(graph,v,u,weight='weight')-q[i,3])])
    error=float(max(errors))
    if error>1e-7:raise AssertionError(f'Independent label check failed: {error}')
    return {'checked_directions':len(errors),'max_abs_difference_m':error,'passed':True}


def canonical(q,n):
    uv=np.sort(np.asarray(q[:,:2],dtype=np.int64),axis=1)
    keys=uv[:,0]*n+uv[:,1]
    _,ix=np.unique(keys,return_index=True)
    return uv[ix],keys[ix]


def native_graph(city):
    nodes=pd.read_csv(RAW/f'node_{city}.csv').sort_values('NodeID')
    edges=pd.read_csv(RAW/f'edge_{city}.csv')
    assert set(['NodeID','Longitude','Latitude']).issubset(nodes.columns)
    assert set(['Origin','Destination','Length']).issubset(edges.columns)
    n=len(nodes);assert np.array_equal(nodes.NodeID.to_numpy(),np.arange(n))
    weights=edges.Length.to_numpy();assert np.isfinite(weights).all() and (weights>=0).all()
    g=nx.DiGraph();g.add_nodes_from(range(n))
    for u,v,w in edges[['Origin','Destination','Length']].itertuples(index=False,name=None):
        u,v=int(u),int(v);assert 0<=u<n and 0<=v<n
        if u!=v and (not g.has_edge(u,v) or w<g[u][v]['weight']):g.add_edge(u,v,weight=float(w))
    sc=sorted(nx.strongly_connected_components(g),key=lambda c:(-len(c),min(c)))
    keep=np.array(sorted(sc[0]),dtype=np.int64)
    mapping=np.full(n,-1,dtype=np.int64);mapping[keep]=np.arange(len(keep))
    h=nx.relabel_nodes(g.subgraph(keep).copy(),{int(old):int(mapping[old]) for old in keep})
    lonlat=nodes[['Longitude','Latitude']].to_numpy()
    zone=int(np.floor((float(np.mean(lonlat[:,0]))+180)/6))+1
    epsg=32600+zone
    proj=np.column_stack(Transformer.from_crs(4326,epsg,always_xy=True).transform(*lonlat.T))
    # Subtract a fixed graph-only origin in float64 before storing float32 features.
    coords=(proj[keep]-proj[keep].mean(0)).astype(np.float32)
    und=nx.Graph();und.add_nodes_from(range(len(h)))
    for u,v,d in h.edges(data=True):
        if not und.has_edge(u,v) or d['weight']<und[u][v]['weight']:und.add_edge(u,v,weight=d['weight'])
    meta={'raw_nodes':n,'raw_edge_rows':len(edges),'deduplicated_nonself_arcs':g.number_of_edges(),
          'strong_components':len(sc),'largest_strong_component_nodes':len(h),'nodes_excluded':n-len(h),
          'directed_arcs_used':h.number_of_edges(),'companion_undirected_edges':und.number_of_edges(),
          'zero_weight_arcs':sum(d['weight']==0 for _,_,d in h.edges(data=True)),
          'arcs_without_reverse_fraction':float(np.mean([not h.has_edge(v,u) for u,v in h.edges()])),
          'native_id_to_protocol_mapping':'largest SCC sorted original IDs -> zero-based',
          'parallel_edges':'minimum length for each ordered pair; self loops removed',
          'weights':'native decimal Length in metres, no integer truncation',
          'coordinate_epsg':epsg,'input_hashes':{f'{kind}_{city}.csv':digest(RAW/f'{kind}_{city}.csv') for kind in ['node','edge']}}
    return h,und,coords,keep,mapping,proj,meta


def clean_published_splits(n,mapping):
    folder=SURVEY/'data/W_Jinan/real_workload_perturb_500k'
    parts={};info={};original={};keysets={}
    for split,tag in [('train','train'),('validation','val'),('test','test')]:
        z=np.load(folder/f'W_Jinan_{tag}.queries.npz',allow_pickle=False)
        q=np.column_stack([z['src']-1,z['dst']-1,z['dist']]);original[split]=q
        assert np.isfinite(q).all() and np.all(q[:,:2]==np.floor(q[:,:2]))
        assert q[:,:2].min()>=0 and q[:,:2].max()<n
        uv,keys=canonical(q,n);keysets[split]=keys
        info[split]={'published_rows':len(q),'unique_unordered_pairs':len(keys),'within_split_repetitions':len(q)-len(keys)}
    info['cross_split_group_overlap']={f'{a}/{b}':int(len(np.intersect1d(keysets[a],keysets[b]))) for a,b in [('train','validation'),('train','test'),('validation','test')]}
    occupied=np.empty(0,dtype=np.int64)
    for i,split in enumerate(['test','validation','train']):
        keys=keysets[split];fresh=keys[~np.isin(keys,occupied)]
        occupied=np.union1d(occupied,keys)  # Reserve even unreachable held-out groups.
        uv=np.column_stack([fresh//n,fresh%n]);mapped=mapping[uv]
        valid=(mapped[:,0]>=0)&(mapped[:,1]>=0)&(mapped[:,0]!=mapped[:,1])
        uv=mapped[valid]
        rng=np.random.default_rng(SEED+i);uv=uv[rng.permutation(len(uv))]
        cap=100000 if split=='train' else 10000
        parts[split]=uv[:cap]
        info[split].update({'removed_overlap_with_higher_priority':len(keys)-len(fresh),
            'excluded_outside_scc_or_self':int((~valid).sum()),'eligible_groups':len(uv),'selected_for_diagnostic':len(parts[split])})
    return parts,info,original


def write_case(name,graph,labels,counts,coords,original_ids):
    a,b=counts[0],counts[0]+counts[1]
    tr,va,te=labels[:a],labels[a:b],labels[b:]
    assert_disjoint_od(tr,va,te)
    assert min(len(tr),len(va),len(te))>0
    diag,h=diagnostic(tr,va,te,len(graph))
    arcs=np.array(list(graph.edges()),dtype=np.int64)
    if not graph.is_directed():arcs=np.concatenate([arcs,arcs[:,::-1]],axis=0)
    dest=DATA/f'{name}.npz'
    np.savez_compressed(dest,train=tr,validation=va,test=te,coordinates=coords,edges=arcs,
                        original_node_ids=original_ids,potential=h)
    diag['dataset_name']=name;diag['data_sha256']=digest(dest)
    (OUT/f'{name}_diagnostic.json').write_text(json.dumps(diag,indent=2,allow_nan=False)+'\n')
    print('DATASET_DIAGNOSTIC',name,json.dumps(diag['split_metrics']['test']),flush=True)
    return {'file':str(dest.relative_to(ROOT)),'sha256':digest(dest),'split_counts':[len(tr),len(va),len(te)],
            'test_direction_energy_explained':diag['split_metrics']['test']['all']['potential_explained_energy_ratio_vs_zero']}


def smoke_tests():
    g=nx.DiGraph();g.add_weighted_edges_from([(0,1,0.),(1,2,1.),(2,0,2.)])
    q=pair_labels(g,[[0,1],[0,2],[1,2]])
    np.testing.assert_allclose(q[:,2:],[[0,3],[1,2],[1,2]])
    nx_crosscheck(g,q,1)


def main():
    start=time.perf_counter();smoke_tests()
    OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    if any(DATA.iterdir()):raise FileExistsError('Refusing to overwrite existing R2 protocol artifacts')
    report={'stage':'data and labels audited, not a model benchmark','seed':SEED,
            'upstream_commit':'dcaa89d38300bfb823eda84ccdfc85c42edbeae8',
            'survey_manifest':json.loads((SURVEY/'manifest.json').read_text()),
            'native_manifest':json.loads((RAW/'manifest.json').read_text()),'cases':{}}
    g,und,coords,keep,mapping,projected,meta=native_graph('jinan')
    nodes=np.loadtxt(SURVEY/'data/W_Jinan/W_Jinan.nodes',delimiter=',')
    assert np.array_equal(nodes[:,0],np.arange(len(nodes))+1)
    assert len(nodes)==len(projected)
    coordinate_error=float(np.max(np.linalg.norm(nodes[:,1:]-projected,axis=1)))
    if coordinate_error>.01:raise ValueError('Original-to-survey node ID mapping not proven by coordinates')
    meta['survey_mapping_max_projected_error_m']=coordinate_error
    published=nx.Graph();published.add_nodes_from(range(len(nodes)))
    edge_rows=np.loadtxt(SURVEY/'data/W_Jinan/W_Jinan.edges',delimiter=',')
    for u,v,w in edge_rows:
        u,v=int(u)-1,int(v)-1
        if not published.has_edge(u,v):published.add_edge(u,v,weight=float(w))
    parts,split_info,original=clean_published_splits(len(nodes),mapping)
    report['jinan_native']=meta;report['jinan_split_protocol']=split_info
    report['protocol_note']='Survey test > validation > train priority for unordered OD overlap removal; equal weight per unordered group, then both directions. Not original workload-frequency weighting.'
    test=original['test'];ix=np.random.default_rng(SEED).choice(len(test),512,replace=False)
    pq=test[ix];measured=pair_labels(published,pq[:,:2].astype(int))
    error=float(np.max(np.abs(measured[:,2]-pq[:,2])))
    report['published_survey_label_check']={'queries':512,'max_abs_difference_m':error,'passed':error<1e-6,'zero_weight_edges':sum(d['weight']==0 for _,_,d in published.edges(data=True))}
    # The sampled discrepancy was traced independently to zero-length edges.
    # Preserve the stored-weight result AND test the minimum-one-metre convention.
    clamped=published.copy()
    for _,_,edge in clamped.edges(data=True):edge['weight']=max(1.,edge['weight'])
    checked=pair_labels(clamped,pq[:,:2].astype(int))
    clamped_error=float(np.max(np.abs(checked[:,2]-pq[:,2])))
    report['published_survey_label_check']['stored_weight_check_passed']=error<1e-6
    report['published_survey_label_check']['minimum_one_metre_check_max_error']=clamped_error
    report['published_survey_label_check']['minimum_one_metre_check_passed']=clamped_error<1e-6
    report['published_survey_label_check']['passed']=clamped_error<1e-6
    report['published_survey_label_check']['convention']='Published labels match max(stored edge weight,1m) on checked queries; native experiments retain raw decimal lengths and recompute labels.'
    if clamped_error>=1e-6:raise ValueError('Published label convention is still unresolved')
    print('PUBLISHED_LABEL_CHECK',json.dumps(report['published_survey_label_check']),flush=True)
    counts=[len(parts[s]) for s in ['train','validation','test']]
    pairs=np.vstack([parts[s] for s in ['train','validation','test']])
    direct=pair_labels(g,pairs);symmetric=pair_labels(und,pairs)
    good=np.isfinite(direct[:,2:]).all(1)&np.isfinite(symmetric[:,2:]).all(1)&(direct[:,2:]>0).all(1)&(symmetric[:,2:]>0).all(1)
    boundaries=np.cumsum([0]+counts)
    cleaned_counts=[int(good[boundaries[i]:boundaries[i+1]].sum()) for i in range(3)]
    report['excluded_nonpositive_or_nonfinite_pairs']=int((~good).sum())
    direct,symmetric=direct[good],symmetric[good]
    np.testing.assert_allclose(symmetric[:,2],symmetric[:,3],rtol=0,atol=1e-7)
    report['undirected_roundoff_max_abs_m']=float(np.max(np.abs(symmetric[:,2]-symmetric[:,3])))
    symmetric[:,3]=symmetric[:,2]  # Undirected distances are symmetric; remove only verified floating-point roundoff.
    if np.any(direct[:,2:]<symmetric[:,2:]-1e-7):raise AssertionError('Symmetrization shortened-distance invariant failed')
    report['independent_label_checks']={'directed':nx_crosscheck(g,direct,SEED), 'undirected':nx_crosscheck(und,symmetric,SEED+1)}
    report['cases']['Jinan_native_directed']=write_case('Jinan_native_directed',g,direct,cleaned_counts,coords,keep)
    report['cases']['Jinan_native_undirected']=write_case('Jinan_native_undirected',und,symmetric,cleaned_counts,coords,keep)
    # A second city provides only structural evidence here; its sampled queries are UNIFORM, not the survey's workload.
    g2,und2,c2,k2,m2,p2,meta2=native_graph('shenzhen');rng=np.random.default_rng(SEED+100)
    n=len(g2);chosen=set()
    while len(chosen)<50000:
        uv=np.sort(rng.integers(0,n,size=(60000,2)),axis=1)
        chosen.update(int(u*n+v) for u,v in uv if u!=v)
    keys=np.array(sorted(chosen),dtype=np.int64);keys=keys[rng.permutation(len(keys))[:50000]]
    uv=np.column_stack([keys//n,keys%n]);q=pair_labels(g2,uv)
    mask=np.isfinite(q[:,2:]).all(1)&(q[:,2:]>0).all(1)
    report['shenzhen_native']=meta2;report['shenzhen_native']['query_source']='seeded uniform unordered pairs, not trajectory workload'
    report['shenzhen_native']['nonpositive_pairs_excluded']=int((~mask).sum());q=q[mask]
    counts2=[int(.8*len(q)),int(.1*len(q)),len(q)-int(.8*len(q))-int(.1*len(q))]
    report['independent_label_checks']['shenzhen']=nx_crosscheck(g2,q,SEED+2)
    report['cases']['Shenzhen_native_directed_uniform']=write_case('Shenzhen_native_directed_uniform',g2,q,counts2,c2,k2)
    report['runtime_seconds']=time.perf_counter()-start
    report['passed']=True
    (OUT/'data_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('DATA_AUDIT_COMPLETE',json.dumps({k:report[k] for k in ['jinan_native','jinan_split_protocol','cases','runtime_seconds']}),flush=True)

if __name__=='__main__':main()
