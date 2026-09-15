"""Four fixed synthetic graphs; all-pairs labels are representation diagnostics."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import networkx as nx
import numpy as np
import scipy
from scipy.sparse.csgraph import floyd_warshall

ROOT = Path(__file__).resolve().parents[1]
CASES = ('tree127', 'cycle97', 'grid12', 'cube6_cycle3')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False)+'\n')
    tmp.replace(path)


def make_graph(name):
    g = nx.DiGraph()
    if name == 'tree127':
        g.add_nodes_from(range(127)); rng = np.random.default_rng(20260914)
        for child in range(1, 127):
            parent = (child-1)//2
            a, b = rng.integers(1, 6, 2)
            g.add_edge(parent, child, weight=int(a)); g.add_edge(child, parent, weight=int(b))
    elif name == 'cycle97':
        g.add_nodes_from(range(97))
        for i in range(97): g.add_edge(i, (i+1)%97, weight=1)
    elif name == 'grid12':
        side = 12; g.add_nodes_from(range(side*side))
        def add(a,b): g.add_edge(a[0]*side+a[1],b[0]*side+b[1],weight=1)
        for r in range(side):
            for c in range(side-1):
                a,b=(r,c),(r,c+1)
                add(a,b) if r%2==0 else add(b,a)
        for c in range(side):
            for r in range(side-1):
                a,b=(r,c),(r+1,c)
                add(a,b) if c%2==0 else add(b,a)
        boundary=([(0,c) for c in range(side)]+[(r,side-1) for r in range(1,side)]
                  +[(side-1,c) for c in range(side-2,-1,-1)]+[(r,0) for r in range(side-2,0,-1)])
        for a,b in zip(boundary,boundary[1:]+boundary[:1]): add(a,b)
        keep=sorted(max(nx.strongly_connected_components(g),key=lambda x:(len(x),-min(x))))
        if len(keep)<100: raise AssertionError('Grid largest SCC too small')
        g=nx.relabel_nodes(g.subgraph(keep).copy(),{v:i for i,v in enumerate(keep)})
        return g,np.array(keep,dtype=np.int64)
    elif name == 'cube6_cycle3':
        g.add_nodes_from(range(192))
        for x in range(64):
            for a in range(3):
                u=3*x+a
                for bit in range(6):g.add_edge(u,3*(x^(1<<bit))+a,weight=1)
                g.add_edge(u,3*x+(a+1)%3,weight=1)
    else:raise ValueError(name)
    return g,np.arange(len(g),dtype=np.int64)


def get_labels(g):
    n=len(g)
    assert nx.is_strongly_connected(g)
    assert all(d['weight']>0 for _,_,d in g.edges(data=True))
    D=np.full((n,n),np.inf,dtype=np.float64)
    for u,values in nx.all_pairs_dijkstra_path_length(g,weight='weight'):
        for v,d in values.items():D[u,v]=d
    matrix=nx.to_numpy_array(g,nodelist=range(n),weight='weight')
    other=floyd_warshall(matrix,directed=True)
    np.testing.assert_array_equal(D,other)
    assert np.isfinite(D).all() and (D[~np.eye(n,dtype=bool)]>0).all()
    return D


def prepare(root=ROOT):
    data=root/'data/protocol_r4b';report=root/'reports/audit-r4b-20260914'
    if data.exists():raise FileExistsError(data)
    data.mkdir(parents=True);report.mkdir(parents=True,exist_ok=True)
    manifest={'classification':'synthetic full-matrix representation/optimization diagnostic, NOT generalization',
              'protocol':'R4 v1.0 stage B with preregistered implementation clarification',
              'graphs':{},'versions':{'python':platform.python_version(),'numpy':np.__version__,
              'scipy':scipy.__version__,'networkx':nx.__version__},
              'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()}
    for index,name in enumerate(CASES):
        g,mapping=make_graph(name);D=get_labels(g);n=len(g)
        rng=np.random.default_rng(20260914+index)
        triples=np.array([rng.choice(n,3,replace=False) for _ in range(2048)],dtype=np.int64)
        edges=np.array([(u,v,d['weight']) for u,v,d in sorted(g.edges(data=True))],dtype=np.float64)
        path=data/f'{name}.npz'
        np.savez_compressed(path,distances=D,edges=edges,node_mapping=mapping,triples=triples)
        if name=='cycle97':np.testing.assert_array_equal(D,(np.arange(n)[None,:]-np.arange(n)[:,None])%n)
        if name=='cube6_cycle3':
            ids=np.arange(n);bits=ids//3;cycles=ids%3
            expected=np.array([[(int(x)^int(y)).bit_count() for y in bits] for x in bits])+(cycles[None,:]-cycles[:,None])%3
            np.testing.assert_array_equal(D,expected)
        manifest['graphs'][name]={'nodes':n,'directed_arcs':g.number_of_edges(),'ordered_fit_pairs':n*(n-1),
            'strongly_connected':True,'strict_positive_weights':True,'independent_label_max_difference':0.,
            'mean_distance':float(D[~np.eye(n,dtype=bool)].mean()),'max_distance':float(D.max()),
            'file':str(path.relative_to(root)),'sha256':sha(path),
            'arrays_sha256':{k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
                            for k,v in [('D',D),('edges',edges),('mapping',mapping),('triples',triples)]}}
        print('R4B_GRAPH_READY',name,json.dumps(manifest['graphs'][name]),flush=True)
    save_json(report/'data_manifest.json',manifest)
    return manifest

if __name__=='__main__':prepare()
