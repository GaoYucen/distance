import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from utils.asymmetric_metrics import L1Tilde,LInfTilde
from utils.audit_protocol import assert_disjoint_od,split_training_pairs,resample_landmark_rows
from models.audit_cross_encoder import CrossEncoder
from scripts.audit_directionality import fit_potential

torch.set_num_threads(2)

@pytest.mark.parametrize('r,s',[(63,1),(62,2),(2,62),(64,0),(0,64)])
def test_potential_collapse(r,s):
    torch.manual_seed(1);x=torch.randn(10,r+s);y=torch.randn(10,r+s)
    got=L1Tilde(r,s)(x,y)
    expected=(y[:,:r]-x[:,:r]).abs().sum(1,keepdim=True)+y[:,r:].sum(1,keepdim=True)-x[:,r:].sum(1,keepdim=True)
    torch.testing.assert_close(got,expected,atol=1e-5,rtol=1e-5)

@pytest.mark.parametrize('r,s',[(-1,2),(0,0),(2,-1)])
def test_bad_dimensions_rejected(r,s):
    with pytest.raises(ValueError):L1Tilde(r,s)


def test_tensor_shape_guard():
    with pytest.raises(ValueError):L1Tilde(2,1)(torch.zeros(2,4),torch.zeros(2,4))


def test_shared_encoder_can_be_asymmetric_and_negative():
    x=torch.tensor([[0.,0.]]);y=torch.tensor([[1.,2.]])
    f=L1Tilde(1,1)
    assert f(x,y).item()==3 and f(y,x).item()==-1


def test_triangle_and_zero_circulation():
    torch.manual_seed(2);x,y,z=[torch.randn(40,5,dtype=torch.float64) for _ in range(3)]
    f=L1Tilde(3,2)
    assert torch.all(f(x,z)<=f(x,y)+f(y,z)+1e-12)
    cycle=f(x,y)-f(y,x)+f(y,z)-f(z,y)+f(z,x)-f(x,z)
    torch.testing.assert_close(cycle,torch.zeros_like(cycle),atol=1e-12,rtol=0)


def test_linf_exact_directed_cycle():
    D=torch.tensor([[0.,1.,2.],[2.,0.,1.],[1.,2.,0.]])
    phi=D.T
    for u in range(3):
        for v in range(3):
            assert LInfTilde()(phi[u:u+1],phi[v:v+1]).item()==D[u,v].item()


def test_potential_solver_recovery_and_cycle_residual():
    h=np.array([0.,.1,-.2,.3])
    q=np.array([[u,v,2+h[v]-h[u],2-h[v]+h[u]] for u in range(4) for v in range(u+1,4)])
    pred,comp,info=fit_potential(q,4)
    np.testing.assert_allclose(pred-pred[0],h,atol=1e-8)
    cyc=np.array([[0,1,1,2],[1,2,1,2],[2,0,1,2]])
    p,_,_=fit_potential(cyc,3)
    err=p[cyc[:,1]]-p[cyc[:,0]]-(cyc[:,2]-cyc[:,3])/2
    assert np.linalg.norm(err)>.1


def test_grouped_split_keeps_reverse_pairs():
    q=np.array([[0,1,2,3],[1,0,3,2],[0,2,4,5],[2,0,5,4],[1,2,3,6],[2,1,6,3]])
    t,v=split_training_pairs(q,.34,42)
    assert_disjoint_od(t,v)
    assert len(t)+len(v)==len(q)
    t2,v2=split_training_pairs(q,.34,42)
    np.testing.assert_array_equal(t,t2);np.testing.assert_array_equal(v,v2)


def test_reverse_leakage_rejected():
    with pytest.raises(ValueError,match='OD leakage'):
        assert_disjoint_od(np.array([[0,1,2]]),np.array([[1,0,3]]))


def test_landmark_resampling_preserves_triples():
    torch.manual_seed(42)
    i=torch.tensor([0,2,3,4]);j=torch.tensor([1,3,4,2]);y=(10*i+j).float().reshape(-1,1)
    i0=i.clone();j0=j.clone();y0=y.clone()
    a,b,c=resample_landmark_rows(i,j,y,{0},.75)
    assert int((a==0).sum())==3
    torch.testing.assert_close(c.flatten(),(10*a+b).float())
    torch.testing.assert_close(i,i0);torch.testing.assert_close(j,j0);torch.testing.assert_close(y,y0)


def test_landmark_absent_leaves_batch_unchanged():
    i=torch.tensor([2,3]);j=torch.tensor([3,2]);y=torch.tensor([23.,32.])
    a,b,c=resample_landmark_rows(i,j,y,{0},.75)
    torch.testing.assert_close(a,i);torch.testing.assert_close(b,j);torch.testing.assert_close(c,y)


def test_cross_decoder_uses_declared_dimensions():
    m=CrossEncoder(4,hidden=8,output_dim=64,mode='l1tilde',r=62,s=2)
    out=torch.zeros(1,128);out[0,74]=-3
    assert m.decode(out).item()==3/64
    out.zero_();out[0,127]=-3
    assert m.decode(out).item()==-3/64


def test_cross_l1_can_be_asymmetric():
    m=CrossEncoder(2,hidden=4,output_dim=2,mode='l1',r=1,s=1)
    m.backbone=nn.Identity();m.head=nn.Linear(2,4,bias=False)
    with torch.no_grad():m.head.weight.zero_();m.head.weight[2,0]=1
    assert m(torch.tensor([[2.,1.]])).item()!=m(torch.tensor([[1.,2.]])).item()


def test_dist2gnn_encoder_receives_gradient_after_repair():
    from models.dist2gnn_model import Dist2GNNModel
    torch.manual_seed(4)
    m=Dist2GNNModel(num_nodes=6,gnn_input_dim=4,gnn_hidden_dim=16,gnn_output_dim=6,
        gnn_num_layers=1,node_features=np.arange(12,dtype=np.float32).reshape(6,2)/12,r=3,s=1)
    m.build_gnn_graph(torch.tensor([[0,1,2,3,4,5],[1,2,3,4,5,0]]))
    i=torch.tensor([0,1,2,3]);j=torch.tensor([2,3,4,5])
    loss=(m(i,j)-1).square().mean();loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in m.gnn.parameters() if p.grad is not None)>0
    assert sum(float(p.grad.abs().sum()) for p in m.pairwise_mlp.parameters() if p.grad is not None)>0
    # Consecutive optimizer steps must not reuse detached/stale embeddings.
    opt=torch.optim.Adam(m.parameters(),lr=.001)
    opt.step();opt.zero_grad();(m(i,j)-1).square().mean().backward();opt.step()


def test_legacy_entry_is_import_safe():
    p=Path('ablation_study/scripts/cross_encoder_test.py')
    spec=importlib.util.spec_from_file_location('safe_legacy_entry',p)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    assert mod.CrossEncoder is CrossEncoder


def test_root_training_no_test_alias_validation():
    code=Path('train.py').read_text()
    assert 'val_dataloader = DataLoader(test_dataset' not in code
    assert 'val_dataloader = DataLoader(val_dataset' in code
    assert 'replicate_test=True' not in code


def test_test_labels_cannot_change_selected_epoch():
    from scripts.audit_pilot import train_one
    coords=np.array([[0.,0.],[1.,.1],[.2,1.],[1.,1.]],dtype=np.float32)
    edges=np.array([[0,1],[1,2],[2,3],[3,0]])
    train=np.array([[0,1,1.,1.1],[0,2,1.2,1.3],[1,2,1.4,1.5]])
    val=np.array([[0,3,1.8,1.9]])
    test=np.array([[1,3,1.3,1.4],[2,3,1.2,1.3]])
    changed=test.copy();changed[:,2:]*=2
    a=train_one('cross_scalar',42,train,val,test,coords,edges,epochs=2,hidden=8,batch_size=6)
    b=train_one('cross_scalar',42,train,val,changed,coords,edges,epochs=2,hidden=8,batch_size=6)
    assert a['best_validation_epoch']==b['best_validation_epoch']
    assert a['history']==b['history']
    assert a['test']['all']['mre_percent']!=b['test']['all']['mre_percent']
