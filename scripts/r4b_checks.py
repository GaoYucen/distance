"""Decoder, gradient, independent interval-union and seed-isolation checks."""
from __future__ import annotations
import math
import subprocess
import numpy as np
import torch
from torch import nn
from torchqmet import IQE, MRNFixed
from r4b_models import MODES, SEEDS, UPSTREAM, UPSTREAM_SHA, SeedBatch, numpy_decode, triangle_check
from r4b_data import ROOT, get_labels, make_graph, save_json, sha


class SliceScale(nn.Module):
    def __init__(self,a,b,scale):
        super().__init__();self.a,self.b,self.scale=a,b,scale
    def forward(self,x):return x[...,self.a:self.b]*self.scale


def run_checks(root=ROOT):
    torch.set_num_threads(1)
    assert subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()==UPSTREAM_SHA
    records=[]
    def record(name,**kwargs):records.append({'name':name,'passed':True,**kwargs})
    for mode in MODES:
        m=SeedBatch(9,mode);mask=~torch.eye(9,dtype=torch.bool);m.calibrate(mask)
        p=m();p.square().sum().backward()
        assert all(torch.isfinite(x.grad).all() for x in m.parameters())
        assert m.table.shape==(3,9,64)
        for i in range(3):
            expected=numpy_decode(m.table[i].detach().numpy(),mode,float(m.raw_alpha[i]),float(m.calibration[i]))
            np.testing.assert_allclose(expected,p[i].detach().numpy(),rtol=3e-6,atol=3e-6)
            chk=triangle_check(expected,1e-10);assert chk['passed']
            np.testing.assert_allclose(np.diag(expected),0.,atol=1e-12)
            assert expected.min()>=-1e-12
        record('numpy_decoder_and_structure_'+mode)
        if mode.startswith('IQE-'):
            ref=IQE(64,dim_per_component=8,reduction='sum' if mode=='IQE-sum' else 'maxmean')
            x=m.table[0].detach().clone().requires_grad_(True)
            rp=ref(x[:,None],x[None,:])*m.calibration[0]
            torch.testing.assert_close(rp,p[0],rtol=2e-6,atol=2e-6)
            record('authors_IQE_'+mode,source_commit=UPSTREAM_SHA)
        if mode=='MRN-L2':
            ref=MRNFixed(64,proj_output_size=32)
            ref.sym_proj=SliceScale(0,32,math.sqrt(32))
            ref.asym_proj=SliceScale(32,64,-1.)
            x=m.table[0]
            rp=ref(x[:,None],x[None,:])*m.calibration[0]
            torch.testing.assert_close(rp,p[0],rtol=2e-6,atol=2e-6)
            record('authors_MRNFixed_postprojection_equivalence',source_commit=UPSTREAM_SHA,
                   convention='plain L2 = authors RMS with sqrt(32) coordinates; potential sign negated')
        batch=SeedBatch(7,mode);mask=~torch.eye(7,dtype=torch.bool);batch.calibrate(mask)
        serial=[SeedBatch(7,mode,(s,)) for s in SEEDS]
        for m1 in serial:m1.calibrate(mask)
        truth=torch.tensor(np.random.default_rng(917).uniform(.4,2,(7,7)),dtype=torch.float32)
        opt=torch.optim.Adam(batch.parameters(),lr=.01,foreach=False)
        opts=[torch.optim.Adam(m1.parameters(),lr=.01,foreach=False) for m1 in serial]
        for _ in range(4):
            opt.zero_grad();loss=(batch()[:,mask]-truth[mask]).square().mean(-1).sum();loss.backward();opt.step()
            for m1,o1 in zip(serial,opts):
                o1.zero_grad();l=(m1()[:,mask]-truth[mask]).square().mean();l.backward();o1.step()
        for i,m1 in enumerate(serial):
            torch.testing.assert_close(batch.table[i],m1.table[0],rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(batch.raw_alpha[i],m1.raw_alpha[0],rtol=1e-5,atol=1e-6)
        record('batched_vs_serial_Adam_'+mode,updates=4)
    for seed in SEEDS:
        m=SeedBatch(8,'IQE-sum',(seed,))
        with torch.no_grad():m.table.copy_(torch.round(m.table*5))
        expected=numpy_decode(m.table[0].detach().numpy(),'IQE-sum',-1.,1.)
        np.testing.assert_allclose(expected,m()[0].detach().numpy(),atol=1e-6)
    record('IQE_empty_overlapping_and_tied_intervals',seeds=list(SEEDS))
    g,_=make_graph('cube6_cycle3');D=get_labels(g);n=len(g)
    nodes=np.arange(n);bits=((nodes//3)[:,None]>>np.arange(6))&1;cycle=nodes%3
    for mode in ('B4','B8','Shared-L1'):
        t=np.zeros((n,64),dtype=np.float32)
        if mode=='Shared-L1':
            t[:,:6]=bits
            for landmark in range(3):t[:,56+landmark]=(cycle-landmark)%3
        else:
            k=int(mode[1:]);width=64//k
            for landmark in range(3):
                t[:,landmark*width:landmark*width+6]=bits
                t[:,landmark*width+width-1]=(cycle-landmark)%3
        p=numpy_decode(t,mode,-1.,1.)
        np.testing.assert_array_equal(p,D)
        record('analytic_product_exact_'+mode,nodes=n,coordinates_stored=64,
               active_coordinates=(21 if mode!='Shared-L1' else 9),max_error=0.)
    output={'passed':True,'checks':len(records),'records':records,
            'baseline_commit':UPSTREAM_SHA,'torch_version':torch.__version__,
            'upstream_files_sha256':{f:sha(UPSTREAM/f) for f in ['torchqmet/iqe.py','torchqmet/mrn.py','torchqmet/reductions.py']},
            'scope':'implementation checks and exact witnesses; not random-init fitting scores'}
    save_json(root/'reports/audit-r4b-20260914/implementation_checks.json',output)
    print('R4B_CHECKS_PASSED',len(records),flush=True)
    return output

if __name__=='__main__':run_checks()
