import numpy as np
import pytest
from scripts.r3_potential_core import mre,optimal_pair_correction,select_shrink,node_design,fit_l2,fit_mre_lp

def test_error_decompositions():
 rng=np.random.default_rng(1);a=rng.uniform(.1,5,80);b=rng.uniform(.1,5,80);s=rng.uniform(.1,5,80);p=rng.normal(size=80)
 d=s-(a+b)/2;r=p-(a-b)/2
 np.testing.assert_allclose(.5*((s+p-a)**2+(s-p-b)**2),d*d+r*r)
 np.testing.assert_allclose(.5*(abs(s+p-a)+abs(s-p-b)),np.maximum(abs(d),abs(r)))
def test_pair_optimum():
 a=np.array([1.,3.,2.,.5]);b=np.array([3.,1.,2.,4.]);s=np.array([3.,1.,1.,2.]);p,bound=optimal_pair_correction(s,a,b)
 np.testing.assert_allclose(.5*(abs(s+p-a)/a+abs(s-p-b)/b),bound)
 for shift in [-3.,-.01,0.,.01,3.]:assert mre(s,p+shift,a,b)>=bound.mean()-1e-12
 assert p[0]!=(a[0]-b[0])/2

def test_component_gauge():
 X,c,k=node_design([[0,1],[1,2],[3,4]],6);t,info=fit_l2(X,[1,2,3]);h=np.zeros(6);h[k]=t
 np.testing.assert_allclose(X@t,[1,2,3],atol=1e-8)
 assert info['success'] and c[0]==c[2] and c[2]!=c[3] and c[4]!=c[5] and not k[[0,3,5]].any()
def test_lp_tree():
 X,_,_=node_design([[0,1],[1,2],[2,3]],4);a=np.array([1.,2.,4.]);b=np.array([3.,3.,1.]);s=np.array([2.5,2.,1.])
 t,info=fit_mre_lp(X,s,a,b,10);_,bound=optimal_pair_correction(s,a,b)
 assert info['success'] and info['numerical_optimum_certified']
 assert mre(s,X@t,a,b)==pytest.approx(bound.mean(),abs=1e-7)
def test_lp_cycle():
 X,_,_=node_design([[0,1],[1,2],[2,0]],3);a=np.ones(3);b=2*a;s=1.5*a
 t1,_=fit_l2(X,(a-b)/2);t2,info=fit_mre_lp(X,s,a,b,10)
 assert abs(np.sum(X@t2))<1e-8 and mre(s,X@t2,a,b)<=mre(s,X@t1,a,b)+1e-7
 assert info['absolute_duality_gap']<1e-7
def test_lp_feature():
 X=np.array([[1.,2.],[2.,-1.],[-1.,3.],[0.,1.]]);a=np.array([3.,2.,1.,1.]);b=np.array([1.,1.,2.,3.]);s=(a+b)/2
 t,info=fit_mre_lp(X,s,a,b,10)
 assert info['success'] and mre(s,X@t,a,b)<=mre(s,np.zeros(4),a,b)+1e-7
def test_shrink():
 rng=np.random.default_rng(45);a=rng.uniform(.1,4,20);b=rng.uniform(.1,4,20);s=rng.uniform(.1,4,20);p=rng.normal(size=20)
 lam=select_shrink(s,p,a,b)
 assert 0<=lam<=1 and mre(s,lam*p,a,b)<=min(mre(s,t*p,a,b) for t in np.linspace(0,1,2001))+1e-12
 assert select_shrink(s,np.zeros(20),a,b)==0
def test_reverse_and_negatives():
 s=np.array([1.,2.]);p=np.array([.2,-3.]);a=np.array([1.,2.]);b=np.array([2.,1.])
 assert mre(s,p,a,b)==pytest.approx(mre(s,-p,b,a)) and np.any(s+p<0)
def test_invalid():
 with pytest.raises(ValueError):mre([1],[0],[0],[1])
 with pytest.raises(ValueError):mre([1],[np.nan],[1],[1])
 with pytest.raises(ValueError):node_design([[1,1]],3)
 with pytest.raises(ValueError):node_design([[0,4]],3)

@pytest.mark.parametrize('seed',range(5))
def test_equivalent_dual(seed):
 from scripts.r3_potential_core import fit_mre_lp_dual
 rng=np.random.default_rng(seed);X=rng.normal(size=(40,5));a=rng.uniform(.2,4,40);b=rng.uniform(.2,4,40);s=rng.uniform(.2,4,40)
 t,p=fit_mre_lp(X,s,a,b,10);tt,d=fit_mre_lp_dual(X,s,a,b,10)
 assert p['success'] and d['success'] and d['numerical_optimum_certified']
 assert abs(mre(s,X@t,a,b)-mre(s,X@tt,a,b))<1e-7
