"""Training-only convex potential probes. No test data accepted by fit functions."""
import time,warnings
import numpy as np
from scipy import sparse
from scipy.optimize import linprog,OptimizeWarning
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr

def arrays(s,a,b):
 s,a,b=[np.asarray(x,dtype=np.float64).reshape(-1) for x in (s,a,b)]
 if not (s.shape==a.shape==b.shape) or not len(s):raise ValueError('Invalid paired shapes')
 if not all(np.isfinite(x).all() for x in (s,a,b)) or (s<0).any() or (a<=0).any() or (b<=0).any():raise ValueError('Require finite s>=0 and a,b>0')
 return s,a,b

def mre(s,p,a,b):
 s,a,b=arrays(s,a,b);p=np.asarray(p,dtype=np.float64).reshape(-1)
 if p.shape!=s.shape or not np.isfinite(p).all():raise ValueError('Invalid potential differences')
 return float(np.mean(.5*(abs(s+p-a)/a+abs(s-p-b)/b)))

def optimal_pair_correction(s,a,b):
 """Label-dependent diagnostic, never a deployed query rule."""
 s,a,b=arrays(s,a,b);S,A=(a+b)/2,(a-b)/2
 return A+(s-S)*np.sign(A),abs(s-S)/np.maximum(a,b)

def select_shrink(s,p,a,b):
 """Exact validation-only convex lambda selection in [0,1], zero tie preference."""
 s,a,b=arrays(s,a,b);p=np.asarray(p,dtype=np.float64);valid=abs(p)>1e-14
 if not valid.any():return 0.
 x=np.r_[(a[valid]-s[valid])/p[valid],(s[valid]-b[valid])/p[valid]]
 w=np.r_[abs(p[valid])/a[valid],abs(p[valid])/b[valid]]
 order=np.argsort(x,kind='stable');x,w=x[order],w[order]
 ix=min(np.searchsorted(np.cumsum(w),.5*w.sum(),side='left'),len(w)-1)
 lam=float(np.clip(x[ix],0.,1.));candidates=[0.,lam,1.]
 scores=[mre(s,t*p,a,b) for t in candidates];best=min(scores)
 return float(min(t for t,v in zip(candidates,scores) if v<=best+1e-12))

def node_design(ids,n_nodes):
 uv=np.asarray(ids,dtype=np.int64)
 if uv.ndim!=2 or uv.shape[1]!=2 or uv.min()<0 or uv.max()>=n_nodes:raise ValueError('Invalid IDs')
 if np.any(uv[:,0]==uv[:,1]):raise ValueError('No self pairs')
 rows=np.repeat(np.arange(len(uv)),2)
 B=sparse.csr_matrix((np.tile([-1.,1.],len(uv)),(rows,uv.ravel())),shape=(len(uv),n_nodes))
 adjacency=sparse.csr_matrix((np.ones(len(uv)),(uv[:,0],uv[:,1])),shape=(n_nodes,n_nodes))
 _,comp=connected_components(adjacency,directed=False);_,roots=np.unique(comp,return_index=True)
 keep=np.ones(n_nodes,dtype=bool);keep[roots]=False
 return B[:,keep],comp,keep

def fit_l2(X,A):
 A=np.asarray(A,dtype=np.float64).reshape(-1)
 if X.shape[0]!=len(A) or not np.isfinite(A).all():raise ValueError('Invalid target')
 t=time.perf_counter()
 if sparse.issparse(X):
  sol=lsqr(X,A,atol=1e-11,btol=1e-11,iter_lim=10000);theta=sol[0];ok=sol[1] in (0,1,2,4,5)
  detail={'solver':'LSQR','stop_code':int(sol[1]),'iterations':int(sol[2])}
 else:
  theta,_,rank,_=np.linalg.lstsq(np.asarray(X,dtype=np.float64),A,rcond=1e-10);ok=True
  detail={'solver':'SVD least squares','rank':int(rank),'rcond':1e-10}
 residual=np.asarray(X@theta).ravel()-A
 detail.update(success=bool(ok),seconds=time.perf_counter()-t,train_A_mse=float(np.mean(residual**2)))
 return theta,detail

def fit_mre_lp(X,s,a,b,time_limit=120.):
 """Weighted LAD LP. Free theta; slacks>=0. Check primal, dual and direct cost."""
 s,a,b=arrays(s,a,b);X=sparse.csr_matrix(X,dtype=np.float64)
 if X.shape[0]!=len(s) or not np.isfinite(X.data).all():raise ValueError('Invalid design')
 n,k=X.shape;D=sparse.vstack([X,-X],format='csr');target=np.r_[a-s,b-s]
 weights=np.r_[1/a,1/b]/(2*n);I=sparse.eye(2*n,format='csr')
 C=sparse.vstack([sparse.hstack([D,-I]),sparse.hstack([-D,-I])],format='csr')
 rhs=np.r_[target,-target];c=np.r_[np.zeros(k),weights];t=time.perf_counter()
 with warnings.catch_warnings():
  warnings.filterwarnings('ignore',category=OptimizeWarning,message='Unrecognized options detected.*')
  sol=linprog(c,A_ub=C,b_ub=rhs,bounds=[(None,None)]*k+[(0,None)]*(2*n),method='highs-ipm',
   options={'time_limit':float(time_limit),'presolve':True,'dual_feasibility_tolerance':1e-8,
   'primal_feasibility_tolerance':1e-8,'ipm_optimality_tolerance':1e-9,'threads':2})
 info={'solver':'SciPy linprog / HiGHS-IPM','success':bool(sol.success),'status':int(sol.status),
  'message':str(sol.message),'iterations':int(sol.nit),'seconds':time.perf_counter()-t,
  'time_limit':time_limit,'requested_threads':2}
 if not sol.success:return None,info
 theta=sol.x[:k];violation=float(max(0.,np.max(C@sol.x-rhs)))
 direct=mre(s,np.asarray(X@theta).ravel(),a,b);dual=float(rhs@sol.ineqlin.marginals)
 gap=abs(float(sol.fun)-dual);zero=mre(s,np.zeros(n),a,b)
 info.update(train_mre=direct,zero_train_mre=zero,lp_objective=float(sol.fun),primal_max_violation=violation,
  dual_objective=dual,absolute_duality_gap=gap,
  numerical_optimum_certified=violation<1e-6 and gap<1e-6 and abs(direct-sol.fun)<1e-6)
 if not info['numerical_optimum_certified'] or direct>zero+1e-6:raise AssertionError(f'LP verification failed: {info}')
 return theta,info

def fit_mre_lp_dual(X,s,a,b,time_limit=120.):
 """Equivalent LAD dual: same model/objective, only k feature equalities."""
 s,a,b=arrays(s,a,b);X=sparse.csr_matrix(X,dtype=np.float64);n,k=X.shape
 if n!=len(s) or not np.isfinite(X.data).all():raise ValueError('Invalid design')
 D=sparse.vstack([X,-X],format='csr');target=np.r_[a-s,b-s];weights=np.r_[.5/a,.5/b];t=time.perf_counter()
 with warnings.catch_warnings():
  warnings.filterwarnings('ignore',category=OptimizeWarning,message='Unrecognized options detected.*')
  sol=linprog(-target,A_eq=D.T.tocsc(),b_eq=np.zeros(k),bounds=np.column_stack([-weights,weights]),method='highs-ipm',
   options={'time_limit':time_limit,'presolve':True,'threads':2,'primal_feasibility_tolerance':1e-8,
   'dual_feasibility_tolerance':1e-8,'ipm_optimality_tolerance':1e-9})
 info={'solver':'HiGHS-IPM / equivalent LAD dual','success':bool(sol.success),'status':int(sol.status),
  'message':str(sol.message),'seconds':time.perf_counter()-t,'iterations':int(sol.nit),'equalities':k,'variables':2*n,'time_limit':time_limit}
 if not sol.success:return None,info
 theta=-np.asarray(sol.eqlin.marginals);direct=mre(s,np.asarray(X@theta).ravel(),a,b);dual=-float(sol.fun)/n
 residual=float(np.max(abs(D.T@sol.x)));violation=float(max(0.,np.max(abs(sol.x)-weights)));gap=abs(direct-dual)
 info.update(train_mre=direct,dual_objective_mre=dual,absolute_duality_gap=gap,dual_equality_residual=residual,
  dual_bound_violation=violation,numerical_optimum_certified=gap<1e-6 and residual<1e-5 and violation<1e-7)
 if not info['numerical_optimum_certified'] or direct>mre(s,np.zeros(n),a,b)+1e-6:raise AssertionError(f'Dual verification failed: {info}')
 return theta,info
