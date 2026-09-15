"""Reproduce original code defects without modifying original files or weights."""
from pathlib import Path
import ast,json,subprocess,types
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
BASE='c4fb4797efbe4485a337cf9ae446e9d28ed14fef'

def source(path):return subprocess.check_output(['git','show',f'{BASE}:{path}'],text=True)

def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    old=types.ModuleType('legacy_dist2gnn')
    exec(compile(source('models/dist2gnn_model.py'),'legacy_dist2gnn','exec'),old.__dict__)
    m=old.Dist2GNNModel(num_nodes=6,gnn_input_dim=4,gnn_hidden_dim=8,gnn_output_dim=4,
        gnn_num_layers=1,node_features=np.arange(12,dtype=np.float32).reshape(6,2)/12,r=3,s=1)
    m.build_gnn_graph(torch.tensor([[0,1,2,3,4,5],[1,2,3,4,5,0]]))
    i=torch.tensor([2,3,4,5]);j=torch.tensor([3,4,5,2]);y=(10*i+j).float()
    m(i,j).sum().backward()
    gnn_grad=sum(int(p.grad is not None) for p in m.gnn.parameters())
    mlp_grad=sum(int(p.grad is not None) for p in m.pairwise_mlp.parameters())
    captured=[]
    def capture(self,a,b,label,criterion,optimizer):
        captured.append((a.clone(),b.clone(),label.clone()))
        return torch.tensor(0.)
    m._train_step=types.MethodType(capture,m)
    m.fit(DataLoader(TensorDataset(i,j,y),batch_size=4),nn.MSELoss(),None,epochs=1,
          landmarks=[0,1],landmark_ratio=.75,display_step=1)
    a,b,label=captured[0]
    bad=int(((10*a+b).float()!=label.flatten()).sum())
    tree=ast.parse(source('ablation_study/scripts/cross_encoder_test.py'))
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='CrossEncoder')
    mod=ast.Module(body=[cls],type_ignores=[])
    env={'nn':nn,'torch':torch,'HIDDEN':8,'OUTPUT_DIM':64,'R':62,'S':2}
    exec(compile(ast.fix_missing_locations(mod),'legacy_cross_class','exec'),env)
    c=env['CrossEncoder'](128,hidden=8,mode='l1tilde');c.net=nn.Identity()
    x=torch.zeros(1,128);x[0,64+10]=-3
    old_value=float(c(x));correct=3/64
    report={'base_commit':BASE,
      'gnn_bypass':{'gnn_parameters_with_grad':gnn_grad,'pairwise_parameters_with_grad':mlp_grad,
                    'reproduced':gnn_grad==0 and mlp_grad>0},
      'landmark_label_corruption':{'batch_rows':4,'mismatched_rows':bad,'reproduced':bad>0},
      'cross_dimension_inversion':{'declared_symmetric_dims':62,'actual_symmetric_dims':2,
                                  'original_output':old_value,'intended_output':correct,'reproduced':old_value!=correct},
      'test_as_validation_static_check': 'val_dataloader = DataLoader(test_dataset' in source('train.py'),
      'cross_test_driven_scheduler_static_check': 'train(mode, X_train, y_train, X_test, y_test)' in source('ablation_study/scripts/cross_encoder_test.py')}
    path=Path('reports/audit-r1-20260913/baseline_reproduction.json')
    path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    assert all(report[k]['reproduced'] for k in ['gnn_bypass','landmark_label_corruption','cross_dimension_inversion'])

if __name__=='__main__':main()
