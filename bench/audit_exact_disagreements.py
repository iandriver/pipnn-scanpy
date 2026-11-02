"""Audit every float32 exact-implementation disagreement with float64 search."""
from pathlib import Path
import json
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='4'
import numpy as np
from scipy.spatial.distance import cdist

import argparse
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,default=Path('bench/results/public_knn'))
p.add_argument('--data',type=Path,default=Path('bench/data/public_knn'))
a=p.parse_args();root=a.output;data=a.data
for name in json.loads((root/'datasets.json').read_text()):
    f=np.load(root/name/'faiss_0.npy')[:,1:];s=np.load(root/name/'sklearn_0.npy')[:,1:]
    different=np.flatnonzero(np.any(np.sort(f,axis=1)!=np.sort(s,axis=1),axis=1))
    x=np.load(data/name/'X.npy').astype(np.float64);records=[]
    for start in range(0,len(different),16):
        qs=different[start:start+16];ds=cdist(x[qs],x,'sqeuclidean')
        ds[np.arange(len(qs)),qs]=np.inf
        truth=np.argpartition(ds,14,axis=1)[:,:15]
        for local,(q,t) in enumerate(zip(qs,truth)):
            cutoff=float(max(ds[local,t]));r=dict(row=int(q),cutoff_squared_distance=cutoff)
            for method,pred in [('faiss',f[q]),('sklearn',s[q])]:
                r[method]=dict(missed_ids=len(set(t)-set(pred)),
                               max_selection_excess_squared=float(max(ds[local,pred])-cutoff))
            records.append(r)
    result=dict(disagreeing_rows=len(different),checked_all_disagreeing_rows=True,
                faiss_oracle_mismatch_rows=sum(r['faiss']['missed_ids']>0 for r in records),
                sklearn_oracle_mismatch_rows=sum(r['sklearn']['missed_ids']>0 for r in records),
                records=records)
    (root/name/'exact_disagreement_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(name,{k:v for k,v in result.items() if k!='records'},flush=True)
