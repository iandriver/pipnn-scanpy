"""Summarize the public PBMC3k benchmark and draw graph comparisons."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata as md
import json
import statistics as stats
import sys
import platform
import numpy as np

ROOT=Path('bench/results/public_knn')
DATA=Path('bench/data/public_knn')
METHODS=['faiss','sklearn','direct','index','scanpy_pynn','pynndescent']
NAMES=dict(faiss='FAISS exhaustive',sklearn='sklearn exact',direct='PiPNN direct',
           index='PiPNN index',scanpy_pynn='Scanpy-configured PyNNDescent',
           pynndescent='PyNNDescent library')


def summarize():
    datasets=json.loads((ROOT/'datasets.json').read_text());results={}
    for name,m in datasets.items():
        truth=np.load(ROOT/name/'sklearn_0.npy')[:,1:]
        labels=np.load(DATA/name/'labels.npy');out={}
        for method in METHODS:
            records=[json.loads((ROOT/name/f'{method}_{r}.json').read_text()) for r in range(3)]
            assert all(rec['input_sha256']==m['input_sha256'] for rec in records)
            pred=np.load(ROOT/name/f'{method}_0.npy')[:,1:];recall=np.empty(len(pred))
            for start in range(0,len(pred),10000):
                p=pred[start:start+10000];t=truth[start:start+len(p)]
                recall[start:start+len(p)]=(p[:,:,None]==t[:,None,:]).any(axis=2).sum(axis=1)/15
            by_type={label:dict(cells=int((labels==i).sum()),recall=float(recall[labels==i].mean()))
                     for i,label in enumerate(m['categories']) if (labels==i).any()}
            rare={label:r for label,r in by_type.items() if r['cells']/len(labels)<.01}
            out[method]=dict(median_seconds=stats.median(r['seconds'] for r in records),
                             min_seconds=min(r['seconds'] for r in records),max_seconds=max(r['seconds'] for r in records),
                             median_peak_mb=stats.median(r['peak_process_mb'] for r in records),
                             graph_stable_across_repeats=len({r['graph_ids_sha256'] for r in records})==1,
                             recall=float(recall.mean()),by_type=by_type,rare=rare,records=records)
        graph=ROOT/name/'downstream.json'
        downstream=json.loads(graph.read_text()) if graph.exists() else None
        if downstream is not None:
            assert downstream.get('reference')=='sklearn', 'run corrected-reference graph checks first'
        audit=json.loads((ROOT/name/'exact_disagreement_audit.json').read_text())
        results[name]=dict(dataset=m,search=out,downstream=downstream,
                           exact_disagreement_audit=audit)
    return results



def figures(case):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    search=case['search'];x=np.arange(len(METHODS))
    fig,axs=plt.subplots(1,3,figsize=(14,4.5))
    for ax,key,title in zip(axs,['median_seconds','median_peak_mb','recall'],
                             ['Warm construction (seconds)','Whole-worker peak RSS (MB)','Neighbor recall@15 (%)']):
        vals=[search[m][key]*(100 if key=='recall' else 1) for m in METHODS]
        ax.bar(x,vals,color=['#0072b2','#56b4e9','#d55e00','#e69f00','#cc79a7','#009e73'])
        ax.set_xticks(x,[NAMES[m] for m in METHODS],rotation=35,ha='right',fontsize=8)
        ax.set_title(title);ax.grid(axis='y',alpha=.2)
        if key=='median_seconds':ax.set_yscale('log')
    fig.suptitle('Public PBMC3k: same cells, fixed search settings, three fresh-process repeats')
    fig.tight_layout();fig.savefig(ROOT/'search.png',dpi=160);fig.savefig(ROOT/'search.svg');plt.close(fig)
    checks=case['downstream']['checks']
    keys=['faiss_umap','direct_umap','index_umap','scanpy_pynn_umap','pynndescent_umap',
          'sklearn_union','sklearn_mutual','sklearn_snn','sklearn_scanpy_jaccard']
    labels=['FAISS + UMAP','PiPNN direct + UMAP','PiPNN index + UMAP','Scanpy PyNNDescent + UMAP',
            'Library PyNNDescent + UMAP','Exact + union','Exact + mutual only','Exact + union-SNN','Exact + Scanpy Jaccard']
    fig,ax=plt.subplots(figsize=(9,5))
    for i,key in enumerate(keys):
        values=[r['ari_to_exact_same_seed'] for r in checks[key]['clustering']['seeds']]
        ax.plot(values,[i]*3,'o',alpha=.6,color='#d55e00' if i<5 else '#0072b2')
    baseline=stats.mean(checks['sklearn_umap']['clustering']['within_graph_seed_ari'])
    ax.axvline(baseline,color='black',ls='--',label='Exact graph: seed-to-seed mean')
    ax.set_yticks(range(len(keys)),labels);ax.invert_yaxis();ax.set_xlim(0,1.02)
    ax.set_xlabel('Leiden ARI against exact + UMAP (1 = identical)')
    ax.set_title('Change search (orange) or graph design (blue); three Leiden seeds')
    ax.legend(fontsize=8,loc='lower left');ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/'design.png',dpi=160);fig.savefig(ROOT/'design.svg');plt.close(fig)


def report(case):
    lines=['# Public PBMC3k: exact search and graph checks','',
           'All 2,638 cells and 50 stored PCs; Euclidean distance, self plus 15 other neighbors, four-thread settings. Three randomized serial measurements in fresh processes.','',
           '| Builder | Warm seconds, median | Peak process MB, median | Recall@15 |',
           '|---|---:|---:|---:|']
    for m in METHODS:
        r=case['search'][m]
        lines.append(f"| {NAMES[m]} | {r['median_seconds']:.4f} | {r['median_peak_mb']:.1f} | {100*r['recall']:.4f}% |")
    lines+=['','![Resource and recall comparison](search.png)','',
            'Timing excludes imports, input loading and a method warmup. Peak RSS includes dependencies, input, warmup and construction. Common output normalization and evaluation are outside the timer; API output formats differ. This small example is not a large-data speed benchmark.','',
            '## Graphs and clustering','','![Graph and clustering comparison](design.png)','',
            'Orange changes search with fixed UMAP weighting. Blue changes graph design on fixed exact neighbors. Each point is one Leiden seed. This is a sensitivity check, not independent biological validation.','',
            '| Graph | Components | Isolated cells | Mean ARI to exact + UMAP |',
            '|---|---:|---:|---:|']
    for key,r in case['downstream']['checks'].items():
        ari=stats.mean(v['ari_to_exact_same_seed'] for v in r['clustering']['seeds'])
        lines.append(f"| {key} | {r['graph']['components']} | {r['graph']['isolated_cells']} | {ari:.6f} |")
    lines+=['','## Small groups','','| Group | Cells | Builder | Recall@15 |','|---|---:|---|---:|']
    for m in METHODS:
        for group,r in case['search'][m]['rare'].items():
            lines.append(f"| {group} | {r['cells']} | {NAMES[m]} | {100*r['recall']:.4f}% |")
    audit=case['exact_disagreement_audit']
    lines+=['','## Precision and scope','',
            f"FAISS/sklearn differing cells: {audit['disagreeing_rows']}. Every disagreement is audited with float64 exhaustive distances. The accuracy reference is sklearn.",'',
            'Scanpy already chooses exact search automatically for a dataset this small. ANN controls force their respective backends. Selected distances are recomputed in float64 and cast to float32 before common UMAP weighting. Leiden uses resolution 1, two iterations, seeds 0/1/2. SNN symmetrization rules and the self-count convention are explained in docs/exact_graph_public.md. The original tutorial annotations are not an independent biological truth set.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=DATA);p.add_argument('--output',type=Path,default=ROOT)
    a=p.parse_args();DATA=a.data;ROOT=a.output
    results=summarize()
    if list(results)!=['pbmc3k']:raise ValueError('This reproduction expects only public PBMC3k.')
    from pipnn import _pipnn
    import scanpy.neighbors._connectivity as connectivity
    receipt=dict(threads=4,nonself_neighbors=15,python=sys.version,platform=platform.platform(),
                 versions={p:md.version(p) for p in ['numpy','scipy','scikit-learn','scanpy','anndata','faiss-cpu','pynndescent','igraph']},
                 source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                                ['bench/bench_exact_graph.py','bench/report_exact_graph.py','bench/audit_exact_disagreements.py',
                                 'python/pipnn/transformer.py']},
                 extension_sha256=hashlib.sha256(Path(_pipnn.__file__).read_bytes()).hexdigest(),
                 scanpy_connectivity_sha256=hashlib.sha256(Path(connectivity.__file__).read_bytes()).hexdigest(),
                 results=results)
    (ROOT/'summary.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    figures(results['pbmc3k']);report(results['pbmc3k'])
    print(ROOT/'report.md')
