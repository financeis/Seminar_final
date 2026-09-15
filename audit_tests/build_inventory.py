"""Capture original source and data provenance without importing research scripts."""
from pathlib import Path
import ast
import importlib.metadata as md
import json
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

try:
    from .common import sha256, jsonable, resolve_paper
except ImportError:
    from common import sha256, jsonable, resolve_paper

BASELINE = '9439107b31f20c4b731ec1dc29f13c178beb311d'


def build(repo):
    repo = Path(repo).resolve()
    output = repo / 'reports/paper_audit'
    dest = output / 'evidence/inventory'
    dest.mkdir(parents=True, exist_ok=True)
    original = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASELINE], cwd=repo, text=True).splitlines()
    original = [p for p in original if Path(p).suffix.lower() in {'.py','.csv','.npz','.pdf'} or p in {'README.md','requirements.txt'}]
    sources = {p: sha256(repo/p) for p in original}
    files = []
    for path in sorted(repo.glob('*.py')):
        text = path.read_text(encoding='utf-8-sig')
        tree = ast.parse(text)
        nodes = []
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom)):
                nodes.append({'kind':type(n).__name__,'name':getattr(n,'name',None),'line':n.lineno,'end':getattr(n,'end_lineno',n.lineno)})
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in {'read_csv','to_csv','savez','load','history','check_call'}:
                nodes.append({'kind':'io','line':n.lineno,'source':ast.get_source_segment(text,n)})
        files.append({'path':path.name,'sha256':sources[path.name],'lines':len(text.splitlines()),'nodes':nodes})
    npz={}
    for path in sorted(repo.glob('*.npz')):
        with np.load(path,allow_pickle=False) as a:
            npz[path.name]={'keys':a.files,'arrays':{k:{'shape':list(a[k].shape),'dtype':str(a[k].dtype)} for k in a.files},
                            'generation_provenance':'No generating commit, seed, PCA basis, or input hashes stored in the array keys.'}
    raw=pd.read_csv(repo/'FRED-MD_2024m12.csv',skiprows=[1]); dates=pd.to_datetime(raw.sasdate)
    manifest={'id':'INVENTORY','status':'PASS','baseline_commit':BASELINE,'python_files':files,'npz':npz,
              'fred_raw':{'rows':len(raw),'variables':len(raw.columns)-1,'start':str(dates.min().date()),'end':str(dates.max().date())},
              'tracked_etf_returns': [p for p in original if p.endswith('.csv') and p!='FRED-MD_2024m12.csv'],
              'missing_methods':['Naive','Black-Litterman','MVO','lns','los','mx','GMM comparison','random regime controls','paired t-test','Nemenyi','conditional transition network','10 percent volatility scaling'],
              'method_presence_basis':'Manual review of every original Python script plus AST inventory; comments are not treated as executable implementations.'}
    (dest/'manifest.json').write_text(json.dumps(jsonable(manifest),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    versions={}
    for name in ['numpy','pandas','scipy','scikit-learn','matplotlib','pypdf','PyYAML','joblib','threadpoolctl']:
        try: versions[name]=md.version(name)
        except md.PackageNotFoundError: versions[name]=None
    env={'baseline_commit':BASELINE,'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':versions,
         'original_file_sha256':sources,'paper':{'path':'../idea_paper.pdf','sha256':sha256(resolve_paper(repo))},
         'data_sources':{'macro':'Repository snapshot named FRED-MD_2024m12.csv; identity with the official December 2024 vintage is unverified','etf':'No real ETF return CSV supplied; original downloader uses Yahoo, paper uses WRDS'},
         'replication_scope':'Local function and isolated script diagnostics; no reproduction of the published return tables.'}
    (output/'environment.json').write_text(json.dumps(env,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    assert len(files)==11
    print('Inventoried',len(files),'scripts and',len(sources),'original inputs; no real ETF CSV supplied')


if __name__=='__main__':
    build(Path(__file__).resolve().parents[1])
