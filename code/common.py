from pathlib import Path
import os, sys, hashlib
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('CIRCADIAN_OUTPUT', ROOT)).resolve()
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.cache' / 'matplotlib'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(OUT / '.cache' / 'numba'))
import numpy as np
import pandas as pd
from scipy import stats
SEED=20260907
AGE_ORDER=['Young','EarlyMid','Mid','Old']
METRICS=['total_g','preference','reward_g','water_g']
for folder in ['data','results','figures','reports','audit']:
    (OUT/folder).mkdir(exist_ok=True,parents=True)
def rng_for(label):
    return np.random.default_rng(SEED+int(hashlib.sha256(str(label).encode()).hexdigest()[:8],16))
def save_table(df,name): df.to_csv(OUT/'results'/f'{name}.csv',index=False)
def bh(p):
    p=np.asarray(p,float); q=np.full(p.shape,np.nan); m=np.isfinite(p)
    if m.any(): q[m]=stats.false_discovery_control(p[m],method='bh')
    return q
def wrap_hours(x): return (np.asarray(x)+12)%24-12
def mean_phase(x):
    x=np.asarray(x); x=x[np.isfinite(x)]
    return np.angle(np.mean(np.exp(2j*np.pi*x/24)))*24/(2*np.pi) %24 if len(x) else np.nan
def resultant(x): return abs(np.mean(np.exp(2j*np.pi*np.asarray(x)/24)))
def mdtable(df,digits=4):
    def fmt(x):
        if isinstance(x,(float,np.floating)):return '' if not np.isfinite(x) else f'{x:.{digits}g}'
        return str(x).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))
