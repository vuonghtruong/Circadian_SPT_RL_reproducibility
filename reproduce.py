"""Portable entry point. Default: verify saved numerical results and render 14 figures."""
from pathlib import Path
import argparse, os, shutil, subprocess, sys

ROOT=Path(__file__).resolve().parent

def execute(script,*args,output=None):
    env=os.environ.copy()
    env['OMP_NUM_THREADS']='1'
    env['OPENBLAS_NUM_THREADS']='1'
    if output is not None:env['CIRCADIAN_OUTPUT']=str(output)
    subprocess.run([sys.executable,str(ROOT/'code'/script),*map(str,args)],cwd=ROOT,env=env,check=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='figures',choices=['verify','figures','statistics','refit','refit-check','sensitivity','phases','lag','genotypes'])
    p.add_argument('--output',type=Path,help='New output directory for optional numerical stages')
    p.add_argument('--workers',type=int,default=1,help='Independent animal workers for refit-check; deterministic seeds are unchanged')
    a=p.parse_args()
    if a.stage=='genotypes':
        execute('genotype_phase_comparisons.py')
        return
    if a.stage=='refit-check':
        dest=(a.output or ROOT/'recomputed'/'refit-check').resolve()
        execute('refit_reproducibility.py','--output',dest,'--workers',a.workers)
        return
    if a.stage in ['verify','figures']:
        execute('verify_package.py')
        if a.stage=='figures':
            execute('export_tables.py')
            execute('figure_style_20261017.py','--base',ROOT,'--dest',ROOT)
            execute('finalize_figures.py')
        return
    dest=(a.output or ROOT/'recomputed'/a.stage).resolve()
    if dest==ROOT or ROOT in dest.parents and dest.parts[len(ROOT.parts)] in ['data','results','code','tables','figures']:
        raise ValueError('Numerical reruns must use a separate output directory')
    if dest.exists():raise FileExistsError(f'Choose a new output directory: {dest}')
    for part in ['data','results']:shutil.copytree(ROOT/part,dest/part)
    execute('numerical_stages.py',a.stage,output=dest)
    print(f'Numerical outputs: {dest}')

if __name__=='__main__':main()
