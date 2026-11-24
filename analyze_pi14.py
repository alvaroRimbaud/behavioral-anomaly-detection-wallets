"""Análisis reproducible PI14: jamás declara hipótesis final con menos de 5 semillas."""
from __future__ import annotations
import argparse
import itertools
import math
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def analyze(root:Path,out:Path):
    seeds=sorted(x for x in root.glob('seed_*') if (x/'run_summary.json').is_file() and (x/'metrics_global.csv').is_file() and (x/'scores_test_all_models.csv').is_file())
    if not seeds: raise ValueError('No hay semillas completas para análisis.')
    out.mkdir(parents=True,exist_ok=True)
    info=[];frames=[];buds=[];configs=[]
    for d in seeds:
        s=json.loads((d/'run_summary.json').read_text())
        info.append(s)
        frames.append(pd.read_csv(d/'metrics_global.csv').assign(seed=int(s['seed'])))
        for fn,arr in [('precision_at_k.csv',buds),('selected_hyperparameters.csv',configs)]:
            if (d/fn).exists():arr.append(pd.read_csv(d/fn).assign(seed=int(s['seed'])))
    values=pd.DataFrame(frames[0]).iloc[0:0] if not frames else pd.concat(frames,ignore_index=True)
    values.to_csv(out/'metricas_por_semilla.csv',index=False)
    n=len(info)
    consistency=all(x['n_users']==info[0]['n_users'] and x['tuning_profile']==info[0]['tuning_profile'] for x in info)
    desc=(values.groupby(['model','feature_set'],as_index=False)
          .agg(n=('pr_auc','size'),media=('pr_auc','mean'),mediana=('pr_auc','median'),de=('pr_auc','std'),minimo=('pr_auc','min'),maximo=('pr_auc','max'),precision=('precision','mean'),recall=('recall','mean'),f1=('f1','mean'),umbral_media=('threshold','mean'),umbral_de=('threshold','std')))
    desc.to_csv(out/'estadistica_descriptiva.csv',index=False)
    effects=[]
    for model,group in values.groupby('model'):
        pivot=group.pivot(index='seed',columns='feature_set',values='pr_auc')
        if not {'raw','raw+individualizadas'}.issubset(pivot.columns):continue
        diff=(pivot['raw+individualizadas']-pivot['raw']).dropna().values
        delta=float(np.mean(diff))
        # Exact two-sided sign-flip test on seed-level paired differences.
        # With 5 seeds the smallest attainable two-sided p is 2/32=0.0625:
        # therefore p<.05 is IMPOSSIBLE with this exact test, even when all
        # five effects have the same sign. Report effect sizes and user-level
        # bootstrap per seed; do not falsely claim a significant seed-level test.
        signs = np.asarray(list(itertools.product([-1,1],repeat=len(diff))),float) if 1<len(diff)<=16 else None
        p = float(np.mean(np.abs(signs@diff / len(diff)) >= abs(delta)-1e-12)) if signs is not None else np.nan
        try:
            r0=float(pivot['raw'].mean())
            pct=float(100*delta/r0) if abs(r0)>1e-12 else float('nan')
        except Exception:
            pct=float('nan')
        effects.append({'comparison':f'{model} híbrido - raw','n_seeds':len(diff),
             'delta_mean':delta,'delta_median':float(np.median(diff)),
             'relative_improvement_pct':pct,'p_signflip_twosided_exact':p,
             'limitation':'n=5 implica mínimo p bilateral exacto 0.0625' if len(diff)==5 else 'n de semillas reducido',
             'decision':'Descriptivo multisemilla; contrastar incertidumbre por usuario; no declarar significancia automática'})
    pd.DataFrame(effects).to_csv(out/'ablacion_efectos.csv',index=False)
    if buds:pd.concat(buds,ignore_index=True).to_csv(out/'precision_at_k_por_semilla.csv',index=False)
    if configs:pd.concat(configs,ignore_index=True).to_csv(out/'configuraciones_seleccionadas.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,5))
    for model, g in desc.groupby('model'):
        g=g.set_index('feature_set')
        if 'raw' in g.index and 'raw+individualizadas' in g.index:
            x=['raw','raw+individualizadas']; y=g.loc[x,'media'].to_numpy()
            ax.plot(x,y,marker='o',label=model)
    ax.set_ylabel('PR-AUC medio');ax.set_title(f'Ablación ({n} semillas, {info[0]["n_users"]} usuarios; {info[0]["tuning_profile"]})')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'ablacion_pr_auc.png',dpi=160);plt.close(fig)
    status={'n_seeds_complete':n,'n_users_per_seed':info[0]['n_users'],'tuning_profile':info[0]['tuning_profile'],'bootstrap_iterations':info[0]['bootstrap_iterations'],'same_conditions':consistency,'is_definitive':bool(n==5 and consistency and info[0]['n_users']>=500 and info[0]['tuning_profile']=='full' and info[0]['bootstrap_iterations']>=1000),'note':'Definitivo solo con escala preregistrada y desviación documentada si <1000 usuarios; p bilateral exacto de 5 semillas no puede ser <.05. No inferir con is_definitive=false.'}
    (out/'estado_analisis.json').write_text(json.dumps(status,indent=2,ensure_ascii=False))
    print(status)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();analyze(a.input,a.output)
