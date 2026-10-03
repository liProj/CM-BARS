from pathlib import Path
import sys,json,re
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
root=Path(__file__).resolve().parent;src=root.parent
for name in ['figures']: (root/name).mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Arial','axes.unicode_minus':False,'pdf.fonttype':42,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
colors=['#174A6E','#D87834','#288A82']
rs=['tannic_acid','p_coumaric_acid','acetosyringone'];rnames=['Tannic acid (TA)','p-Coumaric acid (PC)','Acetosyringone (AS)']
cv=pd.read_csv(root/'figure_data/cv_pooled.csv');lc=cv[cv.protocol=='LOCO'];D=pd.read_csv(root/'figure_data/bbd_design_responses.csv')
def savefig(name,fig):
 fig.savefig(root/'figures'/f'{name}.pdf',bbox_inches='tight');fig.savefig(root/'figures'/f'{name}.png',bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
for j,(r,c) in enumerate(zip(rs,colors)):
 ax[0].plot(D.run,D[r],'-o',color=c,label=rnames[j],ms=4,lw=1)
 zeros=D[D[r]==0];ax[0].scatter(zeros.run,zeros[r],marker='x',color=c,s=48,zorder=4)
 grouped=D.groupby('pH')[r]
 ax[1].errorbar(grouped.mean().index,grouped.mean(),yerr=grouped.sem(),color=c,marker='o',capsize=3,label=rnames[j])
ax[0].axvspan(12.6,15.3,color='#eef1f5');ax[0].set(xlabel='Run number (13-15 are center replicates)',ylabel='Observed adsorption (%)',title='(a) Bounded responses and zeros')
ax[1].set(xlabel='pH',ylabel='Marginal mean adsorption (%)',title='(b) Compound-specific pH trends');ax[1].legend(fontsize=8)
savefig('fig01_data',fig)
names=['OLS-Quad (published)','OLS-Quad + clip','Ridge-Quad','Lasso-Quad','ElasticNet-Quad','PLS-Quad','RandomForest','ExtraTrees','LightGBM','XGBoost','GP-Matern','Tobit-Quad','Tobit-MT-Bayes','CM-BARS','PFN-RSM (prior-fitted)']
labels={n:n for n in names};labels.update({'OLS-Quad (published)':'OLS quadratic','OLS-Quad + clip':'OLS quadratic + clipping','PFN-RSM (prior-fitted)':'PFN-RSM'})
fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
order=lc[lc.method.isin(names)].groupby('method').RMSE.mean().sort_values().index
ax.barh([labels[n] for n in order], [lc[lc.method==n].RMSE.mean() for n in order],color=[colors[1] if n in ['CM-BARS','PFN-RSM (prior-fitted)'] else colors[0] for n in order])
ax.invert_yaxis();ax.set(xlabel='Mean RMSE across compounds (percentage points; lower is better)',title='Leave-one-condition-out prediction')
savefig('fig02_rmse',fig)
methods=['OLS-Quad (published)','ExtraTrees','CM-BARS','PFN-RSM (prior-fitted)']
fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
for i,n in enumerate(methods):
 vals=lc[lc.method==n].set_index('response').loc[rs]
 axes[0].plot(range(3),vals.Coverage90,'o-',label=labels[n]);axes[1].plot(range(3),vals.IntervalWidth90,'o-',label=labels[n])
for a in axes:a.set_xticks(range(3),['TA','PC','AS'])
axes[0].axhline(.9,ls='--',color='grey');axes[0].set(ylim=(0,1.08),ylabel='Empirical coverage',title='(a) Nominal 90% predictive intervals');axes[0].legend(fontsize=8)
axes[1].set(ylabel='Mean interval width (percentage points)',title='(b) Corresponding interval widths')
savefig('fig03_intervals',fig)
variants=['CM-BARS -bounded link','CM-BARS -shrinkage','CM-BARS -coupling','CM-BARS mean-pooled','CM-BARS -zero-part','CM-BARS +GP']
vn=['Gaussian likelihood','No blockwise shrinkage','No coefficient coupling','Mean sharing','No zero component','Added GP deviation']
base=lc[lc.method=='CM-BARS'].set_index('response').loc[rs,'RMSE'].values
delta=np.array([lc[lc.method==n].set_index('response').loc[rs,'RMSE'].values-base for n in variants])
fig,ax=plt.subplots(figsize=(8.8,4.0),layout='constrained')
for j in range(3):ax.barh(np.arange(len(vn))+(j-1)*.23,delta[:,j],height=.21,color=colors[j],label=rnames[j])
ax.set_yticks(range(len(vn)),vn);ax.invert_yaxis();ax.axvline(0,color='black',lw=.7);ax.set(xlabel='Change in RMSE relative to CM-BARS (percentage points)');ax.legend(fontsize=8,loc='lower right')
savefig('fig04_ablation',fig)
hist=pd.read_csv(root/'figure_data/pfn_train_history.csv')
fig,ax=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
ax[0].plot(hist.step,hist.nll,color=colors[0]);ax[0].set(xlabel='Training steps',ylabel='Synthetic-batch negative log score',title='(a) PFN-RSM training record')
for i,n in enumerate(['CM-BARS','PFN-RSM (prior-fitted)','ExtraTrees']):
 v=lc[lc.method==n].set_index('response').loc[rs,'RMSE'].values
 ax[1].bar(np.arange(3)+(i-1)*.23,v,width=.21,color=colors[i],label=labels[n])
ax[1].set_xticks(range(3),['TA','PC','AS']);ax[1].set(ylabel='RMSE (percentage points)',title='(b) LOCO error on real data');ax[1].legend(fontsize=8)
savefig('fig05_pfn',fig)
G=np.load(root/'figure_data/cmbars_grid.npz');Z=G['Zg'];mask=np.isclose(Z[:,1],-1);idx=np.where(mask)[0]
x=4+2*(Z[idx,0]+1);y=.1+.15*(Z[idx,2]+1)
fig,ax=plt.subplots(2,3,figsize=(10,6),layout='constrained')
for j,r in enumerate(rs):
 for row,part in enumerate(['mean','sd']):
  v=G[f'cmbars_{part}_{r}'][idx];im=ax[row,j].tricontourf(x,y,v,levels=12,cmap='viridis');fig.colorbar(im,ax=ax[row,j],shrink=.8)
  ax[row,j].set(xlabel='pH',ylabel='CNF concentration (%)',title=f'{rnames[j]}\n'+('Predictive mean (%)' if row==0 else 'Predictive SD (percentage points)'))
savefig('fig06_surfaces',fig)
