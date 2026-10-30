"""Plot saved measurements without changing or filtering the experiment."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('run',type=Path)
args = parser.parse_args()
with (args.run/'training_steps.csv').open() as stream:
    rows = list(csv.DictReader(stream))
history = json.loads((args.run/'validation_history.json').read_text())
fig, axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
block = 200
points = [rows[i:i+block] for i in range(0,len(rows),block)]
axes[0].plot([int(p[-1]['step']) for p in points],
             [100*sum(float(r['x_error_m']) for r in p)/len(p) for p in points])
axes[0].set(xlabel='SAC training steps',ylabel='Mean X error (cm)',title='Training error, 200-step blocks')
for previous,current in zip(history,history[1:]):
    if current['stage'] != previous['stage']:
        boundary=previous['step']
        axes[0].axvline(boundary,color='gray',linestyle='--',alpha=.7)
        axes[0].text(boundary+100,.97,f"Stage: {[5,10,15][current['stage']]} cm",
                     transform=axes[0].get_xaxis_transform(),va='top',fontsize=9)
axes[0].text(.02,.97,'Start: 5 cm',transform=axes[0].transAxes,va='top',fontsize=9)
fig.suptitle('Training includes exploratory actions; validation uses deterministic actions',fontsize=11)
axes[1].plot([h['step'] for h in history],
             [h['full_validation']['successes']/h['full_validation']['attempts'] for h in history],marker='o')
axes[1].set(xlabel='SAC training steps',ylabel='Success rate',ylim=(-.05,1.05),title='Fixed validation scenes, all distances')
for ax in axes: ax.grid(alpha=.2)
fig.savefig(args.run/'training_curve.png',dpi=160)
