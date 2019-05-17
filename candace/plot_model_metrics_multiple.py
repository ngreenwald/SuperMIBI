# Plot metrics for multiple models

import matplotlib.pyplot as plt
import json
import sys
import numpy as np

## Convert JSON to dict
def js_r(filename):
    with open(filename) as f_in:
        return(json.load(f_in))

## Plot combinations of 2 markers
#markers = ['H3K9Ac','CD45','HLA-Class-1','Ki67','CD20','Vimentin']
#models = []
#for i in range(len(markers)):
#    for j in range(i,len(markers)-1):
#        models.append('epoch50_'+markers[i]+'_'+markers[j+1])

## Plot 50 epoch markers, including 3 and 4 marker models
#models = ['epoch50_CD45_Ki67', 'epoch50_CD45_CD20', 'epoch50_CD45_Vimentin', 'epoch50_Ki67_CD20', 'epoch50_Ki67_Vimentin', 'epoch50_CD20_Vimentin', 'epoch50_ki67_cd45_vimentin', 'epoch50_ki67_cd45_cd20', 'epoch50_ki67_cd20_vimentin', 'epoch50_cd45_cd20_vimentin', 'epoch50_ki67_cd45_cd20_vimentin']

## Plot 300 epoch markers, including 3 and 4 marker models
#models = ['epoch300_CD45_Ki67', 'epoch300_CD45_CD20', 'epoch300_CD45_Vimentin', 'epoch300_Ki67_CD20', 'epoch300_Ki67_Vimentin', 'epoch300_CD20_Vimentin', 'epoch300_ki67_cd45_vimentin', 'epoch300_ki67_cd45_cd20', 'epoch300_ki67_cd20_vimentin', 'epoch300_cd45_cd20_vimentin', 'epoch300_ki67_cd45_cd20_vimentin']

## Plot 300 epoch markers, including 3 and 4 marker models, WITH 11 marker model
#models = ['epoch300_CD45_Ki67', 'epoch300_CD45_CD20', 'epoch300_CD45_Vimentin', 'epoch300_Ki67_CD20', 'epoch300_Ki67_Vimentin', 'epoch300_CD20_Vimentin', 'epoch300_ki67_cd45_vimentin', 'epoch300_ki67_cd45_cd20', 'epoch300_ki67_cd20_vimentin', 'epoch300_cd45_cd20_vimentin', 'epoch300_ki67_cd45_cd20_vimentin', 'epoch1000_11_markers']

## Plot PD1 CD45 dsDNA model
models = ['epoch300_CD45_Ki67', 'epoch300_CD45_CD20', 'epoch300_CD45_Vimentin', 'epoch300_Ki67_CD20', 'epoch300_Ki67_Vimentin', 'epoch300_CD20_Vimentin', 'epoch300_ki67_cd45_vimentin', 'epoch300_ki67_cd45_cd20', 'epoch300_ki67_cd20_vimentin', 'epoch300_cd45_cd20_vimentin', 'epoch300_ki67_cd45_cd20_vimentin', 'epoch300_pd1_cd45_dsDNA']

plot_basedir = '/home/ubuntu/candace/plots/model_metrics/'
plt.figure(figsize=(20,20))

p1 = plt.subplot(2, 2, 1)
plt.xlabel('Epoch')
plt.ylabel('Mean Abs Error')

p2 = plt.subplot(2, 2, 2)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')

p3 = plt.subplot(2, 2, 3)
plt.xlabel('Epoch')
plt.ylabel('Loss')

p4 = plt.subplot(2, 2, 4)
plt.xlabel('Epoch')
plt.ylabel('Learning rate')

cm = plt.get_cmap('tab20')
cmap = cm(np.linspace(0, 1, len(models)))

for i in range(len(models)):
    m = models[i]
    f = '/home/ubuntu/candace/models/' + m + '/history.json'

    ## Fix JSON file if necessary
    # Remove first and last quotes, then replace single quotes with double quotes
    f1 = open(f,'r')
    f2 = open(f+'.tmp','w')
    for line in f1:
        f2.write(line.replace('"',"").replace("'",'"'))
    f1.close()
    f2.close()

    dat = js_r(f+'.tmp')

    p1.plot(range(1,len(dat['mae'])+1), dat['mae'], label=m, color=cmap[i])
    p1.plot(range(1,len(dat['val_mae'])+1), dat['val_mae'], color=cmap[i], linestyle='--')
    p1.annotate(s=m.replace('epoch50_',''), xy=(range(1,len(dat['mae'])+1)[-1],dat['mae'][-1]), xytext=(5,0), textcoords='offset points', va='center', size=6)

    p2.plot(range(1,len(dat['mse'])+1), dat['mse'], label=m, color=cmap[i])
    p2.plot(range(1,len(dat['val_mse'])+1), dat['val_mse'], color=cmap[i], linestyle='--')
    p2.annotate(s=m.replace('epoch50_',''), xy=(range(1,len(dat['mse'])+1)[-1],dat['mse'][-1]), xytext=(5,0), textcoords='offset points', va='center', size=6)

    p3.plot(range(1,len(dat['loss'])+1), dat['loss'], label=m, color=cmap[i])
    p3.plot(range(1,len(dat['val_loss'])+1), dat['val_loss'], color=cmap[i], linestyle='--')
    p3.annotate(s=m.replace('epoch50_',''), xy=(range(1,len(dat['loss'])+1)[-1],dat['loss'][-1]), xytext=(5,0), textcoords='offset points', va='center', size=6)

    p4.plot(range(1,len(dat['lr'])+1), dat['lr'], label=m, color=cmap[i])
    p4.annotate(s=m.replace('epoch50_',''), xy=(range(1,len(dat['lr'])+1)[-1],dat['lr'][-1]), xytext=(5,0), textcoords='offset points', va='center', size=6)

p2.legend()

#plt.savefig(plot_basedir+'epoch50_multiple.pdf')
#plt.savefig(plot_basedir+'epoch50_multiple_channel_metrics.pdf')
#plt.savefig(plot_basedir+'epoch300_multiple_channel_metrics.pdf')
#plt.savefig(plot_basedir+'epoch300_vs_epoch1000_multiple_metrics.pdf')
plt.savefig(plot_basedir+'epoch300_pd1cd45dsDNA_metrics.pdf')

