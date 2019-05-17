# Plot metrics for multiple models
# Compare kernel size 3 and size 7

import matplotlib.pyplot as plt
import json
import sys
import numpy as np

## Convert JSON to dict
def js_r(filename):
    with open(filename) as f_in:
        return(json.load(f_in))

models = ['epoch50_CD45_Ki67', 'epoch50_CD45_CD20', 'epoch50_CD45_Vimentin', 'epoch50_Ki67_CD20', 'epoch50_Ki67_Vimentin', 'epoch50_CD20_Vimentin', 'epoch50_ki67_cd45_vimentin', 'epoch50_ki67_cd45_cd20', 'epoch50_ki67_cd20_vimentin', 'epoch50_cd45_cd20_vimentin', 'epoch50_ki67_cd45_cd20_vimentin']

models_7 = ['kernel7_'+x for x in models]

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
    ### Kernel 3 model
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

    ### Kernel 3 model
    m_7 = models_7[i]
    f_7 = '/home/ubuntu/candace/models/' + m_7 + '/history.json'
    ## Fix JSON file if necessary
    # Remove first and last quotes, then replace single quotes with double quotes
    f1_7 = open(f_7,'r')
    f2_7 = open(f_7+'.tmp','w')
    for line in f1_7:
        f2_7.write(line.replace('"',"").replace("'",'"'))
    f1_7.close()
    f2_7.close()
    dat_7 = js_r(f_7+'.tmp')


    p1.plot(range(1,len(dat['mae'])+1), dat['mae'], label=m, color=cmap[i], linewidth=2)
    p1.plot(range(1,len(dat['val_mae'])+1), dat['val_mae'], color=cmap[i], linewidth=0.5)
    p1.annotate(s=m.replace('epoch50_',''), xy=(range(1,len(dat['mae'])+1)[-1],dat['mae'][-1]), xytext=(5,0), textcoords='offset points', va='center', size=6)

    p1.plot(range(1,len(dat_7['mae'])+1), dat_7['mae'], color=cmap[i], linestyle='--', linewidth=2)
    p1.plot(range(1,len(dat_7['val_mae'])+1), dat_7['val_mae'], color=cmap[i], linestyle='--', linewidth=0.5)

    p2.plot(range(1,len(dat['mse'])+1), dat['mse'], label=m, color=cmap[i], linewidth=2)
    p2.plot(range(1,len(dat['val_mse'])+1), dat['val_mse'], color=cmap[i], linewidth=0.5)
    p2.annotate(s=m.replace('epoch50_',''), xy=(range(1,len(dat['mse'])+1)[-1],dat['mse'][-1]), xytext=(5,0), textcoords='offset points', va='center', size=6)

    p2.plot(range(1,len(dat_7['mse'])+1), dat_7['mse'], color=cmap[i], linestyle='--', linewidth=2)
    p2.plot(range(1,len(dat_7['val_mse'])+1), dat_7['val_mse'], color=cmap[i], linestyle='--', linewidth=0.5)

    p3.plot(range(1,len(dat['loss'])+1), dat['loss'], label=m, color=cmap[i], linewidth=2)
    p3.plot(range(1,len(dat['val_loss'])+1), dat['val_loss'], color=cmap[i], linewidth=0.5)
    p3.annotate(s=m.replace('epoch50_',''), xy=(range(1,len(dat['loss'])+1)[-1],dat['loss'][-1]), xytext=(5,0), textcoords='offset points', va='center', size=6)

    p3.plot(range(1,len(dat_7['loss'])+1), dat_7['loss'], color=cmap[i], linestyle='--', linewidth=2)
    p3.plot(range(1,len(dat_7['val_loss'])+1), dat_7['val_loss'], color=cmap[i], linestyle='--', linewidth=0.5)

    p4.plot(range(1,len(dat['lr'])+1), dat['lr'], label=m, color=cmap[i], linewidth=2)
    p4.annotate(s=m.replace('epoch50_',''), xy=(range(1,len(dat['lr'])+1)[-1],dat['lr'][-1]), xytext=(5,0), textcoords='offset points', va='center', size=6)

    p4.plot(range(1,len(dat_7['lr'])+1), dat_7['lr'], color=cmap[i], linestyle='--', linewidth=2)

p2.legend()

plt.savefig(plot_basedir+'kernel3vs7_multiple_metrics.pdf')

