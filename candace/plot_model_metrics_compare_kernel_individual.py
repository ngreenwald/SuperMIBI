# Plot metrics for multiple models
# Compare kernel size 3 and size 7
# Plot each model in individual plot

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

for i in range(len(models)):

    p_mae = plt.figure(num=0,figsize=(20,20))
    p_mse = plt.figure(num=1,figsize=(20,20))
    p_loss = plt.figure(num=2,figsize=(20,20))
    p_lr = plt.figure(num=3,figsize=(20,20))

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

    plt.figure(0)
    p1_mae = plt.subplot(4,3,i+1)
    p1_mae.plot(range(1,len(dat['mae'])+1), dat['mae'], color='dodgerblue')
    p1_mae.plot(range(1,len(dat['val_mae'])+1), dat['val_mae'], color='dodgerblue',linestyle='--')
    p1_mae.plot(range(1,len(dat_7['mae'])+1), dat_7['mae'], color='slategray')
    p1_mae.plot(range(1,len(dat_7['val_mae'])+1), dat_7['val_mae'], color='slategray', linestyle='--')
    plt.title(m)

    plt.figure(1)
    p1_mse = plt.subplot(4,3,i+1)
    p1_mse.plot(range(1,len(dat['mse'])+1), dat['mse'], color='dodgerblue')
    p1_mse.plot(range(1,len(dat['val_mse'])+1), dat['val_mse'], color='dodgerblue', linestyle='--')
    p1_mse.plot(range(1,len(dat_7['mse'])+1), dat_7['mse'], color='slategray')
    p1_mse.plot(range(1,len(dat_7['val_mse'])+1), dat_7['val_mse'], color='slategray', linestyle='--')
    plt.title(m)

    plt.figure(2)
    p1_loss = plt.subplot(4,3,i+1)
    p1_loss.plot(range(1,len(dat['loss'])+1), dat['loss'], color='dodgerblue')
    p1_loss.plot(range(1,len(dat['val_loss'])+1), dat['val_loss'], color='dodgerblue', linestyle='--')
    p1_loss.plot(range(1,len(dat_7['loss'])+1), dat_7['loss'], color='slategray')
    p1_loss.plot(range(1,len(dat_7['val_loss'])+1), dat_7['val_loss'], color='slategray', linestyle='--')
    plt.title(m)

    plt.figure(3)
    p1_lr = plt.subplot(4,3,i+1)
    p1_lr.plot(range(1,len(dat['lr'])+1), dat['lr'], color='dodgerblue')
    p1_lr.plot(range(1,len(dat_7['lr'])+1), dat_7['lr'], color='slategray')
    plt.title(m)


# Set up legend
points = np.ones(5)
text = ['kernel7 val', 'kernel7 train', 'kernel3 val', 'kernel3 train']
col = ['slategray', 'slategray', 'dodgerblue', 'dodgerblue']
styles = ['--', '-', '--', '-']

plt.figure(0)
plt.gcf().text(0.43, 0.9, "MAE over 50 epochs", fontsize=20)
leg = plt.subplot(4,3,12)
for i in range(len(text)):
    leg.text(-0.1, i, text[i], horizontalalignment='right', verticalalignment='center')
    leg.plot(i*points, linestyle=styles[i], color=col[i])
leg.margins(0.2)
leg.set_axis_off()
plt.savefig(plot_basedir+'kernel3vs7_mae.pdf')

plt.figure(1)
plt.gcf().text(0.43, 0.9, "MSE over 50 epochs", fontsize=20)
leg = plt.subplot(4,3,12)
for i in range(len(text)):
    leg.text(-0.1, i, text[i], horizontalalignment='right', verticalalignment='center')
    leg.plot(i*points, linestyle=styles[i], color=col[i])
leg.margins(0.2)
leg.set_axis_off()
plt.savefig(plot_basedir+'kernel3vs7_mse.pdf')

plt.figure(2)
plt.gcf().text(0.43, 0.9, "Loss over 50 epochs", fontsize=20)
leg = plt.subplot(4,3,12)
for i in range(len(text)):
    leg.text(-0.1, i, text[i], horizontalalignment='right', verticalalignment='center')
    leg.plot(i*points, linestyle=styles[i], color=col[i])
leg.margins(0.2)
leg.set_axis_off()
plt.savefig(plot_basedir+'kernel3vs7_loss.pdf')

plt.figure(3)
plt.gcf().text(0.43, 0.9, "LR over 50 epochs", fontsize=20)
leg = plt.subplot(4,3,12)
text = ['kernel7', 'kernel3']
col = ['slategray', 'dodgerblue']
for i in range(len(text)):
    leg.text(-0.1, i, text[i], horizontalalignment='right', verticalalignment='center')
    leg.plot(i*points, color=col[i])
leg.margins(0.2)
leg.set_axis_off()
plt.savefig(plot_basedir+'kernel3vs7_lr.pdf')


#### Plot one model PER pdf
#for i in range(len(models)):
#
#    plt.figure(figsize=(20,20))
#
#    p1 = plt.subplot(2, 2, 1)
#    plt.xlabel('Epoch')
#    plt.ylabel('Mean Abs Error')
#
#    p2 = plt.subplot(2, 2, 2)
#    plt.xlabel('Epoch')
#    plt.ylabel('Mean Squared Error')
#
#    p3 = plt.subplot(2, 2, 3)
#    plt.xlabel('Epoch')
#    plt.ylabel('Loss')
#
#    p4 = plt.subplot(2, 2, 4)
#    plt.xlabel('Epoch')
#    plt.ylabel('Learning rate')
#
#    ### Kernel 3 model
#    m = models[i]
#    f = '/home/ubuntu/candace/models/' + m + '/history.json'
#    ## Fix JSON file if necessary
#    # Remove first and last quotes, then replace single quotes with double quotes
#    f1 = open(f,'r')
#    f2 = open(f+'.tmp','w')
#    for line in f1:
#        f2.write(line.replace('"',"").replace("'",'"'))
#    f1.close()
#    f2.close()
#    dat = js_r(f+'.tmp')
#
#    ### Kernel 3 model
#    m_7 = models_7[i]
#    f_7 = '/home/ubuntu/candace/models/' + m_7 + '/history.json'
#    ## Fix JSON file if necessary
#    # Remove first and last quotes, then replace single quotes with double quotes
#    f1_7 = open(f_7,'r')
#    f2_7 = open(f_7+'.tmp','w')
#    for line in f1_7:
#        f2_7.write(line.replace('"',"").replace("'",'"'))
#    f1_7.close()
#    f2_7.close()
#    dat_7 = js_r(f_7+'.tmp')
#
#
#    p1.plot(range(1,len(dat['mae'])+1), dat['mae'], color='dodgerblue')
#    p1.plot(range(1,len(dat['val_mae'])+1), dat['val_mae'], color='dodgerblue',linestyle='--')
#
#    p1.plot(range(1,len(dat_7['mae'])+1), dat_7['mae'], color='slategray')
#    p1.plot(range(1,len(dat_7['val_mae'])+1), dat_7['val_mae'], color='slategray', linestyle='--')
#
#    p2.plot(range(1,len(dat['mse'])+1), dat['mse'], label='kernel3_train', color='dodgerblue')
#    p2.plot(range(1,len(dat['val_mse'])+1), dat['val_mse'], label='kernel3_val', color='dodgerblue', linestyle='--')
#
#    p2.plot(range(1,len(dat_7['mse'])+1), dat_7['mse'], label='kernel7_train', color='slategray')
#    p2.plot(range(1,len(dat_7['val_mse'])+1), dat_7['val_mse'], label='kernel7_val', color='slategray', linestyle='--')
#
#    p3.plot(range(1,len(dat['loss'])+1), dat['loss'], label=m, color='dodgerblue')
#    p3.plot(range(1,len(dat['val_loss'])+1), dat['val_loss'], color='dodgerblue', linestyle='--')
#
#    p3.plot(range(1,len(dat_7['loss'])+1), dat_7['loss'], color='slategray')
#    p3.plot(range(1,len(dat_7['val_loss'])+1), dat_7['val_loss'], color='slategray', linestyle='--')
#
#    p4.plot(range(1,len(dat['lr'])+1), dat['lr'], label=m, color='dodgerblue')
#    p4.plot(range(1,len(dat_7['lr'])+1), dat_7['lr'], color='slategray')
#
#    p2.legend()
#    plt.gcf().text(0.43, 0.9, m, fontsize=20)
#    plt.savefig(plot_basedir+'kernel3vs7_'+m+'.pdf')
#
