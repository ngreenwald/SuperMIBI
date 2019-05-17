# python plot_model_metrics.py [name of model]

import matplotlib.pyplot as plt
import json
import sys

name = sys.argv[1]
f = '/home/ubuntu/candace/models/' + name + '/history.json'
plot_basedir = '/home/ubuntu/candace/plots/model_metrics/'

## Fix JSON file if necessary
# Remove first and last quotes, then replace single quotes with double quotes
f1 = open(f,'r')
f2 = open(f+'.tmp','w')
for line in f1:
    f2.write(line.replace('"',"").replace("'",'"'))
f1.close()
f2.close()

## Convert JSON to dict
def js_r(filename):
    with open(filename) as f_in:
        return(json.load(f_in))

dat = js_r(f+'.tmp')

plt.figure(figsize=(20,20))
plt.gcf().text(0.43, 0.9, name, fontsize=20)

plt.subplot(2, 2, 1)
plt.xlabel('Epoch')
plt.ylabel('Mean Abs Error')
plt.plot(range(1,len(dat['mae'])+1), dat['mae'], label='Train Error')
plt.plot(range(1,len(dat['val_mae'])+1), dat['val_mae'], label='Val Error')
plt.legend()

plt.subplot(2, 2, 2)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.plot(range(1,len(dat['mse'])+1), dat['mse'], label='Train Error')
plt.plot(range(1,len(dat['val_mse'])+1), dat['val_mse'], label='Val Error')
plt.legend()

plt.subplot(2, 2, 3)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(1,len(dat['loss'])+1), dat['loss'], label='Train')
plt.plot(range(1,len(dat['val_loss'])+1), dat['val_loss'], label='Val')
plt.legend()

plt.subplot(2, 2, 4)
plt.xlabel('Epoch')
plt.ylabel('Learning rate')
plt.plot(range(1,len(dat['lr'])+1), dat['lr'])

plt.savefig(plot_basedir + name + '_metrics.pdf')

