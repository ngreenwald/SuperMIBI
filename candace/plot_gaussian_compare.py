import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/home/ubuntu/candace/gaussian_compare.csv', index_col=0)

mae_train = df[['gaussian_train_mae', 'epoch50_train_mae']]
mae_val = df[['gaussian_val_mae', 'epoch50_val_mae']]
mse_train = df[['gaussian_train_mse', 'epoch50_train_mse']]
mse_val = df[['gaussian_val_mse', 'epoch50_val_mse']]


## MAE
fig = plt.figure(figsize=(30,10))
# Plot the points
x = [0,1,3,4,6,7,9,10,12,13,15,16,18,19,21,22,24,25,27,28,30,31,33,34,36,37,39,40,42,43]
y_train = mae_train.values.flatten().tolist()
y_val = mae_val.values.flatten().tolist()
plt.scatter(x, y_train, color='dodgerblue', label="Training")
plt.scatter(x, y_val, color='slategray', label="Validation")

# Plot the lines
for i in range(0,len(x),2):
    a = x[i]
    b = a + 1
    plt.plot([a,b], [y_train[i],y_train[i+1]], color='dodgerblue')
    plt.plot([a,b], [y_val[i],y_val[i+1]], color='slategray')

plt.xticks([b+0.5 for b in list(range(0,43,3))], list(df.index))

plt.text(x[0]*1.01, y_train[0]*1.01, "Gaussian")
plt.text(x[1]*1.01, y_train[1]*1.01, "Model")
plt.ylabel('MAE')
plt.legend()
plt.title("Gaussian smoothing vs 50 epoch models, MAE")
plt.savefig('/home/ubuntu/candace/plots/model_metrics/gaussian_compare_mae.pdf')


## MSE
fig = plt.figure(figsize=(30,10))
# Plot the points
x = [0,1,3,4,6,7,9,10,12,13,15,16,18,19,21,22,24,25,27,28,30,31,33,34,36,37,39,40,42,43]
y_train = mse_train.values.flatten().tolist()
y_val = mse_val.values.flatten().tolist()
plt.scatter(x, y_train, color='dodgerblue', label="Training")
plt.scatter(x, y_val, color='slategray', label="Validation")

# Plot the lines
for i in range(0,len(x),2):
    a = x[i]
    b = a + 1
    plt.plot([a,b], [y_train[i],y_train[i+1]], color='dodgerblue')
    plt.plot([a,b], [y_val[i],y_val[i+1]], color='slategray')

plt.xticks([b+0.5 for b in list(range(0,43,3))], list(df.index))

plt.text(x[0]*1.01, y_train[0]*1.01, "Gaussian")
plt.text(x[1]*1.01, y_train[1]*1.01, "Model")
plt.ylabel('MSE')
plt.legend()
plt.title("Gaussian smoothing vs 50 epoch models, MSE")
plt.savefig('/home/ubuntu/candace/plots/model_metrics/gaussian_compare_mse.pdf')


