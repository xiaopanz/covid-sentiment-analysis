import pickle 
import matplotlib.pyplot as plt
import numpy as np

def get_running_avg(data, interval=5):
    data = np.array(data)
    new_data = np.zeros((len(data) - interval))
    
    for i in range(new_data.shape[0]):
        new_data[i] = np.mean(data[i : i + interval])
    
    return new_data

fn = 'acc_list.pkl'

acc_list = pickle.load(open(fn,"rb"))

plt.plot(acc_list, color='lightblue', linewidth=2, alpha=0.75)
plt.plot(get_running_avg(acc_list), color='deepskyblue')
plt.xlabel('iteration')
plt.ylabel('accuracy (%)')
plt.savefig('accuracy_avg.png',)