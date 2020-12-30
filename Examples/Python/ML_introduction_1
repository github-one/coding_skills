```Python
#Lesson-01

from sklearn.datasets import load_boston
data = load_boston()
import pandas as pd
dataframe = pd.DataFrame(data['data'])
dataframe.columns = data['feature_names']
dataframe['price'] = data['target']
import seaborn as sns
%matplotlib inline
sns.heatmap(dataframe.corr(), annot=True, fmt='.1f')

import matplotlib.pyplot as plt
plt.scatter(dataframe['RM'], dataframe['price'])

x = dataframe['RM']
y = dataframe['price']
history_notes = {_x : _y for _x, _y in zip(x, y)}
history_notes[6.57]

similary_ys = [y for _, y in sorted(history_notes.items(), key=lambda x_y: (x_y[0] - 6.57) ** 2)[:3]]
import numpy as np
np.mean(similary_ys)

def knn(query_x, history, top_n=3):
    sorted_notes = sorted(history.items(), key=lambda x_y: (x_y[0] - query_x) ** 2) 
    similar_notes = sorted_notes[:top_n]
    similar_ys = [y for _, y in similar_notes]
    
    return np.mean(similar_ys)
knn(5.4, history_notes)

def loss(y_hat, y):
    return np.mean((y_hat - y) ** 2)
import random
min_loss = float('inf')
best_k, bes_b = None, None

for step in range(1000):
    min_v, max_v = -100, 100
    k, b = random.randrange(min_v, max_v), random.randrange(min_v, max_v)
    y_hats = [k * rm_i  + b for rm_i in x]
    current_loss = loss(y_hats, y)
    
    if current_loss < min_loss:
        min_loss = current_loss
        best_k, best_b = k, b
        print('在第{}步，我们获得了函数 f(rm) = {} * rm + {}, 此时loss是: {}'.format(step, k, b, current_loss))
        
plt.scatter(x, y)
plt.scatter(x, [best_k * rm + best_b for rm in x])

def partial_k(k, b, x, y):
    return 2 * np.mean((k * x + b - y) * x)

def partial_b(k, b, x, y):
    return 2 * np.mean(k * x + b - y)

k, b = random.random(), random.random()
min_loss = float('inf')
best_k, bes_b = None, None
learning_rate = 1e-2

for step in range(2000):
    k, b = k + (-1 * partial_k(k, b, x, y) * learning_rate), b + (-1 * partial_b(k, b, x, y) * learning_rate)
    y_hats = k * x + b
    current_loss = loss(y_hats, y)
    
    if current_loss < min_loss:
        min_loss = current_loss
        best_k, best_b = k, b
        print('在第{}步，我们获得了函数 f(rm) = {} * rm + {}, 此时loss是: {}'.format(step, k, b, current_loss))
        
        
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
sub_x = np.linspace(-10, 10)
plt.plot(sub_x, sigmoid(sub_x))

def random_linear(x):
    k, b = random.random(), random.random()
    return k * x + b
def complex_function(x):
    return (random_linear(x))
for _ in range(10):
    index = random.randrange(0, len(sub_x))
    sub_x_1, sub_x_2 = sub_x[:index], sub_x[index:]
    new_y = np.concatenate((complex_function(sub_x_1), complex_function(sub_x_2)))
    plt.plot(sub_x, new_y)
    
    
```
