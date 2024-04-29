#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable

from tqdm import tqdm

#%% 指定device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using gpu!' if torch.cuda.is_available() else 'Using cpu!')

# device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
# print('Using mps!' if torch.backends.mps.is_built() else 'Using cpu!')

#%%# Load data
file_name = "data/chengdu_directed_shortest_distance_matrix.npy"
sdm = np.load(file_name)
maxLengthy = np.max(sdm)
sdm = sdm/maxLengthy
num = 1
n= sdm.shape[0] # 节点数目

#%%
def get_node(index):
    node1_index = int(index / (n-1))
    node2_index = index % (n-1)
    return node1_index, node2_index

def get_batch(index_list):
    l = len(index_list)
    x1_batch = np.zeros((l, n))
    x2_batch = np.zeros((l, n))
    y_batch = np.zeros((l, 1))
    z = 0
    for i in index_list:
        node1, node2 = get_node(i)
        if node2 >= node1:
            node2 += 1
        x1_batch[z][node1] = 1
        x2_batch[z][node2] = 1
        y_batch[z] = sdm[node1][node2]
        z += 1
    return x1_batch, x2_batch, y_batch

#%%# Parameters
# Network Parameters
n_hidden_1 = int(n*0.2)
n_hidden_2 = 100
n_hidden_3 = 20
n_input = n
n_output = 1
s = 2
r = 3

#%% torch MLP
class MultiLayerPerceptron(nn.Module):
    def __init__(self, n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1*2, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.fc4 = nn.Linear(n_hidden_3, n_output)

    def forward(self, x1, x2):
        layer_11 = F.relu(self.fc1(x1))
        layer_12 = F.relu(self.fc1(x2))
        layer_1 = torch.cat((layer_11, layer_12), 1)
        layer_2 = F.relu(self.fc2(layer_1))
        layer_3 = F.relu(self.fc3(layer_2))
        out_layer = torch.sigmoid(self.fc4(layer_3))
        # out_layer_2 = (torch.mean(torch.abs(out_layer[:, 5:5+s] - out_layer[:, 0:s]), -1, keepdims=True) * s + torch.mean((out_layer[:, 5+s:] - out_layer[:, s:5]), -1, keepdims=True) * r)/5
        return out_layer

model = MultiLayerPerceptron(n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output).to(device)

#%%
learning_rate = 0.01
training_epochs = 20
batch_size = n
display_step = 1
input_l = (n - 1) * n
total_batch = int(input_l/batch_size) + 1
print("total_batch:", total_batch)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# 初始化参数
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)

#%%
loss_min = 100
start_time = time.time()
# Training cycle
for epoch in range(training_epochs):
    avg_cost = 0.
    random_index = np.random.permutation(input_l)
    for j in tqdm(range(total_batch-1)):
        start = j * batch_size
        end = (j + 1) * batch_size
        if end >= input_l:
            end = input_l - 1
        batch_x1, batch_x2, batch_y = get_batch(random_index[start:end])
        batch_x1 = Variable(torch.from_numpy(batch_x1).float()).to(device)
        batch_x2 = Variable(torch.from_numpy(batch_x2).float()).to(device)
        batch_y = Variable(torch.from_numpy(batch_y).float()).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(batch_x1, batch_x2)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        # Compute average loss
        avg_cost += loss.data.item() / total_batch

    # 存储模型
    if avg_cost < loss_min:
        loss_min = avg_cost
        torch.save(model.state_dict(), "param/vdist2vec_model.ckpt")

    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

print("Optimization Finished!")

end_time = time.time()
print("Training time: ", end_time-start_time)

#%% 读取model
model = MultiLayerPerceptron(n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output).to(device)
model.load_state_dict(torch.load("param/vdist2vec_model.ckpt"))

#%%
# evaluation
def get_eval_batch(p1, p2):
    x1_batch = np.zeros(((p2-p1),n))
    x2_batch = np.zeros(((p2-p1),n))
    y_batch = np.zeros(((p2-p1),))
    z = 0
    for j in range(p1, p2):
        node1, node2 = get_node(j)
        if node2 >= node1:
            node2 += 1
        x1_batch[z][node1] = 1
        x2_batch[z][node2] = 1
        y_batch[z] = sdm[node1][node2]
        z += 1
    return x1_batch, x2_batch, y_batch

batch_size = 10000
total_batch = int(input_l/batch_size) + 1
result = []
real_dis = []

start_time = time.time()
for i in range(total_batch-1):
    start = i * batch_size
    end = (i+1)*batch_size
    if end >= input_l:
        end = input_l
    batch_x1, batch_x2, batch_y = get_eval_batch(start, end)
    batch_x1 = Variable(torch.from_numpy(batch_x1).float()).to(device)
    batch_x2 = Variable(torch.from_numpy(batch_x2).float()).to(device)
    batch_y = Variable(torch.from_numpy(batch_y).float()).to(device)
    result_temp = model(batch_x1, batch_x2).detach().cpu().numpy().reshape(-1)
    result = np.append(result, result_temp)
    real_dis = np.append(real_dis, batch_y.detach().cpu().numpy().reshape(-1))
end_time = time.time()
print("Test time:", end_time-start_time)

real_dis = real_dis * maxLengthy
result = result * maxLengthy

abe = np.fabs(real_dis - result)
re = abe/real_dis

mse = (abe ** 2).mean()
maxe = np.max(abe ** 2)
mine = np.min(abe ** 2)
mabe = abe.mean()
maxae = np.max(abe)
minae = np.min(abe)
mre = re.mean()
maxre = np.max(re)
minre = np.min(re)
print ("mean square error:", mse)
print ("max square error:", maxe)
print ("min square error:", mine)
print ("mean absolute error:", mabe)
print ("max absolute error:", maxae)
print ("min absolute error:", minae)
print ("mean relative error:", mre)
print ("max relative error:", maxre)
print ("min relative error:", minre)

#%% 绘制re数据的分布情况
import matplotlib.pyplot as plt

arange = (0, 0.2)

# 创建直方图
plt.hist(re, bins=100, alpha=0.5, color='g', range = arange)
plt.savefig('figure/vdist2vec_distribution.pdf')







