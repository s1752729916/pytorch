import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
dropout = 0.0
def ForeProcess():#数据预处理
    data = pd.read_csv("C:\\SIMULI\\abaqus2.csv",encoding = 'gb18030')
    valid = data[data["作业状态"] == "COMPLETED"]#筛选所有有效数据
    valid = valid.drop(["作业名称","小球材料名称","平板材料名称","作业状态","塑性应力1","塑性应力2","塑性应力3","塑性应力1.1","塑性应力2.1","塑性应力3.1","塑性应变1","塑性应变2","塑性应变3","塑性应变1.1","塑性应变2.1","塑性应变3.1"],axis= 1)
    valid_np = valid.values#将dataFrame转换成numpy
    #valid_np = valid_np[:,0:24]#截取有效数据
    #print(valid_np[:,1:(valid_np.shape[1]-6)])#6个输出之前的所有都是参数
    #print(valid_np[:,(valid_np.shape[1]-6):valid_np.shape[1]])#6个输出之前的所有都是参数
    #print(valid_np[0].shape)
    print(valid_np[0])
    return valid_np

class Norm():
    def __init__(self,X_Raw):
        self.X_Raw  = X_Raw
        self.X_norm = np.zeros((2, self.X_Raw.shape[1]))
        self.max = 0
        self.min = 0
    def Normlize(self):

        X_new = np.zeros(self.X_Raw.shape)#创建一个和X相同大小的矩阵
        self.X_norm = np.zeros((2,self.X_Raw.shape[1]))#存放max和min值，用来反归一化
        self.max = np.max(self.X_Raw[:,0])
        self.min = np.min(self.X_Raw[:,0])
        for i in range(self.X_Raw.shape[1]):
            max = np.max(self.X_Raw[:,i])
            min = np.min(self.X_Raw[:,i])#找第i列的最大值和最小值
            X_new[:,i] = (self.X_Raw[:,i] - min)/(max-min)#归一化
            self.X_norm[0:i] =  min
            self.X_norm[1:i] = max-min
        return X_new
    def DeNormlize(self,x):

        results = np.ones(x.shape[0]).reshape(x.shape)
        results = torch.from_numpy(results)
        for i in range(x.shape[0]):
            results[i]=(x[i]*(self.max-self.min)+self.min)
        return results

        #只有标签值需要反归一化


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #4层隐藏层
        self.InLayer = nn.Linear(7,32)#输入层
        self.H1 =nn.Linear(32,64)
        self.H2 = nn.Linear(64,32)
        self.H3 = nn.Linear(32,16)
        self.H4 = nn.Linear(16,4)
        self.OutLayer = nn.Linear(4,1)
    def forward(self, x):
        x = torch.tanh(self.InLayer(x))
        nn.Dropout(dropout)
        x = torch.tanh(self.H1(x))
        nn.Dropout(dropout)
        x = torch.tanh(self.H2(x))
        nn.Dropout(dropout)
        x = torch.tanh(self.H3(x))
        nn.Dropout(dropout)
        x = torch.tanh(self.H4(x))
        x = self.OutLayer(x)
        return x


norm_data = ForeProcess()
parameters=norm_data[:,0:(norm_data.shape[1]-6)]#6个输出之前的所有都是参数
outputs = norm_data[:,(norm_data.shape[1]-6):norm_data.shape[1]]#将6个输出量保存下来
#############################
#回归目标：小球最大应变,合并数据
#############################

#print(outputs[0])
dataset=np.column_stack(( outputs[:,0],parameters)) #参数+小球最大应变,如果回归目标改变仅需改变outputs的列数即可,label值在第一列[0]
x = Norm(dataset)
dataset = x.Normlize()
train = dataset[0:800]#训练集80%
test = dataset[800:]#测试集
#print(dataset.shape)
#print(train.shape)
#print(test.shape)


#############################
#下面对网络进行训练
#############################
epoch = 100
batch_Size = 4
net = Net()#创建网络
LearnRate = 0.001#学习率
L2 = 0.0 #L2正则化

weight_p, bias_p = [],[]
for name, p in net.named_parameters():
    if 'bias' in name:
        bias_p += [p]
    else:
        weight_p += [p]
# 这里的model中每个参数的名字都是系统自动命名的，只要是权值都是带有weight，偏置都带有bias，
# 因此可以通过名字判断属性，这个和tensorflow不同，tensorflow是可以用户自己定义名字的，当然也会系统自己定义。


criterion = nn.MSELoss()#损失函数 平方差
optimizer = optim.Adam([
          {'params': weight_p, 'weight_decay':L2},
          {'params': bias_p, 'weight_decay':0}
          ],lr = LearnRate,)#Adam优化器





train = DataLoader(train,batch_size=batch_Size,shuffle=False,num_workers=0)#将数据集存入DataLoader中
test = DataLoader(test,batch_size=batch_Size,shuffle=False,num_workers=0)
plt_run_loss = []
plt_test_loss =[]
plt_k =[]
plt_test_label= []
plt_test_output = []
for k in range(epoch):
    run_loss = 0#训练集的损失函数
    test_loss= 0
    for i,data in enumerate(train,0):
        labels,inputs = data[:,0],data[:,1:]

        optimizer.zero_grad()
        inputs = inputs.float()
        labels = labels.float()
        ####################向量转矩阵
        labels = labels.numpy()
        labels = labels.reshape(labels.shape[0],1)
        labels  = torch.from_numpy(labels)
        ####################向量转矩阵

        outputs = net(inputs)#得到结果
        loss = criterion(outputs,labels)#计算损失函数
        loss.backward()#误差反向传播
        optimizer.step()#优化
        run_loss +=loss.item()
    for i, data in enumerate(test, 0):
        labels, inputs = data[:, 0], data[:, 1:]
        optimizer.zero_grad()
        inputs = inputs.float()
        labels = labels.float()
        ####################向量转矩阵
        labels = labels.numpy()
        labels = labels.reshape(labels.shape[0], 1)
        labels = torch.from_numpy(labels)
        ###################将向量转换为矩阵
        outputs = net(inputs)  # 得到结果
        loss = criterion(outputs, labels)  # 计算损失函数
        loss.backward()  # 误差反向传播
        test_loss += loss.item()
    plt_k.append(k)
    plt_run_loss.append(run_loss)
    plt_test_loss.append(test_loss)
    print(k,run_loss,test_loss)

##验证是否正确
inputs = test.dataset[:,1:]
labels= test.dataset[:,0]
outputs = net(torch.from_numpy(inputs).float())
print(outputs)

outputs = x.DeNormlize(outputs)
labels = x.DeNormlize(labels)
print(outputs)
a = []
b = []
for i in range(outputs.shape[0]):
    a.append(labels[i])
    b.append(outputs[i]-labels[i])
print(a)
print(b)
fig = plt.figure(figsize=(10,6),facecolor='gray')
ax1 = fig.add_subplot(1,2,1)  # 第一行的左图
ax2 = fig.add_subplot(1,2,2)  # 第一行的左图
ax1.plot(plt_test_loss)
ax1.plot(plt_run_loss)
plt.title("lr:"+str(LearnRate)+" epoch:"+str(epoch)+" batchSize:"+str(batch_Size)+ "  DropOut:"+str(dropout)+" L2 Regularization:"+str(L2))
ax2.plot(b,'o')

print(233)
plt.show()




