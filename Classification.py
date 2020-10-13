import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


''' ************
classification data
************** '''
'''n_data = torch.ones(100,2)
#数据产生
x0 = torch.normal(2*n_data, 1)
#数据标签为0
y0 = torch.zero(100)
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100)
x = torch.cat((x0,x1),0).type(torch.FloatTensor) #FloatTensor = 32-bit floating
y = tirch.cat((y0,y1),0).type(torch.FloatTensor) #LongTensor = 64-bit integer'''


''' ************
regression data
************** '''
#pytorch只能处理二位数据，作为一个张量进行处理，不能处理一维数据
x = torch.unsqueeze(torch.linspace(-1,1,100), dim=1)
#pow() 方法返回 xy（x 的 y 次方） 的值。 x的2次方加噪点
y = x.pow(2) + 0.2*torch.rand(x.size())
# x = x.numpy()
# x = x.ravel()
# print(x)

#神经网络只能处理Variable
x, y = Variable(x), Variable(y)

#数据展示，scatter散点图
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.scatter(x.data.numpy()[:0], x.data.numpy()[:,1], c = y.data.numpy(), s=100, lw=0)
# plt.show()

#torch.nn.Module Net继承这个模块
class Net(torch.nn.Module):
    #搭建net基础信息
    def __init__(self, n_feature, n_hidden, n_output):
        #继承net，官方步骤
        super (Net, self).__init__()
        #命名为hidden
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)


    #向前传导,x为输入信息
    def forward(self,x):
        # x为n_feature, 输出n_hidden, 用于激活函数,嵌套处隐藏层输出信息
        x = F.relu(self.hidden(x))
        #输出层
        x = self.predict(x)
        return x
    
#2 input 10 nural 2 type eg:[1,0][0,1][1,0,0][0,0,1]
net = Net(n_feature=1, n_hidden=10, n_output=1) 
print (net)

#mehod 2 快速搭建
net2 = torch.nn.Sequential(
    torch.nn.Linear(2,10)
    torch.nn.ReLU(),
    torch.nn.Linear(10.2)
)

#设置为实时打印过程
plt.ion()
plt.show()

#优化神经网络参数，SGD;获取net全部参数，小于1的学习效率
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
#MSR均方差
loss_func = torch.nn.MSELoss()
# loss_func = torch.nn.CrossEntropyLoss()    #此方式用于分类计算概率与标签的误差 [MSELoss用于逻辑回归]

for t in range(100):
    prediction = net(x)

    #计算误差
    loss = loss_func(prediction,y)

    #每次将之前算的梯度清零
    #ps:为解决局部最优问题，可采用模拟退火算法
    optimizer.zero_grad()
    #反向传递，计算梯度
    loss.backward()
    #优化梯度 
    optimizer.step()

#每5次打印一次
    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
       #打印学习情况
        plt.plot(x.data.numpy(),prediction.data.numpy(), 'r-', lw=5)
       #打印出误差多少
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict = {'size':20, 'color':'red'})
        plt.pause(0.1)

        # 分类画图
        plt.cla()
        prediction = torch.max(F.softmax(prediction), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200.  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)


plt.ioff()
plt.show()

