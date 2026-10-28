import torch

x=torch.arange(4.0)
print(x)

x.requires_grad_(True)  # 设置为需要梯度

y=2*torch.dot(x,x)  # y=x^2
print(y)

y.backward()  # 反向传播
print(x.grad)  # dy/dx=4x

x.grad.zero_()  # 梯度清零
y=x.sum()
y.backward()
print(x.grad)  # dy/dx=1

x.grad.zero_()  # 梯度清零
y=x*x
print(y)
y.sum().backward()  # y.sum()对x求导
print(x.grad)  # dy/dx=2x

x.grad.zero_()  # 梯度清零
y=x*x
u=y.detach()  # 分离计算图
z=u*x
z.sum().backward()  # z.sum()对x求导
print(x.grad)  # dy/dx=u= x^2

 