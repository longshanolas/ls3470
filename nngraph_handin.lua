
require 'nngraph'

--We want to compute z=x1+x2.*linear(x3)

x1=nn.Identity()()
x2=nn.Identity()()
x3=nn.Linear(6,4)() --a linear layer with input dim 6, output dim 4
a=nn.CMulTable()({x2,x3}) --perform element-wise tensor multiplication within a table
b=nn.CAddTable()({x1,a}) --perform table addition
m=nn.gModule({x1,x2,x3},{b}) --takes input and gives output

--manual check
i1=torch.rand(5,4) --i1 and i2 agree in dim,
i2=torch.rand(5,4)
i3=torch.rand(5,6)

--true output should be
mout = torch.add(i1,torch.cmul(i2,x3.data.module:forward(i3)))
--gmodel output should be
gout = m:forward({i1,i2,i3})
--check if mout==gout
