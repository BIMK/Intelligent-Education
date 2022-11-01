
import matplotlib.pyplot as plt


plt.ylabel("Time(s)")
x = ['LightGCN','BPR-MF','SGL','PD-GAN', 'PMOEA', 'MOEA-Probs','MORS','EMMR']
y = [2274,511,1198, 4312,18487,22212,4097, 5888]
fig = plt.figure()

ax1 = fig.add_subplot(211)
for tick in ax1.get_xticklabels():  # 将横坐标倾斜30度，纵坐标可用相同方法
    tick.set_rotation(30)
plt.ylabel("Time(s)")
plt.bar(x, y,width = 0.5,color  = "#00BFFF" )
plt.show()
