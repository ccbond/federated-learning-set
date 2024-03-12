import matplotlib.pyplot as plt
import numpy as np
  
#epoch,acc,loss,val_acc,val_loss
# x_axis_data is 0-100
x_axis_data = [1, 2, 4, 8]

macrof1_row = [0.8890, 0.8845, 0.8873, 0.8953]



plt.plot(x_axis_data, macrof1_row, 'bo--', alpha=0.5, linewidth=1, label='FedHAN+SA')

 
  
plt.legend()  #显示上面的label
plt.xlabel('Number of FedHAN+SA attention head')
plt.ylabel('Macro-F1')#accuracy
  
#plt.ylim(-1,1)#仅设置y轴坐标范围
plt.savefig('my_chart2.png', dpi=300)
