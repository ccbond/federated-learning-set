import matplotlib.pyplot as plt
import numpy as np
  
#epoch,acc,loss,val_acc,val_loss
# x_axis_data is 0-100
x_axis_data = [64, 128, 256, 512]

macrof1_row = [0.8927, 0.8894, 0.8876, 0.8869]



plt.plot(x_axis_data, macrof1_row, 'bo--', alpha=0.5, linewidth=1, label='FedHAN+SA')

 
  
plt.legend()  #显示上面的label
plt.xlabel('Demension of FedHAN+SA sem-levle attention vector q')
plt.ylabel('Macro-F1')#accuracy
  
plt.xticks(x_axis_data)

#plt.ylim(-1,1)#仅设置y轴坐标范围
plt.savefig('my_chart3.png', dpi=300)
