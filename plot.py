import matplotlib.pyplot as plt
import os

arr = []

reader = open('plot.txt', 'r')
arr = reader.readlines()
reader.close()
#os.remove('plot.txt')

for i in range(len(arr)):
    if arr[i] == '-nan(ind)\n' and i != 0:
        arr[i] = arr[i - 1]
    elif i == 0:
        arr[i] = 0
    else:
        arr[i] = float(arr[i])

plt.plot(arr)
plt.show()
