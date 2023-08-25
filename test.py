import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y1 = [1, 4, 9, 16]
y2 = [1, 2, 6, 8]

f, a = plt.subplots(1, 2)

a[0].scatter(x,y1)
a[1].scatter(x,y2)
plt.show()
