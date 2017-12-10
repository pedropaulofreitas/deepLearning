import numpy as np

a = np.zeros((10,))
a[0:5] = 1


b = np.random.sample((10,))

b[b >= 0.5] = 1
b[b < 0.5] = 0

print(a)
print(b)

c = a [ a == b ]

print(c)


acc = c.shape[0] / float(a.shape[0])

print(acc)



#a = np.asarray(a,dtype='int')



