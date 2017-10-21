
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.stats import multivariate_normal

x, y =  np.meshgrid(np.linspace(-5, 5, 5), np.linspace(-5, 5, 5))
pos = np.dstack((x, y))

# http://ailaby.com/contour/
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.dstack.html
# http://ailaby.com/contour/
# http://techtipshoge.blogspot.jp/2015/12/blog-post_2.html

cov_matrix = [[1,0],[0,1]]
gauss = multivariate_normal(mean=[0,0], cov=cov_matrix)
plt.contour(x, y, gauss.pdf(pos))
plt.axis('equal')
plt.axis([-3, 3, -3, 3])
plt.show()
"""
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x,y,gauss.pdf(pos))
plt.show()
"""

cov_matrix = [[1,0],[0,3]]
gauss = multivariate_normal(mean=[0,0], cov=cov_matrix)
plt.contour(x, y, gauss.pdf(pos))
plt.axis('equal')
plt.axis([-3, 3, -3, 3])
plt.show()

# mean=3, cov = [[1,0],[0,3]]に従う変数を1000点生成
x, y = np.random.multivariate_normal([3,3],  [[1,0],[0,3]], 1000).T
X = np.dstack((x,y)).reshape((1000,2))
mu = np.array([np.mean(x),np.mean(y)]);mu
print np.mean(x),"\n",np.mean(y)
# まずはコレスキー分解でBを求める
B = np.linalg.cholesky([[1,0],[0,3]])
# これを用いて変数変換する
Z1 = []
Z2 = []
for i in xrange(1000):
    z = np.dot(np.linalg.inv(B),(X-mu)[i])
    Z1.append(z[0])
    Z2.append(z[1])

print np.mean(Z2),"\n",np.mean(Z2)
#共分散行列の固有ベクトルの計算
la, v = np.linalg.eig([[1,0],[0,3]],);v

plt.scatter(Z1,Z2,s=15,c='red',label="changed",alpha=0.45)
plt.scatter(x,y,s=15,label="before",alpha=0.4)
plt.quiver(3, 3, v[0][0], v[0][1])
plt.quiver(3, 3, v[1][0], v[1][1])
plt.legend()
plt.xlabel("x or Z1")
plt.ylabel("y or Z2")

print np.cov(X, rowvar=0)
print np.cov(np.c_[np.array(Z1),np.array(Z2)],rowvar=0)

"""
cov_matrix = [[1,0.5],[0.5,1]]
gauss = multivariate_normal(mean=[0,0], cov=cov_matrix)
plt.contour(x, y, gauss.pdf(pos))
plt.axis('equal')
plt.axis([-3, 3, -3, 3])
plt.show()
"""
