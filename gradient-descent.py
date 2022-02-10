import numpy as np


def mse_grad(x, y, w):
    res = w[0] + w[1] * x - y
    return res.mean(), (res * x).mean()


def gradient_descent(x, y, w, rate, epoch, tol):
    m,c = w[1], w[0]
    n = len(x)

    for i in range(epoch):
        grad = np.asarray(mse_grad(x, y, w))
        diff = -rate * grad
        print(diff)
        w += diff


        y_pred = m*x + c  # The current predicted value of Y
        D_m = (1/n) * sum(x * (y_pred - y))  # Derivative wrt m
        D_c = (1/n) * sum(y_pred - y)  # Derivative wrt c
        m = m - rate * D_m  # Update m
        c = c - rate * D_c  # Update c

        print('grad = {}, grad`={}'.format(grad, [D_c, D_m]))

        if np.all(abs(w) <= tol):
            break

    return m,c


x = np.asarray([1.0, 2.0, 4.0, 0.0])
y = np.asarray([2.5, 3.0, 4.0, 2.0])
w = gradient_descent(x, y, [2, -1], 0.5, 1000, 0.0001)
pred = w[0] + w[1] * x
pred_w = 2 + -1 * x
print(w)
print(pred)
#print(pred_w)




