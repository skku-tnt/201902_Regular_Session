import numpy as np
from matplotlib import pyplot
from scipy import optimize

'''
1주차
'''
X = np.column_stack((np.ones(20), np.exp(1) + np.exp(2) * np.linspace(0.1, 2, 20)))
y = X[:, 1] + np.sin(X[:, 0]) + np.cos(X[:, 1])
theta = np.array([-0.5, 0.5])

class warmUpExercise:
    args = None
    def test():
        A = np.eye(5)
        return A

class computeCost:
    args = {
    'X': X,
    'y': y,
    'theta' : theta
    }
    def test(X, y, theta):
        hypothesis = np.matmul(X, theta)
        J = np.mean((hypothesis - y) ** 2) * 0.5
        return J

class gradientDescent:
    X = np.column_stack((np.ones(20), np.exp(1) + np.exp(2) * np.linspace(0.1, 2, 20)))
    args = {
    'X' : X,
    'y' : y,
    'theta' : theta,
    'alpha' : 0.01,
    'num_iters' : 10
    }
    def test(X, y, theta, alpha, num_iters):
        theta = theta.copy()
        J_history = []
        for i in range(num_iters):
            hypothesis = np.matmul(X, theta)
            for j in range(len(theta)):
                theta[j] = theta[j] - alpha * np.dot((hypothesis - y), X[:,j]) * (1 / len(X))
            J_history.append(computeCost.test(X, y, theta))
        return theta, J_history

'''
2주차
'''
X2 = np.column_stack((X, X[:, 1]**0.5, X[:, 1]**0.25))
y2 = np.power(y, 0.5) + y

class featureNormalize:
    args = {
    'X' : X2[:, 1:4]
    }
    def test(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        X_norm = (X - mu) / sigma
        return X_norm, mu, sigma

class computeCostMulti:
    args = {
    'X' : X2,
    'y' : y2,
    'theta' : np.array([0.1, 0.2, 0.3, 0.4])
    }
    def test(X, y, theta):
        hypothesis = np.dot(X, theta)
        J = np.mean((hypothesis -y ) ** 2) * 0.5
        return J


class gradientDescentMulti:
    args = {
    'X' : X2,
    'y' : y2,
    'theta' : np.array([-0.1, -0.2, -0.3, -0.4]),
    'alpha' : 0.01,
    'num_iters' : 10
    }
    def test(X, y, theta, alpha, num_iters):
        m = y.shape[0]
        theta = theta.copy()
        J_history = []
        for i in range(num_iters):
            hypothesis = np.dot(X, theta)
            for j in range(len(theta)):
                theta[j] = theta[j] - alpha * np.mean((hypothesis - y) * X[:,j])
            J_history.append(computeCostMulti.test(X, y, theta))
        return theta, J_history

class normalEqn:
    args = {
    'X' : X2,
    'y' : y2,
    }
    def test(X, y):
        XTX = np.matmul(X.T, X)
        XTXX = np.dot(np.linalg.pinv(XTX), X.T)
        theta = np.dot(XTXX, y)
        return theta

'''
3주차
'''
X = np.stack([np.ones(20),
              np.exp(1) * np.sin(np.arange(1, 21)),
              np.exp(0.5) * np.cos(np.arange(1, 21))], axis=1)
y = (np.sin(X[:, 0] + X[:, 1]) > 0).astype(float)
theta = np.array([0.25, 0.5, -0.5])

class sigmoid:
    args = {
    'z' : X
    }
    def test(z):
        g = 1 / (1 + np.exp(-z))
        return g

class costFunction:
    args = {
    'X' : X,
    'y' : y,
    'theta' : theta
    }
    def test(X, y, theta):
        grad = np.zeros(theta.shape)
        hypothesis = sigmoid.test(np.matmul(X, theta))
        J = np.mean(-y * np.log(hypothesis) - (1-y) * np.log(1 - hypothesis))
        for i in range(len(theta)):
            grad[i] = np.mean((hypothesis - y) * X[:,i])
        return J, grad

class predict:
    args = {
    'X' : X,
    'theta' : theta
    }
    def test(X, theta):
        p = sigmoid.test(np.matmul(X, theta))
        p[p >= 0.5] = 1
        p[p < 0.5] = 0
        return p

class costFunctionReg:
    args = {
    'X' : X,
    'y' : y,
    'theta' : theta,
    'lambda_' : 0.1
    }
    def test(X, y, theta, lambda_):
        grad = np.zeros(theta.shape)
        hypothesis = sigmoid.test(np.matmul(X, theta))
        J = np.mean(-y * np.log(hypothesis) - (1-y) * np.log(1 - hypothesis)) + ((lambda_ / (2 * len(X))) * np.dot(theta[1:], theta[1:]))
        for i in range(len(theta)):
            if i == 0:
                grad[i] = np.mean((hypothesis - y) * X[:,i])
            else:
                grad[i] = np.mean((hypothesis - y) * X[:,i]) + (lambda_ / len(X)) * theta[i]
        return J, grad

'''
4주차
'''
X = np.vstack([np.ones(10),
               np.sin(np.arange(1, 15, 1.5)),
               np.cos(np.arange(1, 15, 1.5))]).T
y = np.sin(np.arange(1, 31, 3))
Xval = np.vstack([np.ones(10),
                  np.sin(np.arange(0, 14, 1.5)),
                  np.cos(np.arange(0, 14, 1.5))]).T
yval = np.sin(np.arange(1, 11))

class linearRegCostFunction:
    args = {
    'X' : X,
    'y' : y,
    'theta' : np.array([0.1, 0.2, 0.3]),
    'lambda_' : 0.5
    }
    def test(X, y, theta, lambda_):
        grad = np.zeros(theta.shape)
        hyp = np.dot(X, theta)
        J = (sum(np.square(hyp - y)) + (lambda_ * sum(np.square(theta[1:])))) / (2 * m)    
        grad[0] = sum((hyp - y) * X[:, 0]) / m
        grad[1:] = (np.dot(X[:, 1:].T, (hyp-y)) / m) + (lambda_ * theta[1:] / m)
        return J, grad

def trainLinearReg(linearRegCostFunction, X, y, lambda_=0.0, maxiter=200):
    initial_theta = np.zeros(X.shape[1])
    costFunction = lambda t: linearRegCostFunction.test(X, y, t, lambda_)
    options = {'maxiter': maxiter}
    res = optimize.minimize(costFunction, initial_theta, jac=True, method='TNC', options=options)
    return res.x

class learningCurve:
    args = {
    'X' : X,
    'y' : y,
    'Xval' : Xval,
    'yval' : yval,
    'lambda_' : 1
    }
    def test(X, y, Xval, yval, lambda_):
        m = y.size
        error_train = np.zeros(m)
        error_val   = np.zeros(m)
        for i in range(1, len(X)+1):
            theta = trainLinearReg(linearRegCostFunction.test, X[:i,:], y[:i], lambda_)
            error_train[i-1], _ = linearRegCostFunction.test(X[:i,:], y[:i], theta, 0)
            error_val[i-1], _ = linearRegCostFunction.test(Xval, yval, theta, 0)
        return error_train, error_val

class polyFeatures:
    args = {
    'X' : X[1, :].reshape(-1, 1),
    'p' : 8
    }
    def test(X, p):
        X_poly = np.zeros((X.shape[0], p))
        for i in range(p):
            X_poly[:, i] = np.power(X[:, 0], i+1)
        return X_poly

class validationCurve:
    args = {
    'X' : X,
    'y' : y,
    'Xval' : Xval,
    'yval' : yval
    }
    def test(X, y, Xval, yval):
        lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
        error_train = np.zeros(len(lambda_vec))
        error_val = np.zeros(len(lambda_vec))

        for i in range(len(lambda_vec)):
            lambda_ = lambda_vec[i]
            theta = trainLinearReg(linearRegCostFunction.test, X, y, lambda_)
            error_train[i], _ = linearRegCostFunction.test(X, y, theta, 0)
            error_val[i], _ = linearRegCostFunction.test(Xval, yval, theta, 0)
        return lambda_vec, error_train, error_val

def plotFit(polyFeatures, min_x, max_x, mu, sigma, theta, p):
    x = np.arange(min_x - 15, max_x + 25, 0.05).reshape(-1, 1)
    X_poly = polyFeatures(x, p)
    X_poly -= mu
    X_poly /= sigma
    X_poly = np.concatenate([np.ones((x.shape[0], 1)), X_poly], axis=1)
    pyplot.plot(x, np.dot(X_poly, theta), '--', lw=2)

'''
5주차
'''

# 함수 이름에 따라 정답을 가져오는 함수
def get_answer(func_name):
    func_list = globals().items()
    func_dict = {k:v for k, v in list(func_list)}
    return func_dict[func_name]
