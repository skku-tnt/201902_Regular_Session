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
        sigma = np.std(X, axis=0, ddof=1)
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
        m = y.size
        grad = np.zeros(theta.shape)
        hyp = np.dot(X, theta)
        J = (sum(np.square(hyp - y)) + (lambda_ * sum(np.square(theta[1:])))) / (2 * m)
        grad[0] = sum((hyp - y) * X[:, 0]) / m
        grad[1:] = (np.dot(X[:, 1:].T, (hyp-y)) / m) + (lambda_ * theta[1:] / m)
        return J, grad

def trainLinearReg(linearRegCostFunction, X, y, lambda_=0.0, maxiter=200):
    initial_theta = np.zeros(X.shape[1])
    costFunction = lambda t: linearRegCostFunction(X, y, t, lambda_)
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
x1 = np.sin(np.arange(1, 11))
x2 = np.cos(np.arange(1, 11))

def plotData(X, y, grid=False):
    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0

    # Plot Examples
    pyplot.plot(X[pos, 0], X[pos, 1], 'X', mew=1, ms=10, mec='k')
    pyplot.plot(X[neg, 0], X[neg, 1], 'o', mew=1, mfc='y', ms=10, mec='k')
    pyplot.grid(grid)

def svmTrain(X, Y, C, kernelFunction, tol=1e-3, max_passes=5, args=()):
    # make sure data is signed int
    Y = Y.astype(int)
    # Dataset size parameters
    m, n = X.shape

    passes = 0
    E = np.zeros(m)
    alphas = np.zeros(m)
    b = 0

    # Map 0 to -1
    Y[Y == 0] = -1

    # Pre-compute the Kernel Matrix since our dataset is small
    # (in practice, optimized SVM packages that handle large datasets
    # gracefully will **not** do this)

    # We have implemented the optimized vectorized version of the Kernels here so
    # that the SVM training will run faster
    if kernelFunction.__name__ == 'linearKernel':
        # Vectorized computation for the linear kernel
        # This is equivalent to computing the kernel on every pair of examples
        K = np.dot(X, X.T)
    elif kernelFunction.__name__ == 'gaussianKernel':
        # vectorized RBF Kernel
        # This is equivalent to computing the kernel on every pair of examples
        X2 = np.sum(X**2, axis=1)
        K = X2 + X2[:, None] - 2 * np.dot(X, X.T)

        if len(args) > 0:
            K /= 2*args[0]**2

        K = np.exp(-K)
    else:
        K = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):
                K[i, j] = kernelFunction(X[i, :], X[j, :])
                K[j, i] = K[i, j]

    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(m):
            E[i] = b + np.sum(alphas * Y * K[:, i]) - Y[i]

            if (Y[i]*E[i] < -tol and alphas[i] < C) or (Y[i]*E[i] > tol and alphas[i] > 0):
                # select the alpha_j randomly
                j = np.random.choice(list(range(i)) + list(range(i+1, m)), size=1)[0]

                E[j] = b + np.sum(alphas * Y * K[:, j]) - Y[j]

                alpha_i_old = alphas[i]
                alpha_j_old = alphas[j]

                if Y[i] == Y[j]:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                else:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])

                if L == H:
                    continue

                eta = 2 * K[i, j] - K[i, i] - K[j, j]

                # objective function positive definite, there will be a minimum along the direction
                # of linear equality constrain, and eta will be greater than zero
                # we are actually computing -eta here (so we skip of eta >= 0)
                if eta >= 0:
                    continue

                alphas[j] -= Y[j] * (E[i] - E[j])/eta
                alphas[j] = max(L, min(H, alphas[j]))

                if abs(alphas[j] - alpha_j_old) < tol:
                    alphas[j] = alpha_j_old
                    continue
                alphas[i] += Y[i]*Y[j]*(alpha_j_old - alphas[j])

                b1 = b - E[i] - Y[i]*(alphas[i] - alpha_i_old) * K[i, j] \
                     - Y[j] * (alphas[j] - alpha_j_old) * K[i, j]

                b2 = b - E[j] - Y[i]*(alphas[i] - alpha_i_old) * K[i, j] \
                     - Y[j] * (alphas[j] - alpha_j_old) * K[j, j]

                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2)/2

                num_changed_alphas += 1
        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0

    idx = alphas > 0
    model = {'X': X[idx, :],
             'y': Y[idx],
             'kernelFunction': kernelFunction,
             'b': b,
             'args': args,
             'alphas': alphas[idx],
             'w': np.dot(alphas * Y, X)}
    return model


def svmPredict(model, X):
    # check if we are getting a vector. If so, then assume we only need to do predictions
    # for a single example
    if X.ndim == 1:
        X = X[np.newaxis, :]

    m = X.shape[0]
    p = np.zeros(m)
    pred = np.zeros(m)

    if model['kernelFunction'].__name__ == 'linearKernel':
        # we can use the weights and bias directly if working with the linear kernel
        p = np.dot(X, model['w']) + model['b']
    elif model['kernelFunction'].__name__ == 'gaussianKernel':
        # vectorized RBF Kernel
        # This is equivalent to computing the kernel on every pair of examples
        X1 = np.sum(X**2, 1)
        X2 = np.sum(model['X']**2, 1)
        K = X2 + X1[:, None] - 2 * np.dot(X, model['X'].T)

        if len(model['args']) > 0:
            K /= 2*model['args'][0]**2

        K = np.exp(-K)
        p = np.dot(K, model['alphas']*model['y']) + model['b']
    else:
        # other non-linear kernel
        for i in range(m):
            predictions = 0
            for j in range(model['X'].shape[0]):
                predictions += model['alphas'][j] * model['y'][j] \
                               * model['kernelFunction'](X[i, :], model['X'][j, :])
            p[i] = predictions

    pred[p >= 0] = 1
    return pred


def linearKernel(x1, x2):
    return np.dot(x1, x2)


def visualizeBoundaryLinear(X, y, model):
    w, b = model['w'], model['b']
    xp = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    yp = -(w[0] * xp + b)/w[1]

    plotData(X, y)
    pyplot.plot(xp, yp, '-b')

class gaussianKernel:
    args = {
        'x1' : x1,
        'x2' : x2,
        'sigma' : 2
    }
    def test(x1, x2, sigma):
        sim = np.exp(-(np.sum((x1 - x2) ** 2) / (2 * (sigma**2))))
        return sim



def visualizeBoundary(X, y, model):
    plotData(X, y)

    # make classification predictions over a grid of values
    x1plot = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    x2plot = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
    X1, X2 = np.meshgrid(x1plot, x2plot)

    vals = np.zeros(X1.shape)
    for i in range(X1.shape[1]):
        this_X = np.stack((X1[:, i], X2[:, i]), axis=1)
        vals[:, i] = svmPredict(model, this_X)

    pyplot.contour(X1, X2, vals, colors='y', linewidths=2)
    pyplot.pcolormesh(X1, X2, vals, cmap='YlGnBu', alpha=0.25, edgecolors='None', lw=0)
    pyplot.grid(False)

class optimalResult:
    args = None
    def test():
        return 0.3, 0.1

# 함수 이름에 따라 정답을 가져오는 함수
def get_answer(func_name):
    func_list = globals().items()
    func_dict = {k:v for k, v in list(func_list)}
    return func_dict[func_name]
