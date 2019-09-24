import sys
import numpy as np
from matplotlib import pyplot
from scipy.special import expit

sys.path.append('..')


def mapFeature(X1, X2, degree=6):
    """
    Maps the two input features to quadratic features used in the regularization exercise.

    Returns a new feature array with more features, comprising of
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..

    Parameters
    ----------
    X1 : array_like
        A vector of shape (m, 1), containing one feature for all examples.

    X2 : array_like
        A vector of shape (m, 1), containing a second feature for all examples.
        Inputs X1, X2 must be the same size.

    degree: int, optional
        The polynomial degree.

    Returns
    -------
    : array_like
        A matrix of of m rows, and columns depend on the degree of polynomial.
    """
    if X1.ndim > 0:
        out = [np.ones(X1.shape[0])]
    else:
        out = [np.ones(1)]

    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((X1 ** (i - j)) * (X2 ** j))

    if X1.ndim > 0:
        return np.stack(out, axis=1)
    else:
        return np.array(out)


def plotDecisionBoundary(plotData, theta, X, y):
    """
    Plots the data points X and y into a new figure with the decision boundary defined by theta.
    Plots the data points with * for the positive examples and o for  the negative examples.

    Parameters
    ----------
    plotData : func
        A function reference for plotting the X, y data.

    theta : array_like
        Parameters for logistic regression. A vector of shape (n+1, ).

    X : array_like
        The input dataset. X is assumed to be  a either:
            1) Mx3 matrix, where the first column is an all ones column for the intercept.
            2) MxN, N>3 matrix, where the first column is all ones.

    y : array_like
        Vector of data labels of shape (m, ).
    """
    # make sure theta is a numpy array
    theta = np.array(theta)

    # Plot Data (remember first column in X is the intercept)
    plotData(X[:, 1:3], y)

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])

        # Calculate the decision boundary line
        plot_y = (-1. / theta[2]) * (theta[1] * plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        pyplot.plot(plot_x, plot_y)

        # Legend, specific for the exercise
        pyplot.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        pyplot.xlim([30, 100])
        pyplot.ylim([30, 100])
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((u.size, v.size))
        # Evaluate z = theta*x over the grid
        for i, ui in enumerate(u):
            for j, vj in enumerate(v):
                z[i, j] = np.dot(mapFeature(ui, vj), theta)

        z = z.T  # important to transpose z before calling contour
        # print(z)

        # Plot z = 0
        pyplot.contour(u, v, z, levels=[0], linewidths=2, colors='g')
        pyplot.contourf(u, v, z, levels=[np.min(z), 0, np.max(z)], cmap='Greens', alpha=0.4)




class Grader():
    def __init__(self):
        self.part_names = ['Sigmoid Function',
                      'Logistic Regression Cost',
                      'Logistic Regression Gradient',
                      'Predict',
                      'Regularized Logistic Regression Cost',
                      'Regularized Logistic Regression Gradient'
        ]
        self.answer = {0:None, 1:None, 2:None, 3:None, 4:None, 5:None, 6:None}
        self.correct = {0:self._correct_00, 1:self._correct_01, 2:self._correct_02, 3:self._correct_03}
        self.theta = np.array([-24, 0.2, 0.2])
        self.X1 = np.stack([np.ones(20),
                  np.exp(1) * np.sin(np.arange(1, 21)),
                  np.exp(0.5) * np.cos(np.arange(1, 21))], axis=1)
        self.Y1 = (np.sin(self.X1[:, 0] + self.X1[:, 1]) > 0).astype(float)
        self.goodjob = '맞습니다. 다음 단계로 진행하세요.'
        self.tryagain = '다시 한번 시도해보세요.'

        

    def grade(self, num):
        self.correct[num](self.answer[num])

    def _correct_00(self, answer):
        answer = np.round(answer(np.array([[0, 1], [1, 0]])), 3)
        correct = np.round(expit(np.array([[0, 1], [1, 0]])), 3)
        if np.mean(answer == correct) == 1:
            print(self.goodjob)
        else:
            print(self.tryagain)
            

    def _correct_01(self, answer):
        answer_cost, answer_grad = answer(self.theta, self.X1, self.Y1)
        answer_cost = round(answer_cost, 3)
        answer_grad = np.round(answer_grad, 3)
        if answer_cost == 9.549 and np.mean(answer_grad == [-0.4, -0.253, 0.001]) == 1:
            print(self.goodjob)
        else:
            print(self.tryagain)

    def _correct_02(self, answer):
        answer = answer(self.theta, self.X1)
        correct = np.zeros(20)
        if np.mean(answer == correct) == 1:
            print(self.goodjob)
        else:
            print(self.tryagain)

    def _correct_03(self, answer):
        reg_X1 = mapFeature(self.X1[:, 1], self.X1[:, 2])
        answer_cost, answer_gradient = answer(np.zeros(28), reg_X1, self.Y1, 1)
        answer_cost = round(answer_cost, 5)
        answer_gradient = np.round(answer_gradient, 3)
        correct_gradient = np.array([ 1.0000e-01, -1.8500e-01,  2.3000e-02,  1.3830e+00,  4.6000e-02,
       -2.3700e-01, -2.8000e-01,  1.1200e-01, -4.0100e-01,  2.1000e-02,
        9.3880e+00,  2.4800e-01,  3.0700e-01,  3.3000e-02, -7.5700e-01,
        4.0100e-01,  5.7900e-01, -9.0800e-01,  9.0000e-02, -7.5600e-01,
        2.5000e-02,  6.2518e+01,  1.3080e+00,  2.5210e+00,  1.9200e-01,
       -9.4000e-02,  2.0000e-02, -2.0240e+00])
        if answer_cost == 0.69315 and np.mean(answer_gradient == correct_gradient) == 1:
            print(self.goodjob)
        else:
            print(self.tryagain)