import numpy as np
import sys
sys.path.append('..')


class Grader():
    def __init__(self):
        self.part_names = ['Warm up exercise',
                      'Computing Cost (for one variable)',
                      'Gradient Descent (for one variable)']
        self.answer = {0:None, 1:None, 2:None}
        self.correct = {0:self._correct_00, 1:self._correct_01, 2:self._correct_02}
        self.X1 = np.column_stack((np.ones(20), np.exp(1) + np.exp(2) * np.linspace(0.1, 2, 20)))
        self.Y1 = self.X1[:, 1] + np.sin(self.X1[:, 0]) + np.cos(self.X1[:, 1])
        self.theta = np.array([-0.5, 0.5])
        self.alpha = 0.01
        self.num_iters = 10

    def grade(self, num):
        self.correct[num](self.answer[num])

    def _correct_00(self, answer):
        answer = answer()
        if np.mean(answer == np.eye(5)) == 1:
            print('정답')
        else:
            print('오답')

    def _correct_01(self, answer):
        if round(answer(self.X1, self.Y1, self.theta), 3) == 23.607:
            print('정답')
        else:
            print('오답')

    def _correct_02(self, answer):
        theta, J_history = answer(self.X1, self.Y1, self.theta, self.alpha, self.num_iters)
        if (([round(i, 3) for i in theta] == [-0.431, 1.097]) & (round(J_history[-1], 3) == 0.374)):
            print('정답')
        else:
            print('오답')
