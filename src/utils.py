import inspect
import socketio
import time
import numpy as np
from answer import get_answer

class Submit:
    def __init__(self):
        self.name = ''
        self.score = {}

    def connect(self):
        self.name = input('이름을 입력해주세요.')
        self.address = input('주소를 입력해주세요.')

    def submit(self, pred):
        pred = pred
        real = get_answer(pred.__name__)
        true_or_false = check_answer(real.args, real.test, pred)

        self.score[pred.__name__] = str(true_or_false)
        if self.name:
            connect_board(self.name, self.score, self.address)

        message(true_or_false)


    def help_me(self, func_name):
        func = get_answer(func_name)
        src = inspect.getsource(func.test)
        src = src.replace('def test', 'def {}'.format(func_name)).replace('.test', '')
        print('-'*15, '정답 코드', '-'*15)
        print('\n')
        print(src)
        print('-'*15, '정답 코드', '-'*15)


def check_answer(args, real, pred):
    if args == None:
        real_value = change_value_type(real())
        pred_value = change_value_type(pred())
    else:
        real_value = change_value_type(real(**args))
        pred_value = change_value_type(pred(**args))

    true_or_false = np.array([np.abs(np.mean(real_value[i] - pred_value[i])) < 1e-9 for i in range(len(real_value))]).all()
    return true_or_false


def message(true_or_false):
    if true_or_false:
        print('정답입니다.')
    else:
        print('오답입니다.')

def change_value_type(value):
    if type(value) == tuple:
        value = [np.array(i) for i in value]
        return value
    else:
        return np.array([value])

def connect_board(name, score, address):
    sio = socketio.Client()
    sio.connect(f'{address}')
    sio.emit('submit', {
        'name' : name,
        'score' : score})
    print('submission completed')
    time.sleep(3)
    sio.disconnect()
