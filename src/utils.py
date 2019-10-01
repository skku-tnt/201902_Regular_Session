import inspect
import numpy as np
from answer import get_answer


def submit(pred):
    pred = pred
    real = get_answer(pred.__name__)
    true_or_false = check_answer(real.args, real.test, pred)
    message(true_or_false)

def help_me(func_name):
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
    true_or_false = np.array([np.array(real_value[i] == pred_value[i]).all() for i in range(len(real_value))]).all()
    return true_or_false


def message(true_or_false):
    if true_or_false:
        print('정답입니다.')
    else:
        print('오답입니다.')

def change_value_type(value):
    if type(value) == tuple:
        return value
    else:
        return [value]
