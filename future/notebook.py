import json
import os
import re

with open('C:/users/yoon/desktop/project/tnt_code/new/1주차/1주차.ipynb') as f:
    ff = json.load(f)

regex = re.compile(r"((### 코드 입력 ###)[\s\S]+(### 코드 입력 ###))+")
#regex = re.compile('(?<=(### 코드 입력 ###))')

erase_start = False
for cell in ff['cells']:
    if cell['cell_type'] == 'code':
        describes = ''.join(cell['source'])
        print(regex.findall(describes))
        #print(cell['source'])
