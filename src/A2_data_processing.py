import numpy as np
import pandas as pd
import json
import os, sys
cwd_path = os.path.dirname(sys.path[0])
os.chdir(cwd_path)
#path = "res/user_data/robin_task_1.json"
path_dir = 'res/user_data'
#data = json.load(open(path))
for file in os.listdir(path_dir):
    path = path_dir+'/'+file
    with open(path) as f:
        data = json.load(f)
        print(data['turns'])


#df = pd.read_json(data)
#df.head()