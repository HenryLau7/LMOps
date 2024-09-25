import json
import os
with open('/home/aiscuser/LMOps/prompt_optimization/data/BigBench/data_splits.json', 'r') as f:
    data = json.load(f)

all_task_id = [task['task_id'] for task in data]

for task in all_task_id:
    if task in ['word_sorting','presuppositions_as_nli']:
        continue

    data_path = f'/home/aiscuser/LMOps/prompt_optimization/data/BigBench/{task}/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    task_data = [each_task for each_task in data if each_task['task_id'] == task][0]
            
    train_data = []
    correct_label = ['yes', 'correct', 'true', 'plausible']

    for item in task_data['d_train']:
        input_text = item['input']
        label = item['output']
        train_data.append({
            # "label": 1 if label == "yes" else 0,  # 假设'yes'为1，'no'为0
            "label": 1 if label in correct_label else 0, 
            "text": input_text
        })

    val_data = []
    for item in task_data['d_val']:
        input_text = item['input']
        label = item['output']
        # 将数据按所需格式添加到train_data
        val_data.append({
            # "label": 1 if label == "yes" else 0,  # 假设'yes'为1，'no'为0
            "label": 1 if label in correct_label else 0, 
            "text": input_text
        })

    
    # 提取测试集
    test_data = []
    for item in task_data['d_test']:
        input_text = item['input']
        label = item['output']
        # 将数据按所需格式添加到test_data
        test_data.append({
            "label": 1 if label in correct_label else 0, 
            "text": input_text
        })


    # with open(f'{data_path}train.jsonl', 'w') as f_train:
    #     for entry in train_data:
    #         f_train.write(json.dumps(entry) + '\n')

    with open(f'{data_path}val.jsonl', 'w') as f_val:
        for entry in val_data:
            f_val.write(json.dumps(entry) + '\n')

    # with open(f'{data_path}test.jsonl', 'w') as f_test:
    #     for entry in test_data:
    #         f_test.write(json.dumps(entry) + '\n')
