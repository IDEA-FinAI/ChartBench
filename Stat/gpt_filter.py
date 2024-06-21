import os, json, time, re
from tqdm import tqdm
from utils import get_gpt_answer

SYSTEM_MESSAGE = """
Please extract the answer from the model response and type it.

Note:
1. The responses may be a phrase, a number, or a sentence.
2. If the content of the responses is not understandable, return "FAILED".
3. If the content of the responses is understandable, extract the numerical value from it.
4. If the responses is a yes or no judgment, return yes or no.

Special requirements: ** Only numbers, "FAILED" or yes/no are allowed to be returned for each response, please do not return anything else! **

Please read the following example. 

Question 1: Which number is missing?
Model response: The number missing in the sequence is 14.

Question 2: What is the fraction of females facing the camera?
Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.

Question 3: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)
Model response: Ax00 Ax00 Ax00 Ax00 Ax00 Ax00 Ax00 Ax00 Ax00 Ax00 Ax00. 

Question 4: The profit of Aug. is $5,000,000.
Model response: Yes, the profit is quite exciting!

Your answer: 
14
0.6
FAILED
Yes
"""

USER_MESSAGE = """ 
Question {}: {}
Model response: {}
"""


def has_consecutive_chars(string, k=100):
    pattern = r'(.)\1{{{}}}'.format(k-1)
    return bool(re.search(pattern, string))


def query_llm_for_value(samples, undone_keys):
    
    split_K = len(undone_keys)
    
    undone_user_message = []
    for i in range(split_K):
        undone_key = undone_keys[i]
        query = samples[undone_key]["conversation"][0]["query"]
        answer = samples[undone_key]["conversation"][0]["answer"][:100]
        answer = 'FAILED' if has_consecutive_chars(answer) else answer
        user_message = USER_MESSAGE.format(i+1, query, answer)
        undone_user_message.append(user_message)
    query_user_message = '\n'.join(undone_user_message)
    query_user_message = SYSTEM_MESSAGE + f'\nMake sure you reture {split_K} line answer.\n' + query_user_message + '\nYour answer:\n'

    patience = 5
    response_list = []
    while patience > 0:
        response_all = get_gpt_answer(question=query_user_message, model='deepseek-chat')
        response_list = response_all.split('\n')
        if len(response_list) == split_K: break
        patience -= 1
    
    try:
        assert len(response_list) == split_K
    except:
        print('GPT prase error: ', undone_keys)

    return response_list


def eval_json_wrt_model(answer_list, split_K=10):
    
    save_json_path = answer_list.replace("Result/raw", "Result/filter")
    directory = os.path.dirname(save_json_path)
    os.makedirs(directory, exist_ok=True)
    
    samples = [json.loads(line) for line in open(answer_list, 'r').readlines()]

    undone_keys = [] 
    tosave_keys = []
    for idx in tqdm(len(samples)):
        tosave_keys.append(idx)
        if samples[idx]["type"]["QA"] == "Acc+": continue
        if samples[idx]["type"]["QA"] == "GPT-acc" and "gpt_filter" in samples[idx]["conversation"][-1].keys(): continue
        
        undone_keys.append(idx)
        if len(undone_keys) == split_K:
            response_list = query_llm_for_value(samples, undone_keys)
            for key, rsp in zip(undone_keys, response_list):
                samples[key]["conversation"][0]["gpt_filter"] = rsp

            with open(save_json_path, 'a+') as f:
                for key in tosave_keys:
                    json_str = json.dumps(samples[key])
                    f.write(json_str + '\n')
                    
            undone_keys = [] 
            tosave_keys = []

    if undone_keys != [] or tosave_keys != []:
        response_list = query_llm_for_value(samples, undone_keys)
        for key, rsp in zip(undone_keys, response_list):
            samples[key]["conversation"][0]["gpt_filter"] = rsp
        with open(save_json_path, 'a+') as f:
            for key in tosave_keys:
                json_str = json.dumps(samples[key])
                f.write(json_str + '\n')
       

if __name__ == '__main__':
    eval_json_wrt_model("/path/to/ChartBench/Result/raw/BLIP2.jsonl")
