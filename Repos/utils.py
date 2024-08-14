import os, sys, json
import torch
from tqdm import tqdm

class ChartBenchTester:

    def __init__(self, test_index, sys_prompt_acc, sys_prompt_nqa):
        self.test_index = test_index
        self.system_prompt_acc = sys_prompt_acc
        self.system_prompt_nqa = sys_prompt_nqa
        self.image_root = ''
        
    def load_model(self):
        pass
        
    def model_gen(self, question, im_path):
        pass
    
    def reset_image_root(self, p):
        self.image_root = p
        
    def reset_prompt(self, pacc, pnqa):
        self.system_prompt_acc = pacc
        self.system_prompt_nqa = pnqa
        
    def load_jsonl(self, file_path, mode='r'):
        data = []
        with open(file_path, mode) as f:
            for line in f:
                obj = json.loads(line)
                data.append(obj)
        return data

    def save_jsonl(self, file_path, data, mode='w'):
        with open(file_path, mode) as f:
            for obj in data:
                json_str = json.dumps(obj)
                f.write(json_str + '\n')
            
    def infer_all_answers(self, output_path):

        directory = os.path.dirname(output_path)
        os.makedirs(directory, exist_ok=True)
        samples = self.load_jsonl(self.test_index, mode='r')
        
        if os.path.exists(output_path): # ckpt
            ckpt_index = len(self.load_jsonl(output_path, mode='r'))
            print(f'Start from sample {ckpt_index} ...')
        else:
            ckpt_index = -1

        for i in tqdm(range(len(samples))):
            
            if samples[i]['id'] < ckpt_index: continue
            
            im_path = samples[i]["image"].replace('./data', self.image_root)
            if samples[i]["type"]["QA"] == "Acc+":
                Qr = self.system_prompt_acc.format(samples[i]["conversation"][0]["query"])
                Qw = self.system_prompt_acc.format(samples[i]["conversation"][1]["query"])
                with torch.cuda.amp.autocast():
                    Ar = self.model_gen(Qr, im_path)
                    Aw = self.model_gen(Qw, im_path)
                samples[i]["conversation"][0]["query"] = Qr
                samples[i]["conversation"][1]["query"] = Qw
                samples[i]["conversation"][0]["answer"] = Ar
                samples[i]["conversation"][1]["answer"] = Aw

            if samples[i]["type"]["QA"] == "GPT-acc":
                Qr = self.system_prompt_nqa.format(samples[i]["conversation"][0]["query"])
                with torch.cuda.amp.autocast():
                    Ar = self.model_gen(Qr, im_path)
                samples[i]["conversation"][0]["query"] = Qr
                samples[i]["conversation"][0]["answer"] = Ar

            self.save_jsonl(output_path, [samples[i]], mode='a+')


# blip style
prompt_v1 = 'Question: {} Please answer yes or no. Answer:'

# in context learning style
prompt_v2 = '''You are a data analyst, good at dealing with chart data. Now you are required to analyze a chart for the User. You only need to answer [yes] or [no].
Here is an example:
User: <image>
User: The figure is a line chart. Please answer yes or no.
You: yes.

Following the above example:
The query from the User is: {} Please answer yes or no.
Your Answer:
'''

# vanilla style
prompt_v3 = '''You are a data analyst, good at dealing with chart data. Now you are required to analyze a chart for the User. You only need to answer [yes] or [no].
The query from the User is: {} Please answer yes or no.
Your Answer:'''

# no or yes 
prompt_v4 = '''You are a data analyst, good at dealing with chart data. Now you are required to analyze a chart for the User. You only need to answer [no] or [yes].
The query from the User is: {} Please answer no or yes.
Your Answer:'''

# no or yes blip style
prompt_v5 = 'Question: {}. Please answer no or yes. Answer:'

# cot_style
chartcotv1 = '''Carefully examine this chart and accurately understand its chart type, title, legend, labels, and coordinate system elements. 
Based on your observations, determine whether the following user assertion about the chart are correct. 
The assertion is '{}'.
Please provide a simple 'Yes' or 'No' response without any additional content.
Your Answer:'''

chartcotv2 = '''Carefully examine this chart and determine whether the following user assertion about the chart are correct.
User assertion: '{}'.
Let's thinking the following qustions one by one first:
1. What is user's assertion?
2. What are queried entities?
3. What are corosponding color / line style / legend / ... for these entities?
4. What is this chart type? if it is bar / line / scatter plot, please notice its cordinate / ticks ...
5. What are the entities value?
6. What are entities ralationship?
Combined with your answers, please provide a simple 'Yes' or 'No' response without any additional content.
Your Answer:
'''

chartqa_v1 = 'user:\nAnswer the question using a single word or phrase. {}\nassistant:\n'

nqa_pot = 'Program of Thought: {}'
acc_pot = 'Fact Checking: {}'

sys_prompt = {
    'blip2 style': prompt_v1,
    'ICL style': prompt_v2,
    'llava style': prompt_v3,
    'llava style no or yes': prompt_v4,
    'blip style no or yes': prompt_v5,
    'chartqa': chartqa_v1,
    'nqa_pot': nqa_pot,
    'acc_pot': acc_pot
}
