import os, json


# NOTE prompt styles

# blip style
prompt_v1 = 'Question: {}. Please answer yes or no. Answer:'

# in context learning style
prompt_v2 = '''You are a data analyst, good at dealing with chart data. Now you are required to analyze a chart for the User. You only need to answer [yes] or [no].
Here is an example:
User: <image>
User: The figure is a line chart. Please answer yes or no.
You: yes.

Following the above example:
The query from the User is: {} Please answer yes or no.
Your Answer:'''

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

# test 1
prompt_v6 = 'You are an excellent data analyst, especially skilled in analyzing chart data. Please analyze a chart provided by the user and simply answer [yes] or [no] according to the content of the chart. The query from the User is: {} Please answer yes or no. Your Answer:'

# test 2
prompt_v7 = 'According to the chart, answer the question: {}. You only need to answer yes or no.'

# test 3
prompt_v8 = 'You only need to answer yes or no. Question: {}'

# test 4
prompt_v9 = 'You are an excellent data analyst, especially skilled in analyzing chart data. Please analyze the chart provided by the user and simply answer the question using a single word or phrase. The query from the User is: {} Please answer yes or no.'

# cot_style
chartcotv1 = '''Carefully examine this chart and accurately understand its chart type, title, legend, labels, and coordinate system elements. 
Based on your observations, determine whether the following user assertion about the chart are correct. 
The assertion is '{}'.
Please provide a simple 'Yes' or 'No' response without any additional content.
Your Answer:'''

chartcotv2 = '''Carefully examine this chart and determine whether the following user assertion about the chart are correct.
The assertion is '{}'.
Let's thinking the following qustions one by one:
1. What is user's assertion?
2. What are queried entities?
3. What are corosponding color / line style / legend / ... for these entities?
4. What is this chart type? if it is bar / line / scatter plot, please notice its cordinate / ticks ...
5. What are the entities value?
6. What are entities ralationship?
Combined with your answers, please provide a simple 'Yes' or 'No' response without any additional content.
Your Answer:'''

'''
    NOTE required!
'''
prompt_yes_or_no = prompt_v1
task_name = 'BLIP2_Style'


'''
    NOTE base root
'''
pre_root = '/home/qiyiyan/xzz/ChartLLM/ChartBench'
now_root = '/data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartBench'
meta_root = '/data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartBench/QA/Acc+/index.json'

def load_meta():
    QA_meta_list = []
    with open(meta_root, 'r') as fmeta:
        meta = json.load(fmeta)
        chart_type = list(meta.keys())
        for chart in chart_type:
            for image_type in meta[chart].keys():
                QA_path = meta[chart][image_type]['QA_path']
                QA_path = os.path.join(now_root, QA_path)
                QA_meta_list.append(QA_path)
    return QA_meta_list
