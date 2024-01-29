import os, re, json
import pandas as pd

# env = xzz_2.0

META_PATH = './QA/Acc+/index.json'

def fuzzy_match(sentence):
    sentence = str(sentence)
    contains_yes = re.search(r'\byes\b', sentence, re.IGNORECASE) is not None
    contains_no = re.search(r'\bno\b', sentence, re.IGNORECASE) is not None
    return contains_yes, contains_no


class GLM_judger:

    def __init__(self):
        super(GLM_judger, self).__init__()
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        from transformers import AutoTokenizer, AutoModel
        ckpt_path = '/data/FinAi_Mapping_Knowledge/qiyiyan/models/chatglm3-6b/ZhipuAI/chatglm3-6b'
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(ckpt_path, trust_remote_code=True).half().cuda()
        self.model = model.eval()
        self.chat_templet = '''
        Here is the robot answer to user query. Please help me judge whether the robot replies yes or no.
        Case: '{}'
        Give your answer as succinctly as possible, just including only 'yes' / 'no' / 'I dont know'.
        '''

    def judge(self, chat_content):
        response, _ = self.model.chat(self.tokenizer, chat_content, history=[])
        return fuzzy_match(response)


class NLTK_judger:

    def __init__(self):
        super(NLTK_judger, self).__init__()
        import nltk
        # nltk.download('vader_lexicon')
        from nltk.sentiment import SentimentIntensityAnalyzer
        self.sia = SentimentIntensityAnalyzer()
        self.positive_keywords = ["Yes", "Yes,", "My answer is yes"]
        self.negative_keywords = ["No", "No,", "I dont know"]

    def judge(self, chat_content):

        sentiment_score = self.sia.polarity_scores(chat_content)["compound"]

        if sentiment_score >= 0.2:
            return True, False
        elif sentiment_score <= -0.2:
            return False, True

        for word in self.positive_keywords:
            if word in chat_content:
                return True, False

        for word in self.negative_keywords:
            if word in chat_content:
                return False, True

        return False, True


class Vanilla_judger:

    def __init__(self):
        super(Vanilla_judger, self).__init__()
        self.model = None

    def judge(self, chat_content):
        return fuzzy_match(chat_content)


def load_meta():
    QA_meta_list = []
    with open(META_PATH, 'r') as fmeta:
        meta = json.load(fmeta)
        chart_type = list(meta.keys())
        for chart in chart_type:
            for image_type in meta[chart].keys():
                QA_path = meta[chart][image_type]['QA_path']
                QA_meta_list.append(QA_path)
    return QA_meta_list


def summary_inference(model_name, judger, task_name):

    SAVE_ROOT = f'Eval/{task_name}'
    save_path = META_PATH.replace('QA', SAVE_ROOT)
    save_path = save_path.replace('index.json', f'Eval_Result/{model_name}')
    os.makedirs(save_path, exist_ok=True)
    
    log_path = os.path.join(save_path, 'log.txt')
    logger = open(log_path, 'w')
    
    for QA_path in load_meta():
        QA_path = QA_path.replace('QA', SAVE_ROOT)
        QA_path = QA_path.replace('meta.json', f'{model_name}.json')
        # print(QA_path) # For debug
        with open(QA_path, 'r', encoding='utf-8') as fj:
            meta = json.load(fj)
            file_list = list(meta.keys())
            for file in file_list:
                QAs = meta[file]["QA"]
                chart_type = meta[file]["chart_type"]
                image_type = meta[file]["image_type"]
                QA_type = meta[file]["QA_type"]
                for key in QAs.keys():
                    file_name = file.replace('.txt', '')
                    index = f'{QA_type}\t{chart_type}\t{image_type}\t{file_name}\t{key}'

                    Ar = meta[file]["QA"][key]['Ar']
                    isYes_Ar, isNo_Ar = judger.judge(Ar)

                    Aw = meta[file]["QA"][key]['Aw']
                    isYes_Aw, isNo_Aw = judger.judge(Aw)
                    
                    log_content = f'{index}\t{isYes_Ar}\t{isNo_Aw}\n'
                    logger.write(log_content)
                    
    logger.close()


def parse_result(model_name, task_name, judge_level, verbose=True):
    
    index2level = {
        'QA_type': 0,
        'chart_type': 1,
        'image_type': 2,
        'key': 4
    }
    
    res_acc = {}
    res_acc_plus = {} # true and false
    res_acc_plus_tt = {} # both are true
    res_acc_plus_ff = {} # both are false
    idx = index2level[judge_level]
    log_root = os.path.join('Eval', task_name, f'Acc+/Eval_Result/{model_name}')
    log_path = os.path.join(log_root, 'log.txt')
    
    with open(log_path, 'r') as fmeta:
        for line in fmeta.readlines():
            items = line.strip('\n').split('\t')
            if items[idx] not in res_acc_plus.keys(): res_acc_plus[items[idx]] = []
            if items[idx] not in res_acc_plus_tt.keys(): res_acc_plus_tt[items[idx]] = []
            if items[idx] not in res_acc_plus_ff.keys(): res_acc_plus_ff[items[idx]] = []
            if items[idx] not in res_acc.keys(): res_acc[items[idx]] = []
            Aw = items[-1] == 'True'
            Ar = items[-2] == 'True'
            # acc
            res_acc[items[idx]].append(1 if Ar else 0)
            res_acc[items[idx]].append(1 if Aw else 0)
            # acc+ yes and no
            flag_acc_plus = 1 if Ar and Aw else 0
            res_acc_plus[items[idx]].append(flag_acc_plus)
            # acc+ yes and yes
            flag_acc_plus_tt = 1 if Ar and not Aw else 0
            res_acc_plus_tt[items[idx]].append(flag_acc_plus_tt)
            # acc+ no and no
            flag_acc_plus_ff = 1 if not Ar and Aw else 0
            res_acc_plus_ff[items[idx]].append(flag_acc_plus_ff)

    if verbose: print(f'Model name: {model_name}\n')
    key_excel = []
    acc_excel = []
    acc_plus_excel = []
    acc_plus_tt_excel = []
    acc_plus_ff_excel = []
    for key in res_acc.keys():
        key_excel.append(key)
        acc = sum(res_acc[key])/len(res_acc[key])
        acc_plus = sum(res_acc_plus[key])/len(res_acc_plus[key])
        acc_plus_tt = sum(res_acc_plus_tt[key])/len(res_acc_plus_tt[key])
        acc_plus_ff = sum(res_acc_plus_ff[key])/len(res_acc_plus_ff[key])
        
        acc_excel.append(acc*100)
        acc_plus_excel.append(acc_plus*100)
        acc_plus_tt_excel.append(acc_plus_tt*100)
        acc_plus_ff_excel.append(acc_plus_ff*100)
        if verbose: 
            print(f'{judge_level}: {key}')
            print(f'Acc: {acc*100:.2f}%')
            print(f'Acc+: {acc_plus*100:.2f}%')
            print(f'Acc+ yes and yes: {acc_plus_tt*100:.2f}%')
            print(f'Acc+ no and no: {acc_plus_ff*100:.2f}%')
            print('\n')
    
    data = {
        'Key': key_excel,
        "Acc": acc_excel,
        "Acc+": acc_plus_excel,
        "Acc+tt": acc_plus_tt_excel,
        "Acc+ff": acc_plus_ff_excel
    }
    
    df = pd.DataFrame(data)
    df.to_excel(os.path.join(log_root, f'Acc_{judge_level}.xlsx'), index=False)
    return data


def format_result(data, task_name, judge_level):
    
    log_root = os.path.join('Eval', task_name, f'Acc+/Eval_Result')
    
    res_acc = {}
    res_acc_plus = {}
    res_acc_plus_tt = {}
    res_acc_plus_ff = {}
    for model_name in data.keys():
        Keys = data[model_name]['Key']
        Acc = data[model_name]['Acc']
        Acc_plus = data[model_name]['Acc+']
        Acc_plus_tt = data[model_name]['Acc+tt']
        Acc_plus_ff = data[model_name]['Acc+ff']
        res_acc['Types'] = Keys
        res_acc[model_name] = Acc
        res_acc_plus['Types'] = Keys
        res_acc_plus[model_name] = Acc_plus
        res_acc_plus_tt['Types'] = Keys
        res_acc_plus_tt[model_name] = Acc_plus_tt
        res_acc_plus_ff['Types'] = Keys
        res_acc_plus_ff[model_name] = Acc_plus_ff
              
    # save all model info
    df = pd.DataFrame(res_acc)
    df.to_excel(os.path.join(log_root, f'Acc_{judge_level}_all.xlsx'), index=False)
    df = pd.DataFrame(res_acc_plus)
    df.to_excel(os.path.join(log_root, f'Acc+_{judge_level}_all.xlsx'), index=False)
    df = pd.DataFrame(res_acc_plus_tt)
    df.to_excel(os.path.join(log_root, f'Acc+tt_{judge_level}_all.xlsx'), index=False)
    df = pd.DataFrame(res_acc_plus_ff)
    df.to_excel(os.path.join(log_root, f'Acc+ff_{judge_level}_all.xlsx'), index=False)
    return


def eval_models(model_names, task):
    judge_level = ['QA_type', 'chart_type', 'image_type', 'key']
    judger = Vanilla_judger()
    
    # count all models
    for name in model_names:
        summary_inference(name, judger, task_name=task)
    
    # analyze all metric
    for level in judge_level:
        data_all_acc = {}
        for name in model_names:
            data_acc = parse_result(name, task, level)
            data_all_acc[name] = data_acc
        format_result(data_all_acc, task, level)
    return


def eval_online_models():
    # online models
    eval_models(['GPT4V'], 'online_ERNIE')
    eval_models(['GPT4V'], 'online_gpt4v')
    return 

if __name__ == '__main__':
    
    model_names = [
        # 'blip2-flan-t5-xxl',
        # 'cogvlm-chat', 
        # # 'fuyu-8b',
        #  'instructblip-vicuna-7b',
        #  'internlm-xcomposer-7b',
        # #  'LaVIN',
        #  'llava-v1.5-13b',
        'minigpt_v2',
        #  'mplug-owl-bloomz-7b-multilingual',
        #  'Qwen-VL-Chat',
        #  'shikra-7b',
        #  'SPHINX',
        #  'visualglm-6b',
        #  'ChartLlama-13b'
        ]

    task_names = [
        'BLIP2_Style'
        ]

    for task in task_names:
        eval_models(model_names, task)

    # eval_online_models()