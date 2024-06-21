import os
import sys, copy
sys.path.append('../')
from PIL import Image
import pickle
# from adapter.model_adapter import infer_adapter
from base_decoder.model_base_decoder import infer_base_decoder
from auxiliary_decoder.model_auxiliary_decoder import infer_auxiliary_decoder
from tools.csv2triplet import csv2triples
import os
import torch
import torch.nn as nn
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from auxiliary_decoder.train.utils.prompter import Prompter
from peft import PeftModel
from utils import sys_prompt, ChartBenchTester

import warnings
warnings.filterwarnings("ignore")

CKPT_PATH = '/path/to/models/ChartVLM-base'
TEST_INDEX = '/path/to/ChartBench/test.jsonl'
SAVE_PATH = '/path/to/ChartBench/Result/raw/ChartVLM.jsonl'


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out


class CustomChartBenchTester(ChartBenchTester):
    
    def load_model(self, base_model_path):
        classifier_model_path = os.path.join(base_model_path, 'instruction_adapter', 'mlp_classifier.pth')
        tokenizer_path = os.path.join(base_model_path, 'instruction_adapter', 'vectorizer.pkl' )
        with open(tokenizer_path, 'rb') as file:
            self.vectorizer = pickle.load(file)
        classifier = MLPClassifier(input_dim=1719, hidden_dim=512, output_dim=6)
        classifier.load_state_dict(torch.load(classifier_model_path))
        self.classifier = classifier.eval()

        self.device_1 = self.device_2 = self.device_3 = 'cuda:0'
        self.base_decoder_1 = Pix2StructForConditionalGeneration.from_pretrained(os.path.join(base_model_path,'base_decoder')).to(self.device_1)
        self.base_decoder_2 = Pix2StructForConditionalGeneration.from_pretrained(os.path.join(base_model_path,'base_decoder','title_type')).to(self.device_2)
        self.processor_base_decoder = Pix2StructProcessor.from_pretrained(os.path.join(base_model_path,'base_decoder'))
        self.processor_base_decoder.image_processor.is_vqa = False
        print('classifier, decoder_1, decoder_2 are loaded successfully...')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(os.path.join(base_model_path,'auxiliary_decoder', 'base'), trust_remote_code=False)
        llama_model = LlamaForCausalLM.from_pretrained(
                    os.path.join(base_model_path,'auxiliary_decoder', 'base'),
                    load_in_8bit=True,
                    torch_dtype=torch.float16,
                    device_map=self.device_3,
                    trust_remote_code=False,
                )
        self.auxiliary_decoder = PeftModel.from_pretrained(
            llama_model,
            os.path.join(base_model_path,'auxiliary_decoder'),
            torch_dtype=torch.float16,
        )
        self.auxiliary_decoder.config.pad_token_id = self.llama_tokenizer.pad_token_id = 0  # unk
        self.auxiliary_decoder.config.bos_token_id = 1
        self.auxiliary_decoder.config.eos_token_id = 2
        self.auxiliary_decoder.half().eval().to(self.device_3)
        print('auxiliary_decoder is loaded successfully...')

        self.stream_output=False
        self.auxiliary_prompter = Prompter('alpaca')
        self.auxiliary_generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
        )
        
    def infer_adapter(self, question):
        questions = [question]
        inputs = self.vectorizer.transform(questions).toarray()
        inputs = torch.tensor(inputs, dtype=torch.float32)
        output = self.classifier(inputs)
        _, predicted_label = torch.max(output, dim=1)
        return predicted_label.item()
 
    def model_gen(self, question, im_path):

        image = Image.open(im_path)
        num = self.infer_adapter(question)
        inputs_base_decoder = self.processor_base_decoder(images=image, return_tensors="pt")

        if num == 0:   #csv
            inputs_base_decoder = inputs_base_decoder.to(self.device_1)
            predictions_base_decoder = self.base_decoder_1.generate(**inputs_base_decoder, max_new_tokens = 1280)
            output_base_decoder = self.processor_base_decoder.decode(predictions_base_decoder[0], skip_special_tokens=True)
            return output_base_decoder.split("</s>")[0]

        if num in [1, 4]: #des sum redraw
            inputs_base_decoder = inputs_base_decoder.to(self.device_1)
            predictions_base_decoder = self.base_decoder_1.generate(**inputs_base_decoder, max_new_tokens = 1280)
            csv = self.processor_base_decoder.decode(predictions_base_decoder[0], skip_special_tokens=True)
            trip = csv2triples(csv, separator='\\t', delimiter='\\n')
            inputs_base_decoder = inputs_base_decoder.to(self.device_2)
            predictions_base_decoder = self.base_decoder_2.generate(**inputs_base_decoder, max_new_tokens = 100)
            title_type = self.processor_base_decoder.decode(predictions_base_decoder[0], skip_special_tokens=True)
            inputs = '<data> '+ ','.join(trip) + ' ' + title_type
            
            prompt = self.auxiliary_prompter.generate_prompt(question, inputs)
            inputs = self.llama_tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device_3)
            with torch.no_grad():
                generation_output = self.auxiliary_decoder.generate(
                    input_ids=input_ids,
                    generation_config=self.auxiliary_generation_config,
                    return_dict_in_generate=False,
                    output_scores=True,
                    max_new_tokens=512,
                )
            output = self.llama_tokenizer.decode(generation_output[0])
            output = output.split('Response:\n')[-1]
            return output.split("</s>")[0]

        if num == 2 : #type
            inputs_base_decoder = inputs_base_decoder.to(self.device_2)
            predictions_base_decoder = self.base_decoder_2.generate(**inputs_base_decoder, max_new_tokens = 1280)
            output_base_decoder = self.processor_base_decoder.decode(predictions_base_decoder[0], skip_special_tokens=True)
            return output_base_decoder.split('<type> ')[-1].split("</s>")[0]

        if num == 3: #title
            inputs_base_decoder = inputs_base_decoder.to(self.device_2)
            predictions_base_decoder = self.base_decoder_2.generate(**inputs_base_decoder, max_new_tokens = 100)
            output_base_decoder = self.processor_base_decoder.decode(predictions_base_decoder[0], skip_special_tokens=True)
            return output_base_decoder.split('<type>')[0].split('<title>')[-1].split("</s>")[0]

        if num == 5: #QA
            inputs_base_decoder = inputs_base_decoder.to(self.device_1)
            predictions_base_decoder = self.base_decoder_1.generate(**inputs_base_decoder, max_new_tokens = 1280)
            csv = self.processor_base_decoder.decode(predictions_base_decoder[0], skip_special_tokens=True)
            trip = csv2triples(csv, separator='\\t', delimiter='\\n')
            inputs_base_decoder = inputs_base_decoder.to(self.device_2)
            predictions_base_decoder = self.base_decoder_2.generate(**inputs_base_decoder, max_new_tokens = 100)
            title_type = self.processor_base_decoder.decode(predictions_base_decoder[0], skip_special_tokens=True)

            inputs = '<data> '+ ','.join(trip) + ' ' + title_type.split('<type>')[-1] + ' <question> '+ question
            ins = '''
            Given the following triplet data (marked by <data>) with the title (marked by <title>) and the question related to the data (marked by <question>), give the answer with no output of hints, explanations or notes.
            '''
            prompt = self.auxiliary_prompter.generate_prompt(ins, inputs)
            inputs = self.llama_tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device_3)
            with torch.no_grad():
                generation_output = self.auxiliary_decoder.generate(
                    input_ids=input_ids,
                    generation_config=self.auxiliary_generation_config,
                    return_dict_in_generate=False,
                    output_scores=True,
                    max_new_tokens=100,
                )
            output = self.llama_tokenizer.decode(generation_output[0])
            output = output.split('Response:\n')[-1].split("</s>")[0]
            return output

        if num > 5:
            return 'Sorry, I can not deal with this task.'



if __name__ == '__main__':
    tester = CustomChartBenchTester(
        test_index=TEST_INDEX,
        sys_prompt_acc=sys_prompt['blip2 style'],
        sys_prompt_nqa=sys_prompt['chartqa']
    )
    tester.load_model(CKPT_PATH)
    tester.infer_all_answers(SAVE_PATH)
