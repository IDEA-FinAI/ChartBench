import shortuuid
import time
import re, json
import openai

openai.api_key = ""
deepseek_key = ""

def get_gpt_answer(
    question: dict, system_prompt: str='', model: str="gpt-3.5-turbo", 
    num_choices: int=1, max_tokens: int=2048, 
    temperature: float=0.1, model_url: str=None
):
    assert model in ['gpt-3.5-turbo', 'deepseek-chat', 'Qwen/Qwen1.5-32B-Chat']
    if openai.__version__ < '1.0.0':
        from fastchat.llm_judge.common import chat_compeletion_openai
        from fastchat.model.model_adapter import get_conversation_template
        choices = []
        for i in range(num_choices):
            conv = get_conversation_template(model)
            turns = []
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            output = chat_compeletion_openai(model, conv, temperature, max_tokens)
            conv.update_last_message(output)
    else:
        from openai import OpenAI
        if 'gpt' in model:
            openai_api_key = openai.api_key
            openai_api_base = "https://api.openai-proxy.com/v1"
            # openai_api_base = None
        elif 'deepseek-chat' in model:
            openai_api_key = deepseek_key
            openai_api_base = "https://api.deepseek.com/"
        elif model == 'Qwen/Qwen1.5-32B-Chat':
            openai_api_key = "EMPTY"
            openai_api_base = "http://192.168.80.3:9413/v1"
        else:
            openai_api_key = "EMPTY"
            openai_api_base = model_url
        
        # print(openai_api_base)
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        if system_prompt == '': system_prompt = SYSTEM_PROMPT
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=temperature
        )
        output = response.choices[0].message.content

    ans = {
        "answer_id": shortuuid.uuid(),
        "model_id": model,
        "answer": output,
        "tstamp": time.time(),
    }

    return output