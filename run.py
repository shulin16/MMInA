import argparse
import json
import os

# from vllm import LLM, SamplingParams
import subprocess
from typing import Any

import google.generativeai as genai
import openai
import tiktoken
import torch
from beartype import beartype
from beartype.door import is_bearable

# Fuyu implementation
from PIL import Image

from agent.prompts import *
from browser_env import Trajectory
from browser_env.actions import (
    Action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
)
from browser_env.utils import Observation, StateInfo
from llms import lm_config
from llms.providers.openai_utils import (
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
)
from models.llama.tokenizer import Tokenizer

# Set your API key here
GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)
import base64
import requests

from transformers import FuyuForCausalLM, FuyuProcessor
from collections import Counter

import requests
root = "/home/data2/stian/MMInA/sampling_data/caption/"
                    

# fuyu implementation: load model and processor
model_id = "adept/fuyu-8b"
processor_fuyu = FuyuProcessor.from_pretrained(model_id)
model_fuyu = FuyuForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="cuda:0"
).to(torch.bfloat16)


class GPT4VisionClient:
    def __init__(self):
        # Set your API key here
        self.api_key = "" 
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def query(self, text, image_paths):
        content_items = [
            {
                "type": "text",
                "text": text
            }
        ]

        for img_path in image_paths:
            base64_image = self.encode_image(img_path)
            content_items.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": content_items,
                }
                
            ],
            "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        a = response.json()
        return a["choices"][0]["message"]["content"]
    

class Agent:
    """Base class for the agent"""

    def __init__(self, *args: Any) -> None:
        pass

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        raise NotImplementedError

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        raise NotImplementedError


class TeacherForcingAgent(Agent):
    """Agent that follows a pre-defined action sequence"""

    def __init__(self) -> None:
        super().__init__()

    @beartype
    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    @beartype
    def set_actions(self, action_seq: str | list[str]) -> None:
        if isinstance(action_seq, str):
            action_strs = action_seq.strip().split("\n")
        else:
            action_strs = action_seq
        action_strs = [a.strip() for a in action_strs]

        actions = []
        for a_str in action_strs:
            try:
                if self.action_set_tag == "playwright":
                    cur_action = create_playwright_action(a_str)
                elif self.action_set_tag == "id_accessibility_tree":
                    cur_action = create_id_based_action(a_str)
                else:
                    raise ValueError(
                        f"Unknown action type {self.action_set_tag}"
                    )
            except ActionParsingError as e:
                cur_action = create_none_action()

            cur_action["raw_prediction"] = a_str
            actions.append(cur_action)

        self.actions: list[Action] = actions

    @beartype
    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        return self.actions.pop(0)

    @beartype
    def reset(
        self,
        test_config_file: str,
    ) -> None:
        with open(test_config_file) as f:
            ref_actions = json.load(f)["reference_action_sequence"]
            tag = ref_actions["action_set_tag"]
            action_seq = ref_actions["action_sequence"]
            self.set_action_set_tag(tag)
            self.set_actions(action_seq)


class PromptAgent(Agent):
    """prompt-based agent that emits action given the history"""

    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        prompt_constructor: PromptConstructor,
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag

    @beartype
    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    @beartype
    def extract_past_questions(self, cnt: int, hist_fold: str) -> str:
        import html2text
        html_path = hist_fold
        history = max(1, cnt - 1)
        html_file = os.path.join(html_path, f'render_{history}.html')
        # print('extract html history', html_file)
        with open(html_file, 'r', encoding='utf-8') as html_input:
            html_content = html_input.read()
            # Convert HTML to Markdown
            h = html2text.HTML2Text()
            h.ignore_links = True
            h.ignore_images = True
            markdown_content = h.handle(html_content)
        # print(markdown_content)
        return markdown_content
    
    @beartype
    def extract_past_questions_multiple(self, cnt: int, hist_fold: str, num: int) -> str:
        import html2text
        html_path = hist_fold
        history = max(1, cnt - 1)
        if cnt == 1:
            html_file = os.path.join(html_path, f'render_1.html')
            if not os.path.exists(html_file):
                return ''
            with open(html_file, 'r', encoding='utf-8') as html_input:
                html_content = html_input.read()
                # Convert HTML to Markdown
                h = html2text.HTML2Text()
                h.ignore_links = True
                h.ignore_images = True
                markdown_content = h.handle(html_content)
            # print(markdown_content)
            return markdown_content
        else:
            markdown_content = ""
            for i in range(cnt-1, cnt-1-num, -1):
                if i > 0:
                    html_file = os.path.join(html_path, f'render_{i}.html')
                    if not os.path.exists(html_file):
                        break
                    # print('extract html history', html_file)
                    with open(html_file, 'r', encoding='utf-8') as html_input:
                        html_content = html_input.read()
                        # Convert HTML to Markdown
                        h = html2text.HTML2Text()
                        h.ignore_links = True
                        h.ignore_images = True
                        markdown_content = markdown_content + "\n" + h.handle(html_content)
                        # print(markdown_content)
            return markdown_content

    @beartype
    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: dict[str, Any]
    ) -> Action:
        prompt = self.prompt_constructor.construct(
            trajectory, intent, meta_data
        )
        lm_config = self.lm_config
        if lm_config.provider == "openai":
            if lm_config.mode == "chat":
                response = generate_from_openai_chat_completion(
                    messages=prompt,
                    model=lm_config.model,
                    temperature=lm_config.gen_config["temperature"],
                    top_p=lm_config.gen_config["top_p"],
                    context_length=lm_config.gen_config["context_length"],
                    max_tokens=lm_config.gen_config["max_tokens"],
                    stop_token=None,
                )
            elif lm_config.mode == "completion":
                response = generate_from_openai_completion(
                    prompt=prompt,
                    engine=lm_config.model,
                    temperature=lm_config.gen_config["temperature"],
                    max_tokens=lm_config.gen_config["max_tokens"],
                    top_p=lm_config.gen_config["top_p"],
                    stop_token=lm_config.gen_config["stop_token"],
                )
            else:
                raise ValueError(
                    f"OpenAI models do not support mode {lm_config.mode}"
                )

        try:
            parsed_response = self.prompt_constructor.extract_action(response)
            if self.action_set_tag == "id_accessibility_tree":
                action = create_id_based_action(parsed_response)
            elif self.action_set_tag == "playwright":
                action = create_playwright_action(parsed_response)
            else:
                raise ValueError(f"Unknown action type {self.action_set_tag}")

            action["raw_prediction"] = response

        except ActionParsingError as e:
            action = create_none_action()
            action["raw_prediction"] = response

        return action

    @beartype
    def next_action_custom(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any],
        model=None,
        args: argparse.Namespace = None,
    ) -> Action:
        print("cnt1 : ", args.cnt1," cnt2 : ", args.cnt2)
        print("history: ", args.hist)
        # switch of past history
        if args.hist:
            # meta_data["past_history"] = self.extract_past_questions(args.cnt1, args.hist_fold) # debug: only works for html
            meta_data["past_history"] = self.extract_past_questions_multiple(args.cnt1, args.hist_fold, args.hist_num) # debug: only works for html
            print(f"=======================using past {args.hist_num} histories=======================")
            print("hist_fold: ",args.hist_fold)
        else:
            meta_data["past_history"] = ''
        prompt = self.prompt_constructor.construct(
            trajectory, intent, meta_data
        )
        # TODO: add classification b4 function call
        lm_config = self.lm_config
        folder_path1 = f"./full_result_shopping/{lm_config.model}/{args.cnt1}"
        folder_path2 = f"{folder_path1}/{args.cnt2}"
        os.makedirs(folder_path2, exist_ok=True)
        file_path = f"{folder_path2}/prompt.json"
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(prompt, json_file, ensure_ascii=False, indent=4)

        if "otter" in lm_config.model.lower():
            if lm_config.mode == "chat":
                lang_only = True
                if lang_only:
                    response = get_response_lo(prompt, model=model)
        elif "wizardlm" in lm_config.model.lower():
            if lm_config.mode == "chat":
                response = generate_from_openai_chat_completion(
                    messages=prompt,
                    model=lm_config.model,
                    temperature=lm_config.gen_config["temperature"],
                    top_p=lm_config.gen_config["top_p"],
                    context_length=lm_config.gen_config["context_length"],
                    max_tokens=lm_config.gen_config["max_tokens"],
                    stop_token=None,
                )
        elif "codellama" in lm_config.model.lower():
            caption, caption_name = args.caption, args.caption_name
            caption_path = os.path.join(root, caption_name)
            # use codellama instruction as "chat" mode
            if lm_config.mode == "chat":
                script_path = "/home/data2/stian/MMInA/models/codellama_gen.py"
                modified_prompt = {"role": "user", "content": ""}
                modified_prompt["content"] = prompt[0]["content"] + prompt[-1]["content"]
                if caption:
                    print(f"================Using BLIP captioning: {caption} =========================")
                    with open(caption_path, "r") as f:
                        captions = f.read()
                        modified_prompt['content'] = modified_prompt['content'] + "IMAGE CAPTIONS:\n" + captions
                # save as temperory json file
                temp_path = "/home/data2/stian/MMInA/sampling_data/temp.json"
                with open(temp_path, "w") as f:
                    json.dump(modified_prompt, f)

                result = subprocess.run(
                    [
                        "torchrun",
                        script_path,
                        "--instruction_path",
                        temp_path,
                        "--ckpt_dir",
                        "/home/data2/stian/MMInA/models/CodeLlama-7b-Instruct/",
                        "--tokenizer_path",
                        "/home/data2/stian/MMInA/models/CodeLlama-7b-Instruct/tokenizer.model",
                        "--max_seq_len",
                        "1024",
                        "--max_batch_size",
                        "1",
                    ],
                    stdout=subprocess.PIPE,
                )
                # Access the output
                # response = result.stdout.decode('utf-8').split("Here is the next action I will perform:\n\n")[-1]
                response = "```" + result.stdout.decode("utf-8").split(
                    "In summary, the next action I will perform is ```"
                )[-1].split("```")[0] + "```"
                # response = result.stdout.decode('utf-8')
                print(response)
                # response = wrapper_response(response)
                
            elif lm_config.mode == "completion":
                response = model.text_completion(
                    prompts,
                    max_gen_len=None,
                    temperature=lm_config.gen_config["temperature"],
                    top_p=lm_config.gen_config["top_p"],
                )
        elif "fuyu" in lm_config.model.lower():
            if lm_config.mode == "chat":
                # prepare input_image for the model
                imgpath = "/home/data2/stian/MMInA/sampling_data/imgs_caption/output_img.png"
                img = Image.open(imgpath)

                processor = processor_fuyu
                model = model_fuyu

                # prepare input_text for the model
                modified_prompt = {"role": "user", "content": ""}
                for i in range(len(prompt)):
                    p = prompt[i]["content"]
                    # p = p.replace("\n", " ")
                    # p = p.replace("\t", " ")
                    modified_prompt["content"] += p + "\n"
                # save as temperory json file
                temp_path = "/home/data2/stian/MMInA/sampling_data/temp_fuyu.json"
                with open(temp_path, "w") as f:
                    json.dump(modified_prompt, f)
                text_prompt = modified_prompt["content"]

                # preprocess input for the model
                inputs = processor(text=text_prompt, images=img, return_tensors="pt").to("cuda:0")

                # autoregressively generate text
                generation_output = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    pad_token_id=model.config.eos_token_id,
                )
                response = processor.batch_decode(
                    generation_output[:, :],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )[0]  # return the string format of the response
                response = wrapper_response_fuyu(response)

            elif lm_config.mode == "completion":
                pass

        elif "mind2act" in lm_config.model.lower():
            pass
        elif "webshop" in lm_config.model.lower():
            assistant = "GPT"
            caption, caption_name = args.caption, args.caption_name
            caption_path = os.path.join(root, caption_name)
            def bart_predict(input, model, skip_special_tokens=True, **kwargs):
                input_ids = bart_tokenizer(input)["input_ids"]
                input_ids = torch.tensor(input_ids).unsqueeze(0)
                output = model.generate(input_ids, max_length=512, **kwargs)
                return bart_tokenizer.batch_decode(
                    output.tolist(), skip_special_tokens=skip_special_tokens
                )

            from transformers import (
                BartForConditionalGeneration,
                BartTokenizer,
            )

            bart_tokenizer = BartTokenizer.from_pretrained(
                "facebook/bart-large"
            )
            print("bart tokenizer loaded")
            # Load pretrained bart model
            bart_path = "/home/data2/stian/MMInA/models/webshop/ckpts/web_search/checkpoint-800"
            bart_model = BartForConditionalGeneration.from_pretrained(
                bart_path, torch_dtype=torch.bfloat16
            )
            print("bart model loaded", bart_path)
            with open("/home/data2/stian/MMInA/accessibility_tree.json") as f:
                data = json.load(f)
                acc_tree = data[0]
                print("accessibility tree loaded")
            # Set your API key here
            openai.api_key = ""
            if caption: 
                print(f"================Using BLIP captioning: {caption} =========================")
                # Load captions
                with open(caption_path, "r") as f:
                    captions = f.read()
                    prompt[-1]['content'] = prompt[-1]['content'] + "IMAGE CAPTIONS:\n" + captions + "\nGenerate 10 valid actions in str within a str of List either starts with 'search[' or 'click[id]' based on the given OBSERVATION, OBJECTIVE, PREVIOUS ACTION.\n"
            else:
                prompt[-1]["content"] += "\nGenerate 10 valid actions in str within a str of List either starts with 'search[' or 'click[id]' based on the given OBSERVATION, OBJECTIVE, PREVIOUS ACTION in this format: ['search[ ]', 'click[]', ...]\n"
            if assistant == "GPT":
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-1106",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.\n\nHere's the information you'll have:\nThe user's objective: This is the task you're trying to complete.\nThe current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.\nThe current web page's URL: This is the page you're currently navigating.\nThe open tabs: These are the tabs you have open.\nThe previous action: This is the action you just performed. It may be helpful to track your progress.\nThe current images' captions on this page. It may be helpful to provide more information.\n\nYou should generate a list of 10 valid actions either starts with 'search[' or 'click[id]' based on the given OBSERVATION, OBJECTIVE, PREVIOUS ACTION.",
                        },
                        # {
                        #     "role": "user",
                        #     "content": "OBSERVATION:\n[1744] link 'HP CB782A#ABA 640 Inkjet Fax Machine (Renewed)'\n\t\t[1749] StaticText '$279.49'\n\t\t[1757] button 'Add to Cart'\n\t\t[1760] button 'Add to Wish List'\n\t\t[1761] button 'Add to Compare'\nURL: http://onestopmarket.com/office-products/office-electronics.html\nOBJECTIVE: What is the price of HP Inkjet Fax Machine\nPREVIOUS ACTION: None\nLet's think step-by-step. This page list the information of HP Inkjet Fax Machine, which is the product identified in the objective. Its price is $279.49. I think I have achieved the objective. I will issue the stop action with the answer. In summary, the next action I will perform is ```stop [$279.49]```\nOBSERVATION:\n[164] textbox 'Search' focused: True required: False\n[171] button 'Go'\n[174] link 'Find directions between two points'\n[212] heading 'Search Results'\n[216] button 'Close'\nURL: http://openstreetmap.org\nOBJECTIVE: Show me the restaurants near CMU\nPREVIOUS ACTION: None\nLet's think step-by-step. This page has a search box whose ID is [164]. According to the nominatim rule of openstreetmap, I can search for the restaurants near a location by \"restaurants near\". I can submit my typing by pressing the Enter afterwards. In summary, the next action I will perform is ```type [164] [restaurants near CMU] [1]```\nOBSERVATION:\nTab 0 (current): Magento Admin\n\n[1] RootWebArea 'Magento Admin' focused: True\n\t[11] HeaderAsNonLandmark ''\n\t\t[13] link 'Magento Admin Panel'\n\t\t\t[16] img 'http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7780/static/version1681922233/adminhtml/Magento/backend/en_US/images/magento-logo.svg'\n\t[7] group 'Welcome, please sign in'\n\t\t[20] LineBreak '\\n'\n\t\t[35] StaticText 'Username'\n\t\t[10] textbox 'Username *' focused: True required: False\n\t\t[40] StaticText 'Password'\n\t\t[31] textbox 'Password *' required: False\n\t\t[23] LayoutTable ''\n\t\t\t[32] button 'Sign in'\n\t\t\t[33] link 'Forgot your password?'\n\t[12] FooterAsNonLandmark ''\n\t\t[14] link 'Magento'\n\t\t[15] StaticText 'Copyright \u00a9 2023 Magento Commerce Inc. All rights reserved.'\nURL: http://luma.com/admin\nOBJECTIVE: i am looking for blue color toothbrushes that helps to maintain my oral hygiene, and price lower than 60.00 dollars\nPREVIOUS ACTION: None\nIMAGE CAPTIONS: magento logo with orange and black background\n",
                        # },
                        {
                            "role": "assistant",
                            "content": "['search[blue oral hygiene toothbrushes]', 'search[blue oral hygiene toothbrush]', 'search[blue dental hygiene toothbrushes]', 'search[blue oral hygiene teethbrushes]', 'search[blue dental hygiene toothbrush]', 'click[1234]', 'search[blue oral hygiene toothpaste]', 'search[blue oral hygiene toothbrushing]', 'click[1744]', 'search[blue toothbrushes]', 'stop [$279.49]']",
                        },
                        prompt[-1]
                    ],
                )
                valid_acts = response["choices"][0]["message"]["content"]
                import ast
                valid_acts = ast.literal_eval(valid_acts)
            elif assistant == "Gemini":
                model = genai.GenerativeModel('gemini-pro')
                model_input = prompt[-1]['content'] + "\nGenerate 10 valid actions in str within a str of List either starts with 'search[' or 'click[id]' based on the given OBSERVATION, OBJECTIVE, PREVIOUS ACTION.\n"
                response = model.generate_content(model_input)
                to_markdown(response.text)
                response = response.candidates[0].content.parts[0].text
                import re
                pattern = r'(search|click)(\[.*?\])'
                matching_strings = re.findall(pattern, response)
                valid_acts = []
                for match in matching_strings:
                    action_type, content_with_brackets = match
                    valid_act = action_type + content_with_brackets
                    valid_acts.append(valid_act)

            if valid_acts[0].startswith("search"):
                query = bart_predict(intent, bart_model, num_return_sequences=5, num_beams=5)
                query = query[0]
                action = f"search[{query}]"
            else:
                action = valid_acts[0]

            response = wrapper_response_webshop(action)
        elif "gemini-pro" == lm_config.model.lower():
            caption, caption_name = args.caption, args.caption_name
            caption_path = os.path.join(root, caption_name)
            modified_prompt = ""
            for i in range(len(prompt)):
                p = prompt[i]["content"]
                modified_prompt += p + "\n"
            if caption:
                print(f"================Using BLIP captioning: {caption} =========================")
                with open(caption_path, "r") as f:
                    captions = f.read()
                    modified_prompt += modified_prompt + "IMAGE CAPTIONS:\n" + captions
                    
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(modified_prompt)
            to_markdown(response.text)
            response = response.candidates[0].content.parts[0].text
            
        elif "gemini-pro-vision" == lm_config.model.lower():
            # prepare input_image for the model
            imgpath = "/home/data2/stian/MMInA/sampling_data/imgs_gemini/output_img.jpg"
            try:
                img = Image.open(imgpath)
            except:
                img = Image.open("/home/data2/stian/MMInA/white.jpg")
            model = genai.GenerativeModel("gemini-pro-vision")
            modified_prompt = ""
            for i in range(len(prompt)):
                p = prompt[i]["content"]
                modified_prompt += p + "\n"
            response = model.generate_content(
                [modified_prompt, img], stream=True
            )
            response.resolve()
            response = response.text
            print(response)
            response_path = f"{folder_path2}/response.json"
            with open(response_path, 'w', encoding='utf-8') as f:
                json.dump(response, f, ensure_ascii=False, indent=4)
        elif "gpt4v" == lm_config.model.lower():
            # Prepare prompts
            extracted_prompt = ""
            for i in range(len(prompt)):
                p = prompt[i]["content"]
                extracted_prompt += p + "\n"
            
            # Prepare imgs
            def get_image_paths(folder):
                supported_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
                img_paths = []
                for filename in os.listdir(folder):
                    if any(filename.lower().endswith(ext) for ext in supported_extensions):
                        img_paths.append(os.path.join(folder, filename))
                return img_paths
            folder_path = "/home/data2/stian/MMInA/imgbin"
            img_paths = get_image_paths(folder_path)
            print(f"img_paths: {img_paths}")
            force_prompt = '\nGenerate the action in the correct format. Start with a "In summary, the next action I will perform is" phrase, followed by action inside ``````'
            client = GPT4VisionClient()
            response = client.query(extracted_prompt + force_prompt, img_paths)
            print(response)
            file_path2 = f"{folder_path2}/response.json"
            with open(file_path2, 'w', encoding='utf-8') as json_file2:
                json.dump(response, json_file2, ensure_ascii=False, indent=4)
        elif "gpt4" == lm_config.model.lower():
            caption, caption_name = args.caption, args.caption_name
            caption_path = os.path.join(root, caption_name)
            # Set your API key here
            openai.api_key = ""
            if caption:
                print(f"================Using BLIP captioning: {caption} =========================")
                print(f"caption_path: {caption_path, os.path.exists(caption_path)}")
                with open(caption_path, "r") as f:
                    captions = f.read()
                    # captions_t = remov_duplicates(captions)
                    prompt[-1]['content'] = prompt[-1]['content'] + "IMAGE CAPTIONS:\n" + captions
            response = openai.ChatCompletion.create(
                model="gpt-4-0125-preview",
                messages=prompt,
            )
            response = response["choices"][0]["message"]["content"]
        elif "llava" in lm_config.model.lower():
            model = model_llava
        else:
            raise ValueError(
                f"Your custom model {lm_config.model} do not support mode {lm_config.mode}"
            )

        try:
            parsed_response = self.prompt_constructor.extract_action(response)
            if self.action_set_tag == "id_accessibility_tree":
                action = create_id_based_action(parsed_response)
            elif self.action_set_tag == "playwright":
                action = create_playwright_action(parsed_response)
            else:
                raise ValueError(f"Unknown action type {self.action_set_tag}")

            action["raw_prediction"] = response

        except ActionParsingError as e:
            action = create_none_action()
            action["raw_prediction"] = response

        return action

    def reset(self, test_config_file: str) -> None:
        pass


# Wrapper for get_response
def wrapper_response(res: str) -> str:
    idx_1 = 46 + res.find("In summary, the next action I will perform is ")
    idx_2 = res.find(".\n\nPlease")
    action = res[idx_1:idx_2]
    action = "```" + action + "```"
    return res[:idx_1] + action + res[idx_2:]


def wrapper_response_fuyu(res: str) -> str:
    res = "```" + res.split("\x04 ")[1].replace("\n", "") + "```"
    return res


def wrapper_response_webshop(res: str) -> str:
    res = "```" + res + "```"
    return res


# Get response from language-only model
def get_response_lo(prompt: str, model=None, image_processor=None) -> str:
    lang_x = model.text_tokenizer(
        [
            get_formatted_prompt(prompt),
        ],
        return_tensors="pt",
    )

    model_dtype = next(model.parameters()).dtype

    lang_x_input_ids = lang_x["input_ids"]
    lang_x_attention_mask = lang_x["attention_mask"]

    generated_text = model.generate(
        lang_x=lang_x_input_ids.to(model.device),
        attention_mask=lang_x_attention_mask.to(model.device),
        max_new_tokens=512,
        num_beams=3,
        no_repeat_ngram_size=3,
        # pad_token_id=model.text_tokenizer.eos_token_id,
    )
    parsed_output = (
        model.text_tokenizer.decode(generated_text[0])
        .split("<answer>")[-1]
        .lstrip()
        .rstrip()
        .split("<|endofchunk|>")[0]
        .lstrip()
        .rstrip()
        .lstrip('"')
        .rstrip('"')
    )
    return parsed_output

def to_markdown(text):
    import textwrap
    from IPython.display import Markdown
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def remov_duplicates(input):

    # split input string separated by space
    input = input.split(" ")

    # now create dictionary using counter method
    # which will have strings as key and their 
    # frequencies as value
    UniqW = Counter(input)

    # joins two adjacent elements in iterable way
    s = " ".join(UniqW.keys())
    return s

def bart_predict(input, model, skip_special_tokens=True, **kwargs):
    input_ids = bart_tokenizer(input)["input_ids"]
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    output = model.generate(input_ids, max_length=512, **kwargs)
    return bart_tokenizer.batch_decode(
        output.tolist(), skip_special_tokens=skip_special_tokens
    )


def predict(obs, info, model, softmax=False, rule=False, bart_model=None):
    valid_acts = info["valid"]
    if valid_acts[0].startswith("search["):
        if bart_model is None:
            return valid_acts[-1]
        else:
            goal = process_goal(obs)
            query = bart_predict(
                goal, bart_model, num_return_sequences=5, num_beams=5
            )
            # query = random.choice(query)  # in the paper, we sample from the top-5 generated results.
            query = query[
                0
            ]  # ... but use the top-1 generated search will lead to better results than the paper results.
            return f"search[{query}]"

    if rule:
        item_acts = [
            act for act in valid_acts if act.startswith("click[item - ")
        ]
        if item_acts:
            return item_acts[0]
        else:
            assert "click[buy now]" in valid_acts
            return "click[buy now]"

    state_encodings = tokenizer(
        process(obs), max_length=512, truncation=True, padding="max_length"
    )
    action_encodings = tokenizer(
        list(map(process, valid_acts)),
        max_length=512,
        truncation=True,
        padding="max_length",
    )
    batch = {
        "state_input_ids": state_encodings["input_ids"],
        "state_attention_mask": state_encodings["attention_mask"],
        "action_input_ids": action_encodings["input_ids"],
        "action_attention_mask": action_encodings["attention_mask"],
        "sizes": len(valid_acts),
        "images": info["image_feat"].tolist(),
        "labels": 0,
    }
    batch = data_collator([batch])
    # make batch cuda
    batch = {k: v.cuda() for k, v in batch.items()}
    outputs = model(**batch)
    if softmax:
        idx = torch.multinomial(F.softmax(outputs.logits[0], dim=0), 1)[
            0
        ].item()
    else:
        idx = outputs.logits[0].argmax(0).item()
    return valid_acts[idx]


def get_formatted_prompt(prompt: str) -> str:
    return f"<image>User: {prompt} GPT:<answer>"


def get_response(image, prompt: str, model=None, image_processor=None) -> str:
    input_data = image

    if isinstance(input_data, Image.Image):
        if input_data.size == (224, 224) and not any(
            input_data.getdata()
        ):  # Check if image is blank 224x224 image
            vision_x = torch.zeros(
                1, 1, 1, 3, 224, 224, dtype=next(model.parameters()).dtype
            )
        else:
            vision_x = (
                image_processor.preprocess([input_data], return_tensors="pt")[
                    "pixel_values"
                ]
                .unsqueeze(1)
                .unsqueeze(0)
            )  # 1x1x1x3x224x224
    else:
        raise ValueError("Invalid input data. Expected PIL Image.")

    lang_x = model.text_tokenizer(
        [
            get_formatted_prompt(prompt),
        ],
        return_tensors="pt",
    )

    model_dtype = next(model.parameters()).dtype

    vision_x = vision_x.to(dtype=model_dtype)
    lang_x_input_ids = lang_x["input_ids"]
    lang_x_attention_mask = lang_x["attention_mask"]

    generated_text = model.generate(
        vision_x=vision_x.to(model.device),
        lang_x=lang_x_input_ids.to(model.device),
        attention_mask=lang_x_attention_mask.to(model.device),
        max_new_tokens=512,
        num_beams=3,
        no_repeat_ngram_size=3,
    )
    parsed_output = (
        model.text_tokenizer.decode(generated_text[0])
        .split("<answer>")[-1]
        .lstrip()
        .rstrip()
        .split("<|endofchunk|>")[0]
        .lstrip()
        .rstrip()
        .lstrip('"')
        .rstrip('"')
    )
    return parsed_output


def construct_llm_config(args: argparse.Namespace) -> lm_config.LMConfig:
    llm_config = lm_config.LMConfig(
        provider=args.provider, model=args.model, mode=args.mode
    )

    if args.provider == "openai":
        llm_config.gen_config["temperature"] = args.temperature
        llm_config.gen_config["top_p"] = args.top_p
        llm_config.gen_config["context_length"] = args.context_length
        llm_config.gen_config["max_tokens"] = args.max_tokens
        llm_config.gen_config["stop_token"] = args.stop_token
        llm_config.gen_config["max_obs_length"] = args.max_obs_length
    elif args.provider == "custom":
        if "otter" in args.model.lower():
            llm_config.gen_config[
                "temperature"
            ] = args.pt_model.config.temperature
            llm_config.gen_config["top_p"] = args.pt_model.config.top_p
            llm_config.gen_config["context_length"] = args.pt_model.config.max_length
            llm_config.gen_config["max_tokens"] = args.max_tokens
            llm_config.gen_config["stop_token"] = args.stop_token
            llm_config.gen_config["max_obs_length"] = args.max_obs_length
        elif "wizardlm" in args.model.lower():
            sampling_params = SamplingParams(args.temperature, args.top_p)
            llm_config.gen_config["temperature"] = args.temperature
            llm_config.gen_config["top_p"] = args.top_p
            # llm_config.gen_config["context_length"] =
            llm_config.gen_config["max_tokens"] = args.max_tokens
        elif "codellama" in args.model.lower():
            llm_config.gen_config["temperature"] = args.temperature
            llm_config.gen_config["top_p"] = args.top_p
            llm_config.gen_config["context_length"] = 4096
            llm_config.gen_config["max_tokens"] = args.max_tokens
            llm_config.gen_config["max_obs_length"] = args.max_obs_length
        elif "fuyu" in args.model.lower():
            llm_config.gen_config["temperature"] = args.temperature
            llm_config.gen_config["top_p"] = args.top_p
            llm_config.gen_config["context_length"] = 4096
            llm_config.gen_config["max_tokens"] = args.max_tokens
            llm_config.gen_config["max_obs_length"] = args.max_obs_length
        elif "webshop" in args.model.lower():
            llm_config.gen_config["temperature"] = args.temperature
            llm_config.gen_config["top_p"] = args.top_p
            llm_config.gen_config["context_length"] = 4096 # 4096
            llm_config.gen_config["max_tokens"] = args.max_tokens
            llm_config.gen_config["max_obs_length"] = args.max_obs_length
        elif "gemini" in args.model.lower():
            llm_config.gen_config["temperature"] = args.temperature
            llm_config.gen_config["top_p"] = args.top_p
            llm_config.gen_config["context_length"] = 4096
            llm_config.gen_config["max_tokens"] = args.max_tokens
            llm_config.gen_config["stop_token"] = args.stop_token
            llm_config.gen_config["max_obs_length"] = args.max_obs_length
        elif "gpt" in args.model.lower():
            llm_config.gen_config["temperature"] = args.temperature
            llm_config.gen_config["top_p"] = args.top_p
            llm_config.gen_config["context_length"] = 4096
            llm_config.gen_config["max_tokens"] = args.max_tokens
            llm_config.gen_config["stop_token"] = args.stop_token
            llm_config.gen_config["max_obs_length"] = args.max_obs_length
        elif "llava" in args.model.lower():
            llm_config.gen_config["temperature"] = args.temperature
            llm_config.gen_config["top_p"] = args.top_p
            llm_config.gen_config["context_length"] = 4096
            llm_config.gen_config["max_tokens"] = args.max_tokens
            llm_config.gen_config["stop_token"] = args.stop_token
            llm_config.gen_config["max_obs_length"] = args.max_obs_length         
    else:
        raise NotImplementedError(f"provider {args.provider} not implemented")
    return llm_config


def construct_agent(args: argparse.Namespace) -> Agent:
    llm_config = construct_llm_config(args)

    agent: Agent
    if args.agent_type == "teacher_forcing":
        agent = TeacherForcingAgent()
    elif args.agent_type == "prompt":
        with open(args.instruction_path) as f:
            constructor_type = json.load(f)["meta_data"][
                "prompt_constructor"
            ]  #'CoTPromptConstructor'

        # Define tokenizer
        if args.provider == "openai":
            tokenizer = tiktoken.encoding_for_model(llm_config.model)
            # tokenizer.encode returns a list of token ids [51, 1609, 30694, 374, 2523, 0]
            # tokenizer.decode () returns a string, and get an input of list of token ids

        elif args.provider == "custom":
            if "otter" in args.model.lower():
                tokenizer = args.pt_model.text_tokenizer

            # TODO: add wizardlm tokenizer
            elif "wizardlm" in args.model.lower():
                tokenizer = args.pt_model.text_tokenizer  # ??
            elif "codellama" in args.model.lower():
                tokenizer_path = "/home/data2/stian/MMInA/models/CodeLlama-7b-Instruct/tokenizer.model"
                tokenizer = Tokenizer(model_path=tokenizer_path)
            elif "fuyu" in args.model.lower():
                tokenizer = processor_fuyu.tokenizer
            elif "webshop" in args.model.lower():
                from transformers import BartTokenizer
                tokenizer = BartTokenizer.from_pretrained(
                    "facebook/bart-large", torch_dtype=torch.bfloat16
                )
            elif "gemini" in args.model.lower():
                tokenizer = tiktoken.encoding_for_model("gpt-4-vision-preview")
            elif "gpt" in args.model.lower():
                tokenizer = tiktoken.encoding_for_model("gpt-4-vision-preview")
            elif "llava" in args.model.lower():
                tokenizer = tokenizer_llava
            else:
                raise NotImplementedError(
                    f"Provider {args.provider} not implemented."
                )
        prompt_constructor = eval(constructor_type)(
            args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        agent = PromptAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
        )
    else:
        raise NotImplementedError(
            f"agent type {args.agent_type} not implemented"
        )
    return agent