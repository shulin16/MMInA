import argparse
import json
import os
import base64
import requests
from typing import Any


import openai
import tiktoken
import torch
from beartype import beartype
from beartype.door import is_bearable

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

# custom model implementation
import base64
import requests

from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter

root = "./cache/caption/"


class GPT4VisionClient:
    def __init__(self):
        # Set your API key here
        self.api_key = os.getenv("OPENAI_API_KEY") 
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def query(self, text, image_paths):
        content_items = [{"type": "text", "text": text}]
        for img_path in image_paths:
            base64_image = self.encode_image(img_path)
            content_items.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
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
        action_strs = action_seq.strip().split("\n") if isinstance(action_seq, str) else action_seq
        action_strs = [a.strip() for a in action_strs]

        actions = []
        for a_str in action_strs:
            try:
                if self.action_set_tag == "playwright":
                    cur_action = create_playwright_action(a_str)
                elif self.action_set_tag == "id_accessibility_tree":
                    cur_action = create_id_based_action(a_str)
                else:
                    raise ValueError(f"Unknown action type {self.action_set_tag}")
            except ActionParsingError:
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
    def extract_past_questions_multiple(self, cnt: int, hist_fold: str, num: int) -> str:
        import html2text
        html_path = hist_fold
        markdown_content = ""
        for i in range(cnt-1, cnt-1-num, -1):
            if i > 0:
                html_file = os.path.join(html_path, f'render_{i}.html')
                if not os.path.exists(html_file):
                    break
                with open(html_file, 'r', encoding='utf-8') as html_input:
                    html_content = html_input.read()
                    h = html2text.HTML2Text()
                    h.ignore_links = True
                    h.ignore_images = True
                    markdown_content += "\n" + h.handle(html_content)
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
                raise ValueError(f"OpenAI models do not support mode {lm_config.mode}")

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
        
        print("Current processing task: ", args.task_cnt," hop: ", args.hop_cnt)
        # switch of past history
        if args.hist:
            meta_data["past_history"] = self.extract_past_questions_multiple(args.task_cnt, args.hist_fold, args.hist_num)
            print(f"Using past {args.hist_num} histories...")
            # print("hist_fold: ",args.hist_fold)
        else:
            meta_data["past_history"] = ''
        prompt = self.prompt_constructor.construct(
            trajectory, intent, meta_data
        )
        lm_config = self.lm_config
        model_res_path = os.path.join(args.result_dir, lm_config.model, args.domain)
        if args.hist:
            model_res_path = os.path.join(model_res_path, f'hist_{args.hist_num}')
        task_res_path = os.path.join(model_res_path, f"task_{args.cnt1}")
        hop_res_path = os.path.join(task_res_path, f"hop_{args.cnt2}")
        os.makedirs(hop_res_path, exist_ok=True)
        
        # Save the prompt
        prompt_path = f"{hop_res_path}/prompt.json"
        with open(prompt_path, 'w', encoding='utf-8') as json_file:
            json.dump(prompt, json_file, ensure_ascii=False, indent=4)
            
        # =======baselines implementation=======

        if "cogagent" in lm_config.model.lower():
            print("Using cogagent...")
            from PIL import Image
            # Dictionary mapping format keys to format strings
            format_dict = {
                "action_op_sensitive": "(Answer in Action-Operation-Sensitive format.)",
                "status_plan_action_op": "(Answer in Status-Plan-Action-Operation format.)",
                "status_action_op_sensitive": "(Answer in Status-Action-Operation-Sensitive format.)",
                "status_action_op": "(Answer in Status-Action-Operation format.)",
                "action_op": "(Answer in Action-Operation format.)",
            }
            format_key = "action_op_sensitive"
            format_str = format_dict[format_key]
            
            platform_str = f"(Platform: Mac)\n"
            # Initialize history lists
            history_step = []
            history_action = []
            round_num = 1
            # image = Image.open(img_path).convert("RGB")
            image = trajectory[0]['observation']['image']
            image = Image.fromarray(image).convert("RGB")
            
            # Verify history lengths match
            if len(history_step) != len(history_action):
                raise ValueError("Mismatch in lengths of history_step and history_action.")

            # Format history steps for output
            history_str = "\nHistory steps: "
            for index, (step, action) in enumerate(zip(history_step, history_action)):
                history_str += f"\n{index}. {step}\t{action}"
            
            # Compose the query with task, platform, and selected format instructions
            task = intent
            query = f"Task: {task}{history_str}\n{platform_str}{format_str}"
            tokenizer = args.loaded_tokenizer
            
            prompt[-1]["image"] = image

            inputs = tokenizer.apply_chat_template(
                        prompt,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_tensors="pt",
                        return_dict=True,
                    ).to(model.device)
            # Generation parameters
            gen_kwargs = {
                "max_length": args.max_obs_length,
                "do_sample": True,
                "top_k": 50,
            }
            # Generate response
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs["input_ids"].shape[1]:]
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Model response:\n{response}")
        
        
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

# Prepare the image
def get_image_paths(folder):
    supported_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    img_paths = []
    for filename in os.listdir(folder):
        if any(filename.lower().endswith(ext) for ext in supported_extensions):
            img_paths.append(os.path.join(folder, filename))
    return img_paths

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
        llm_config.gen_config.update({
            "temperature": args.temperature,
            "top_p": args.top_p,
            "context_length": args.context_length,
            "max_tokens": args.max_tokens,
            "stop_token": args.stop_token,
            "max_obs_length": args.max_obs_length
        })
    elif args.provider == "custom":
        if "otter" in args.model.lower():
            llm_config.gen_config.update({
                "temperature": args.pt_model.config.temperature,
                "top_p": args.pt_model.config.top_p,
                "context_length": args.pt_model.config.max_length,
                "max_tokens": args.max_tokens,
                "stop_token": args.stop_token,
                "max_obs_length": args.max_obs_length
            })
        elif "wizardlm" in args.model.lower():
            llm_config.gen_config.update({
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens
            })
        elif "codellama" in args.model.lower():
            llm_config.gen_config.update({
                "temperature": args.temperature,
                "top_p": args.top_p,
                "context_length": 4096,
                "max_tokens": args.max_tokens,
                "max_obs_length": args.max_obs_length
            })
        elif "fuyu" in args.model.lower():
            llm_config.gen_config.update({
                "temperature": args.temperature,
                "top_p": args.top_p,
                "context_length": 4096,
                "max_tokens": args.max_tokens,
                "max_obs_length": args.max_obs_length
            })
        elif "webshop" in args.model.lower():
            llm_config.gen_config.update({
                "temperature": args.temperature,
                "top_p": args.top_p,
                "context_length": 4096,
                "max_tokens": args.max_tokens,
                "max_obs_length": args.max_obs_length
            })
        elif "gemini" in args.model.lower():
            llm_config.gen_config.update({
                "temperature": args.temperature,
                "top_p": args.top_p,
                "context_length": 4096,
                "max_tokens": args.max_tokens,
                "stop_token": args.stop_token,
                "max_obs_length": args.max_obs_length
            })
        elif "gpt" in args.model.lower():
            llm_config.gen_config.update({
                "temperature": args.temperature,
                "top_p": args.top_p,
                "context_length": 4096,
                "max_tokens": args.max_tokens,
                "stop_token": args.stop_token,
                "max_obs_length": args.max_obs_length
            })
        elif "llava" in args.model.lower():
            llm_config.gen_config.update({
                "temperature": args.temperature,
                "top_p": args.top_p,
                "context_length": 4096,
                "max_tokens": args.max_tokens,
                "stop_token": args.stop_token,
                "max_obs_length": args.max_obs_length
            })
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
            if args.model == "cogagent-9b":
                tokenizer = AutoTokenizer.from_pretrained(os.environ["HF_MODEL_ID"], trust_remote_code=True)
                args.loaded_tokenizer = tokenizer
            elif args.model == "your_customized_model":
                # add your model's tokenizer here
                pass
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
