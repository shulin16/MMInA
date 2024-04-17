"""Script to run end-to-end evaluation on the benchmark"""
import argparse
import glob
import json
import logging
import os
import random
import time
from pathlib import Path

import openai
import torch
from beartype import beartype

# Setup the environment
os.environ[
"SHOPPING"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7770"
os.environ[
    "SHOPPING_ADMIN"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7780/admin"
os.environ[
    "REDDIT"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:9999"
os.environ[
    "GITLAB"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8023"
os.environ[
    "MAP"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000"
os.environ[
    "WIKIPEDIA"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
os.environ[
    "HOMEPAGE"
] = "PASS"  # The home page is not currently hosted in the demo site
print("Done setting up URLs")

# from vllm import LLM, SamplingParams
from models.llama import Llama

# Custom model imports
# from models.otter.modeling_otter import OtterForConditionalGeneration

from agent import (
    Agent,
    PromptAgent,
    TeacherForcingAgent,
    construct_agent,
)
from agent.prompts import *
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.actions import is_equivalent
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from evaluation_harness import evaluator_router

LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the browser"
    )
    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=["accessibility_tree", "html", "image"],
        default="accessibility_tree",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=720)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)

    parser.add_argument("--max_steps", type=int, default=30)

    parser.add_argument(
        "--save_dir_sampling",
        type=str,
        default="/home/data2/stian/MMInA/sampling_data/",
    ) # Not in use

    # agent config
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="agent/prompts/jsons/p_cot_id_actree_2s.json",
    )
    parser.add_argument(
        "--parsing_failure_th",
        help="When concesecutive parsing failure exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When concesecutive repeating action exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )
    parser.add_argument("--domain", type=str, default="samplesshopping")
    parser.add_argument("--hist", type=bool, default=False)
    parser.add_argument("--hist_fold", type=str, default="/home/data2/stian/MMInA/results_mm/gemini-pro-vision")
    parser.add_argument("--hist_num", type=int, default=1)
    parser.add_argument("--caption", type=bool, default=False)
    parser.add_argument("--caption_name", type=str, default="captions.txt")
    parser.add_argument("--cnt1", type=int, default=0)
    parser.add_argument("--cnt2", type=int, default=0)
    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--pt_model", default=None)
    parser.add_argument("--pt_processor", default=None)
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=800)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=1920,
    )

    # example config
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=1000)

    # logging related
    parser.add_argument("--result_dir", type=str, default="")
    args = parser.parse_args()

    # check the whether the action space is compatible with the observation space
    if (
        args.action_set_tag == "id_accessibility_tree"
        and args.observation_type != "accessibility_tree"
    ):
        raise ValueError(
            f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}"
        )

    return args

@beartype
def early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int]
) -> tuple[bool, str]:
    """Check whether need to early stop"""

    # reach the max step
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    last_k_actions: list[Action]
    action_seq: list[Action]

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(
            [
                action["action_type"] == ActionTypes.NONE
                for action in last_k_actions
            ]
        ):
            return True, f"Failed to parse actions for {k} times"

    # Case: same action for k times
    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if len(action_seq) == 0:
        return False, ""

    last_action: Action = action_seq[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k_actions) >= k:
            if all(
                [
                    is_equivalent(action, last_action)
                    for action in last_k_actions
                ]
            ):
                return True, f"Same action for {k} times"

    else:
        # check the action sequence
        if (
            sum([is_equivalent(action, last_action) for action in action_seq])
            >= k
        ):
            return True, f"Same typing action for {k} times"

    return False, ""

@beartype
def test(
    args: argparse.Namespace,
    agent: Agent | PromptAgent | TeacherForcingAgent,
    config_file_list: list[str],
) -> None:
    scores = []    
    max_steps = args.max_steps

    early_stop_thresholds = {
        "parsing_failure": args.parsing_failure_th,
        "repeating_action": args.repeating_action_failure_th,
    }

    env = ScriptBrowserEnv(
        headless=True,
        slow_mo=args.slow_mo,
        observation_type=args.observation_type,
        current_viewport_only=args.current_viewport_only,
        viewport_size={
            "width": args.viewport_width,
            "height": args.viewport_height,
        },
        save_trace_enabled=args.save_trace_enabled,
        sleep_after_execution=args.sleep_after_execution,
        caption = args.caption,
        caption_name=args.caption_name,
    )
    all_pass = 0
    all_all = 0
    for config_file in config_file_list:
        try:
            render_helper = RenderHelper(
                config_file, args.result_dir, args.action_set_tag
            )

            # get intent
            with open(config_file) as f:
                _c = json.load(f)
                intent = _c["intent"]
                task_id = _c["task_id"]
                # hop_cnt = _c["hop_cnt"]

            numbers = re.findall(r'\d+',config_file)
            args.cnt1 = int(numbers[0]) if numbers else None
            folder_path = f"./full_result/{args.cnt1}"
            args.cnt2 = 0
            
            os.makedirs(folder_path,exist_ok=True)
            logger.info(f"[Config file]: {config_file}")
            logger.info(f"[Intent]: {intent}")

            agent.reset(config_file)
            trajectory: Trajectory = []
            obs, info = env.reset(
                options={"config_file": config_file}, cnt1=args.cnt1, cnt2=args.cnt2
            )  # file path -> json file of webstate
            state_info: StateInfo = {"observation": obs, "info": info, "current_url": obs["current_url"]}
            # print("State info: ", state_info)
            trajectory.append(state_info)
            # print("==========trajectory==========", trajectory)
            current_url = obs["current_url"]
            print("CURRENT: ",current_url)
            
            meta_data = {"action_history": ["None"]}
            
            print("config_file: ",config_file)
            check_list = []
            # url = []
            
            cnt_ans = 0
            
            ## 2hop implementations
            cnt_cl = 2
            with open (config_file,"r") as file:
                data = json.load(file)
                reference = data['eval']['reference_answers']['must_include']
            nxt = reference[0]
            
            all_all = all_all + cnt_cl
            
            flag = True
            all_view_url = []
            
            meta_data = {"action_history": ["None"]}
            while True:
                current_url = current_url.lower()
                all_view_url.append(current_url)
                now_url = check_list[cnt_ans]
                if flag:
                    if "kiwix" not in current_url and current_url!=None and "kiwix" in now_url and check_list[cnt_ans+1] in current_url:
                        cnt_ans = cnt_ans + 1
                        flag = False
                    if "momondo" in current_url and "momondo" in now_url:
                        if flight in current_url:
                            cnt_ans = cnt_ans + 1
                    if "7770" in current_url and "7770" in now_url:
                        if shop in current_url:
                            cnt_ans = cnt_ans + 1
                    if now_url in current_url:
                        if " " in city:
                            fro,en = city.split(" ")
                            if fro in current_url and en in current_url:
                                cnt_ans = cnt_ans + 1
                        else:
                            if city in current_url:
                                cnt_ans = cnt_ans +1
                print("ckpt1")
                
                if "kiwix" not in current_url and nxt in current_url:
                    if flag:
                        cnt_ans = cnt_ans+1
                        flag = False
                fg = True
                for item in reference:
                    item=item.lower()
                    if item not in current_url:
                        fg= False
                if fg and cnt_ans == 1: cnt_ans = 2
                
                early_stop_flag, stop_info = early_stop(
                    trajectory, max_steps, early_stop_thresholds
                )

                if early_stop_flag:
                    action = create_stop_action(f"Early stop: {stop_info}")
                else:
                    try:
                        folder_path = f"./full_result/{args.cnt1}/{args.cnt2}"
                        os.makedirs(folder_path, exist_ok=True)
                        if args.provider == "openai":
                            action = agent.next_action(
                                trajectory, intent, meta_data=meta_data
                            )
                            args.cnt2 = args.cnt2 + 1
                        elif args.provider == "custom":
                            action = agent.next_action_custom(
                                trajectory,
                                intent,
                                meta_data=meta_data,
                                model=args.pt_model,
                                args=args,
                            )
                            args.cnt2 = args.cnt2 + 1
                        else:
                            raise NotImplementedError(
                                f"Provider {args.provider} not implemented"
                            )
                        print("action", action)
                    except ValueError as e:
                        # get the error message
                        action = create_stop_action(f"ERROR: {str(e)}")

                trajectory.append(action)
                # print("==========trajectory==========", trajectory)
                action_str = get_action_description(
                    action,
                    state_info["info"]["observation_metadata"],
                    action_set_tag=args.action_set_tag,
                    prompt_constructor=agent.prompt_constructor
                    if isinstance(agent, PromptAgent)
                    else None,
                )
                render_helper.render(
                    action, state_info, meta_data, args.render_screenshot
                )
                meta_data["action_history"].append(action_str)

                if action["action_type"] == ActionTypes.STOP:
                    break

                obs, _, terminated, _, info, current_url = env.step(action, args.cnt1, args.cnt2)
                print("CURRENT: ",current_url)
                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)

                if terminated:
                    # add a action place holder
                    trajectory.append(create_stop_action(""))
                    break
            all_pass = all_pass + cnt_ans

            logger.info(f"[Result] success rate: {cnt_ans}/{cnt_cl}")
            
            evaluator = evaluator_router(config_file)
            score = evaluator(
                trajectory=trajectory,
                config_file=config_file,
                page=env.page,
                client=env.get_page_client(env.page),
            )

            scores.append(score)
                
            # if check_list[cnt_ans]=="end":
            if cnt_ans==2:
                logger.info(f"[Result] (PASS) {config_file}")
            else:
                logger.info(f"[Result] (FAIL) {config_file}")

            if args.save_trace_enabled:
                env.save_trace(
                    Path(args.result_dir) / "traces" / f"{task_id}.zip"
                )
         
            print(f"Current score: {sum(scores) / len(scores)}")
            
            if args.save_trace_enabled:
                env.save_trace(
                    Path(args.result_dir) / "traces" / f"{task_id}.zip"
                )

        except openai.error.OpenAIError as e:
            logger.info(f"[OpenAI Error] {repr(e)}")
        except Exception as e:
            logger.info(f"[Unhandled Error] {repr(e)}]")
            import traceback

            # write to error file
            with open(Path(args.result_dir) / "error.txt", "a") as f:
                f.write(f"[Config file]: {config_file}\n")
                f.write(f"[Unhandled Error] {repr(e)}\n")
                f.write(traceback.format_exc())  # write stack trace to file
        logger.info(f"[Result] all success rate: {all_pass}/{all_all}")
        render_helper.close()

    env.close()
    logger.info(f"Average score: {sum(scores) / len(scores)}")    


def prepare(args: argparse.Namespace) -> None:
    # convert prompt python files to json
    from agent.prompts import to_json

    to_json.run()

    # prepare result dir
    result_dir = args.result_dir
    if not result_dir:
        result_dir = (
            f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
        )
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        logger.info(f"Create result dir: {result_dir}")

    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)

    # log the log file
    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")


def get_unfinished(config_files: list[str], result_dir: str) -> list[str]:
    result_files = glob.glob(f"{result_dir}/*.html")
    task_ids = [
        os.path.basename(f).split(".")[0].split("_")[1] for f in result_files
    ]
    unfinished_configs = []
    for config_file in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        if task_id not in task_ids:
            unfinished_configs.append(config_file)
    return unfinished_configs


@beartype
def dump_config(args: argparse.Namespace) -> None:
    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump config to {config_file}")


if __name__ == "__main__":
    args = config()
    args.sleep_after_execution = 2.5
    prepare(args)
    test_file_list = []
    st_idx = args.test_start_idx
    ed_idx = args.test_end_idx
    for i in range(st_idx, ed_idx):
        # test_file_list.append(f"config_files/{i}.json")
        test_file_list.append(f"mmina/{args.domain}/{i}.json")
    # test_file_list = get_unfinished(test_file_list, args.result_dir)
    print(f"Total {len(test_file_list)} tasks left")
    print(f"Initial history info: 1. Use history: {args.hist}; 2. History number: {args.hist_num}")
    args.render = True
    args.render_screenshot = True
    args.save_trace_enabled = True

    if args.provider == "custom":
        if args.model == "otter-7b":
            # import otter
            load_bit = "bf16"
            precision = {}
            if load_bit == "bf16":
                precision["torch_dtype"] = torch.bfloat16
            elif load_bit == "fp16":
                precision["torch_dtype"] = torch.float16
            elif load_bit == "fp32":
                precision["torch_dtype"] = torch.float32
            pt_model = OtterForConditionalGeneration.from_pretrained(
                "luodian/OTTER-Image-MPT7B",
                device_map="sequential",
                **precision,
            )
            pt_model.to(torch.device("cuda"))
            pt_model.lang_encoder.config.init_device = "cuda"
            pt_model.text_tokenizer.padding_side = "left"
            args.pt_model = pt_model
        elif args.model == "WizardLM/WizardCoder-Python-34B-V1.0":
            pt_model = LLM(model=args.model, tensor_parallel_size=n_gpus)
        elif args.model == "codellama-7b":
            ckpt_dir = "/home/data2/stian/MMInA/models/CodeLlama-7b-Instruct"
            tokenizer_path = "/home/data2/stian/MMInA/models/CodeLlama-7b-Instruct/tokenizer.model"
            max_seq_len = 4096
            max_batch_size = 4
        elif args.model == "fuyu-8b":
            pass
        elif args.model == "webshop":
            pass
        elif args.model == "gemini-pro-vision" or args.model == "gemini-pro":
            pass
        elif args.model == "gpt4" or args.model == "gpt4v":
            pass
        elif args.model =="llava-1.5":
            pass
        else:
            raise NotImplementedError(f"model {args.model} not implemented")

    args.current_viewport_only = True
    dump_config(args)

    agent = construct_agent(args)
    print(test_file_list)
    test(args, agent, test_file_list)
