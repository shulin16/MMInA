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

MMINA_DICT={
    'shopping': 200,
    'wikipedia': 308,
    'normal': 176,
    'compare': 100,
    'multi567': 180,
    'multipro': 86,
    'full': 1050
}  # total: 1050

print("Done setting up URLs")

# from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM


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
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{str(random.randint(0, 9999)).zfill(4)}.log"

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
        "--imgbin_dir",
        type=str,
        default="./cache/imgbin/",
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
    parser.add_argument("--domain", type=str, default="shopping", choices=['full', 'normal', 'multi567', 'compare', 'multipro', 'shopping', 'wikipedia'])
    parser.add_argument("--hist", action='store_true', default=False)
    parser.add_argument("--hist_fold", type=str, default="./cache/history/")
    parser.add_argument("--hist_num", type=int, default=1)
    parser.add_argument("--caption", action='store_true', default=False)
    parser.add_argument("--caption_name", type=str, default="captions.txt")
    parser.add_argument("--task_cnt", type=int, default=0)
    parser.add_argument("--hop_cnt", type=int, default=0)
    # lm config
    parser.add_argument("--provider", type=str, default="custom")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--loaded_tokenizer", default=None)
    parser.add_argument("--loaded_model", default=None)
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=500)
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
    scores = {}
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
    # initialize the counter to calculate hsr
    all_pass = 0
    all_all = 0
    for config_file in config_file_list:
        sub_domain = config_file.split('/')[1]
        scores[sub_domain] = {}
        scores[sub_domain]['task_scores'] = []
        scores[sub_domain]['hop_scores'] = []
        
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
            args.task_cnt = int(numbers[0]) if numbers else None
            args.hop_cnt = 0
            
            logger.info(f"[Config file]: {config_file}")
            logger.info(f"[Intent]: {intent}")

            agent.reset(config_file)
            trajectory: Trajectory = []
            obs, info = env.reset(
                options={"config_file": config_file}, task_cnt=args.task_cnt, hop_cnt=args.hop_cnt
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
            
            if is_domain_type(sub_domain, '2hop'):
                ## 2hop implementations
                cnt_cl = 2
                with open (config_file,"r") as file:
                    data = json.load(file)
                    reference = data['eval']['reference_answers']['must_include']
                nxt = reference[0]
            elif is_domain_type(sub_domain, 'multihop'):
                ## Multihop implementations
                with open (config_file,"r") as file:
                    data = json.load(file)
                    check_list = data['procedure']
                    city = data['city']
                    flight = data['flight']
                    shop = data['shop']
                cnt_cl = len(check_list)
            elif is_domain_type(sub_domain, 'singlehop'):
                cnt_cl = 1

            # calculate sum of total hop numbers of all tasks
            all_all += cnt_cl
            
            flag = True
            all_view_url = []
            
            meta_data = {"action_history": ["None"]}


            while True:
                current_url = current_url.lower()
                all_view_url.append(current_url)

                # if multihop tasks: check whether the current url is the last url in the check_list
                if is_domain_type(sub_domain, '2hop'):
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

                elif is_domain_type(sub_domain, 'multihop'):
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
                
                early_stop_flag, stop_info = early_stop(
                    trajectory, max_steps, early_stop_thresholds
                )

                if early_stop_flag:
                    action = create_stop_action(f"Early stop: {stop_info}")
                else:
                    try:
                        if args.provider == "openai":
                            action = agent.next_action(
                                trajectory, intent, meta_data=meta_data
                            )
                            args.hop_cnt = args.hop_cnt + 1
                        elif args.provider == "custom":
                            action = agent.next_action_custom(
                                trajectory,
                                intent,
                                meta_data=meta_data,
                                model=args.loaded_model,
                                args=args,
                            )
                            args.hop_cnt = args.hop_cnt + 1
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

                obs, _, terminated, _, info, current_url = env.step(action, args.task_cnt, args.hop_cnt)
                print("CURRENT: ",current_url)
                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)

                if terminated:
                    # add a action place holder
                    trajectory.append(create_stop_action(""))
                    break
            all_pass += cnt_ans

            # if is_domain_type(sub_domain, 'multihop'):
            #     logger.info(f"[Result] current single task success rate: {cnt_ans}/{cnt_cl}")
            # evaluate the scores
            evaluator = evaluator_router(config_file)
            score = evaluator(
                trajectory=trajectory,
                config_file=config_file,
                page=env.page,
                client=env.get_page_client(env.page),
            )
            
            '''
            
            criteria = {
                'shopping': lambda: score == 1,
                'wikipedia': lambda: score == 1,
                'normal': lambda: cnt_ans == 2,
                'compare': lambda: cnt_ans == 2,
                'multi567': lambda: check_list[cnt_ans] == "end",
                'multipro': lambda: check_list[cnt_ans] == "end"
            }
            if domain in criteria:
                result = "PASS" if criteria[domain]() else "FAIL"
                logger.info(f"[Result] ({result}) {config_file}")
            else:
                raise NotImplementedError(f"Domain {domain} not implemented")
            
            if args.save_trace_enabled:
                env.save_trace(
                    Path(args.result_dir) / "traces" / f"{task_id}.zip"
                )
            '''
            
            # Define success conditions for each domain type
            success_conditions = {
                'singlehop': lambda: score == 1,
                '2hop': lambda: cnt_ans == 2,
                'multihop': lambda: check_list[cnt_ans] == "end"
            }

            # Special handling for singlehop domain
            if is_domain_type(sub_domain, 'singlehop') and score == 1:
                cnt_ans += 1

            # Evaluate success based on domain type
            for domain_type, check_success in success_conditions.items():
                if is_domain_type(sub_domain, domain_type):
                    passed = check_success()
                    result = "PASS" if passed else "FAIL"
                    logger.info(f"[Result] ({result}) {config_file}")
                    scores[sub_domain]['task_scores'].append(int(passed))
                    break
            else:
                raise NotImplementedError(f"Unsupported domain type: {sub_domain}")
            
            scores[sub_domain]['task_success_rate'] = sum(scores[sub_domain]['task_scores']) / len(scores[sub_domain]['task_scores'])
            # update the success rate for each subdomain
            logger.info(f"[Result] Current task success rate: {scores[sub_domain]['task_success_rate']}")
            
            if not is_domain_type(sub_domain, 'singlehop'):
                scores[sub_domain]['hop_scores'].append(f"{cnt_ans}/{cnt_cl}")
                logger.info(f"[Result] Current hop success rate: {all_pass}/{all_all}")
                scores[sub_domain]['hop_success_rate'] = f"{all_pass}/{all_all}"
            
            if args.save_trace_enabled:
                env.save_trace(
                    Path(args.result_dir) / "traces" / f"{task_id}.zip"
                )

        except openai.OpenAIError as e:
            logger.info(f"[OpenAI Error] {repr(e)}")
        except Exception as e:
            logger.info(f"[Unhandled Error] {repr(e)}]")
            import traceback

            # write to error file
            with open(Path(args.result_dir) / "error.txt", "a") as f:
                f.write(f"[Config file]: {config_file}\n")
                f.write(f"[Unhandled Error] {repr(e)}\n")
                f.write(traceback.format_exc())  # write stack trace to file
        
        
        render_helper.close()

    env.close()

    with open(Path(args.result_dir) / "scores.json", "w") as f:
        json.dump(scores, f, indent=4)
    
    if args.domain == 'full':
        save_scores_to_json(scores, args.result_dir)

def prepare(args: argparse.Namespace) -> None:
    # convert prompt python files to json
    from agent.prompts import to_json

    to_json.run()

    datetime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    # prepare cache dir
    cache_dir = "./cache"
    if not Path(cache_dir).exists():
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Create cache dir: {cache_dir}")
    
    # prepare imgbin dir
    args.imgbin_dir = os.path.join(cache_dir, f"imgbin_{datetime}")
    if not Path(args.imgbin_dir).exists():
        Path(args.imgbin_dir).mkdir(parents=True, exist_ok=True)
        os.environ["IMG_BIN_DIR"] = args.imgbin_dir
        logger.info(f"Create imgbin dir: {args.imgbin_dir}")
    
    # prepare history dir if use history
    if args.hist:
        args.hist_fold = os.path.join(cache_dir, f"history_{args.model}_{datetime}")
        hist_fold = args.hist_fold
        if not Path(hist_fold).exists():
            Path(hist_fold).mkdir(parents=True, exist_ok=True)
            os.environ["HIST_DIR"] = args.hist_fold
            logger.info(f"Create history dir: {hist_fold}")
    
    # prepare result dir
    result_dir = args.result_dir
    if not result_dir:
        result_dir = (
            f"cache/results_{datetime}_{args.model}_{args.domain}"
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

def is_domain_type(domain, domain_type) -> bool:
    domain_map = {
        'multihop': ['compare', 'multi567', 'multipro'],
        '2hop': ['normal'],
        'singlehop': ['shopping', 'wikipedia']
    }
    return domain in domain_map.get(domain_type, [])

def save_scores_to_json(scores: dict, result_dir: str) -> None:
    """Save scores to JSON file organized by sub-domains"""    
    # Create scores directory if it doesn't exist
    scores_dir = Path(result_dir) / "scores"
    scores_dir.mkdir(exist_ok=True)
    
    # Calculating total scores
    results = {}
    results['task_success_rate'] = {}
    results['hop_success_rate'] = {}
    # Calculate total success rates
    results['task_success_rate']['hop1'] = (scores['shopping']['task_success_rate'] * MMINA_DICT['shopping'] + scores['wikipedia']['task_success_rate'] * MMINA_DICT['wikipedia']) / (MMINA_DICT['shopping'] + MMINA_DICT['wikipedia'])
    results['task_success_rate']['hop2-4'] = (scores['normal']['task_success_rate'] * MMINA_DICT['normal'] + scores['compare']['task_success_rate'] * MMINA_DICT['compare']) / (MMINA_DICT['normal'] + MMINA_DICT['compare'])
    results['task_success_rate']['hop5+'] = (scores['multi567']['task_success_rate'] * MMINA_DICT['multi567'] + scores['multipro']['task_success_rate'] * MMINA_DICT['multipro']) / (MMINA_DICT['multi567'] + MMINA_DICT['multipro'])
    
    results['hop_success_rate']['hop1'] = results['task_success_rate']['hop1']
    results['hop_success_rate']['hop2-4'] = (scores['normal']['hop_success_rate'].split('/')[0] + scores['compare']['hop_success_rate'].split('/')[0]) / (scores['normal']['hop_success_rate'].split('/')[1] + scores['compare']['hop_success_rate'].split('/')[1])
    results['hop_success_rate']['hop5+'] = (scores['multi567']['hop_success_rate'].split('/')[0] + scores['multipro']['hop_success_rate'].split('/')[0]) / (scores['multi567']['hop_success_rate'].split('/')[1] + scores['multipro']['hop_success_rate'].split('/')[1])
    
    # Save to JSON file
    json_path = scores_dir / "results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
            
    logger.info(f"Results saved to {json_path}")

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
    
    # Prepare test file list
    test_file_list = []
    assert args.domain in ['full', 'shopping', 'wikipedia', 'normal', 'multi567', 'compare', 'multipro'], f"Invalid domain {args.domain}"
    task_num = MMINA_DICT[args.domain]
    
    if args.domain == 'full':
        for domain in ['shopping', 'wikipedia', 'normal', 'multi567', 'compare', 'multipro']:
            sub_task_num = MMINA_DICT[domain]
            for i in range(sub_task_num):
                test_file_list.append(f"mmina/{domain}/{i+1}.json")
    else:
        test_file_list = [f"mmina/{args.domain}/{i+1}.json" for i in range(task_num)]
    
    # test_file_list = get_unfinished(test_file_list, args.result_dir)
    logger.info(f"Total {len(test_file_list)} tasks left")
    
    if args.hist:
        logger.info(f"Initial history info: Use history: {args.hist} | History number: {args.hist_num}")
    args.render = True
    args.render_screenshot = True
    args.save_trace_enabled = True

    if args.provider == "custom":
        # add your customized model here
        if args.model == "your_customized_model":
            # add specific initialization here
            pass
        elif args.model == "cogagent-9b":
            args.loaded_model = AutoModelForCausalLM.from_pretrained(
                        os.environ["HF_MODEL_ID"],
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True,
                        device_map="auto",
                        # quantization_config=BitsAndBytesConfig(load_in_8bit=True), # For INT8 quantization
                        # quantization_config=BitsAndBytesConfig(load_in_4bit=True), # For INT4 quantization
                    ).eval()
            
        else:
            raise NotImplementedError(f"model {args.model} not implemented")

    args.current_viewport_only = True
    dump_config(args)

    agent = construct_agent(args)
    print(test_file_list)
    test(args, agent, test_file_list)
    logger.info(f"Test finished. Log file: {LOG_FILE_NAME}")