import json
import re
import os
import traceback
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from typing import Any, TypedDict, Union

import cairosvg
import numpy as np
import numpy.typing as npt
import requests
from beartype import beartype
from bs4 import BeautifulSoup
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import CDPSession, Page, ViewportSize

from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

from browser_env.constants import (
    ASCII_CHARSET,
    FREQ_UNICODE_CHARSET,
    IGNORED_ACTREE_PROPERTIES,
    UTTERANCE_MAX_LENGTH,
)

from .utils import (
    AccessibilityTree,
    BrowserConfig,
    BrowserInfo,
    Observation,
    png_bytes_to_numpy,
)

# from aibrowser.combine_img import MergeImage


class ObservationProcessor:
    def process(self, page: Page, client: CDPSession) -> Observation:
        raise NotImplementedError


class ObservationMetadata(TypedDict):
    obs_nodes_info: dict[str, Any]


def create_empty_metadata() -> ObservationMetadata:
    return {
        "obs_nodes_info": {},
    }


def image_unifier(img_url: str, idx: int):
    if img_url[-3:] == "svg":
        try:
            # Fetch the SVG data from the URL
            response = requests.get(img_url)
            # Convert the SVG data to a PNG image using cairosvg
            png_img = cairosvg.svg2png(bytestring=response.content)  # png_data
            # with open(png_file_path, "wb") as f:
            #     f.write(png_data)

            # Save the PNG image to a file
            png_img = Image.open(BytesIO(png_img))
            png_file_path = (
                "/home/data2/stian/MMInA/sampling_data/imgs_caption/" + str(idx) + ".png"
            )
            png_img.save(png_file_path)
            img_url = png_file_path
            print(f"Saved the PNG image to: {png_file_path}")

        except:
            traceback.print_exc()
            img_url = ""
    return img_url

def merge_img(img_urls: list, text: list, save_pth: str):
    images = []
    
    for img_url in img_urls:
        response = requests.get(img_url)
        if img_url[-3:] == "svg":
            png_img = cairosvg.svg2png(bytestring=response.content)
            image = Image.open(BytesIO(png_img))
        else:
            image = Image.open(BytesIO(response.content))
        images.append(image)
    print(images)
    font_size = 120
    font = ImageFont.truetype("/home/data2/stian/MMInA/aibrowser/Times New Roman.ttf", font_size)
    # Assuming `images` is your list of Image objects
    text_heights = [int(ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(txt, font=font)) for txt in text]
    max_text_height = max(text_heights)
    widths, heights = zip(*(i.size for i in images))
    max_height = max(heights) + max_text_height
    total_widths = sum(widths)

    new_image = Image.new("RGB", (total_widths, max_height), "white")


    x_offset = 0
    text_color = (0, 0, 0)

    num = 1

    for image, txt in zip(images, text):
        draw = ImageDraw.Draw(new_image)
        text_size = draw.textlength(txt, font=font)
        
        new_image.paste(image, (x_offset, 0))
        
        x = x_offset + (image.width - text_size) // 2
        y = max_height - max_text_height
        draw.text((x, y), "["+str(num)+"]" + txt, fill=text_color, font=font)
        num+=1
        
        x_offset += image.width

    new_image.save(save_pth)
        
class TextObervationProcessor(ObservationProcessor):
    def __init__(
        self,
        observation_type: str,
        current_viewport_only: bool,
        viewport_size: ViewportSize,
        caption: bool,
        caption_name: str,
    ):
        self.observation_type = observation_type
        self.current_viewport_only = current_viewport_only
        self.viewport_size = viewport_size
        self.observation_tag = "text"
        self.meta_data = (
            create_empty_metadata()
        )  # use the store meta data of this observation type
        self.caption = caption
        self.caption_name = caption_name
        
        
        # Captioning
        if self.caption:
            self.blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.bfloat16)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.blip_model.to(self.device)
            print("```````````````This method use Blip to caption, Done Blip Model````````````````")
            # Save caption
            root = "/home/data2/stian/MMInA/sampling_data/caption/"
            self.full_file_path = os.path.join(root, self.caption_name)
            # Check if the file already exists, and if so, delete it
            if os.path.exists(self.full_file_path):
                os.remove(self.full_file_path)
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.full_file_path), exist_ok=True)
            print(f"Create full path {self.full_file_path}")
    @beartype
    def fetch_browser_info(
        self,
        page: Page,
        client: CDPSession,
    ) -> BrowserInfo:
        # extract domtree
        tree = client.send(
            "DOMSnapshot.captureSnapshot",
            {
                "computedStyles": [],
                "includeDOMRects": True,
                "includePaintOrder": True,
            },
        )

        print("*********************************************")
        tree_json = json.dumps(tree, indent=2)
        with open("tree.json", "w") as file:
            file.write(tree_json)
        print("Done tree fetch")
        print("*********************************************")

        # calibrate the bounds, in some cases, the bounds are scaled somehow
        bounds = tree["documents"][0]["layout"]["bounds"]
        b = bounds[0]
        n = b[2] / self.viewport_size["width"]
        bounds = [[x / n for x in bound] for bound in bounds]
        tree["documents"][0]["layout"]["bounds"] = bounds
        # add union bound placeholder
        tree["documents"][0]["layout"]["unionBounds"] = [None for _ in bounds]

        # extract browser info
        win_upper_bound = page.evaluate("window.pageYOffset")
        win_left_bound = page.evaluate("window.pageXOffset")
        win_width = page.evaluate("window.screen.width")
        win_height = page.evaluate("window.screen.height")
        win_right_bound = win_left_bound + win_width
        win_lower_bound = win_upper_bound + win_height
        device_pixel_ratio = page.evaluate("window.devicePixelRatio")
        assert device_pixel_ratio == 1.0, "devicePixelRatio is not 1.0"

        config: BrowserConfig = {
            "win_upper_bound": win_upper_bound,
            "win_left_bound": win_left_bound,
            "win_width": win_width,
            "win_height": win_height,
            "win_right_bound": win_right_bound,
            "win_lower_bound": win_lower_bound,
            "device_pixel_ratio": device_pixel_ratio,
        }

        # assert len(tree['documents']) == 1, "More than one document in the DOM tree"
        info: BrowserInfo = {"DOMTree": tree, "config": config}

        print("***************************************")
        html_content = page.content()
        # Save the HTML content to a file
        with open("page.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        print("Done page fetch")
        print("***************************************")
        return info

    @beartype
    @staticmethod
    def partially_in_viewport(
        bound: list[float], config: BrowserConfig
    ) -> bool:
        [x, y, width, height] = bound
        elem_left_bound = x
        elem_top_bound = y
        elem_right_bound = x + width
        elem_lower_bound = y + height

        ok = (
            elem_left_bound < config["win_right_bound"]
            and elem_right_bound >= config["win_left_bound"]
            and elem_top_bound < config["win_lower_bound"]
            and elem_lower_bound >= config["win_upper_bound"]
        )

        return ok

    @beartype
    def retrieve_viewport_info(self, info: BrowserInfo) -> None:
        """Add viewport related information to the DOMTree
        1. add union bound, which is a union of all the bounds of the nodes in the subtree
        This is only used when current_viewport_only is enabled since it is quite slow

        TODO[robert1003]: improve
        """
        tree = info["DOMTree"]
        document = tree["documents"][0]
        nodes = document["nodes"]
        parent = nodes["parentIndex"]
        node_names = nodes["nodeName"]

        layout = document["layout"]
        layout_node_cursor = layout["nodeIndex"]
        bounds = layout["bounds"]

        graph = defaultdict(lambda: [])
        assert len(node_names) == len(parent)
        for node_idx in range(len(node_names)):
            parent_idx = parent[node_idx]
            if parent_idx != -1:
                graph[parent_idx].append(node_idx)

        union_bounds: list[list[float] | None] = [None for _ in bounds]

        def valid_bbox(bound: list[float] | None) -> bool:
            if bound is None:
                return False
            # no width or height
            if np.isclose(bound[2], 0):
                return False
            if np.isclose(bound[3], 0):
                return False
            return True

        def add_union_bound(idx: int) -> list[float] | None:
            if idx in layout_node_cursor:
                cursor = layout_node_cursor.index(idx)
                node_bound = bounds[cursor].copy()
                tree_bounds: list[Any] = [node_bound]
                for child_idx in graph[idx]:
                    child_bound = add_union_bound(child_idx)
                    tree_bounds.append(
                        child_bound.copy() if child_bound else None
                    )

                tree_bounds = [b for b in tree_bounds if valid_bbox(b)]
                # convert to absolute coordinates
                for i in range(len(tree_bounds)):
                    tree_bounds[i][2] = tree_bounds[i][0] + tree_bounds[i][2]
                    tree_bounds[i][3] = tree_bounds[i][1] + tree_bounds[i][3]

                if len(tree_bounds) == 0:
                    assert not valid_bbox(node_bound)
                    node_union_bound = [0.0, 0.0, 0.0, 0.0]
                else:
                    left_bound = min([b[0] for b in tree_bounds])
                    top_bound = min([b[1] for b in tree_bounds])
                    right_bound = max([b[2] for b in tree_bounds])
                    bottom_bound = max([b[3] for b in tree_bounds])
                    node_union_bound = [
                        left_bound,
                        top_bound,
                        right_bound - left_bound,
                        bottom_bound - top_bound,
                    ]

                # update the list
                union_bounds[cursor] = node_union_bound
            else:
                node_union_bound = None

            return node_union_bound

        add_union_bound(0)
        info["DOMTree"]["documents"][0]["layout"]["unionBounds"] = union_bounds

    @beartype
    def current_viewport_html(self, info: BrowserInfo) -> str:
        # adopted from [natbot](https://github.com/nat/natbot)
        tree = info["DOMTree"]
        strings = tree["strings"]
        document = tree["documents"][0]
        nodes = document["nodes"]
        attributes = nodes["attributes"]
        node_value = nodes["nodeValue"]
        parent = nodes["parentIndex"]
        node_names = nodes["nodeName"]

        layout = document["layout"]
        layout_node_cursor = layout["nodeIndex"]
        union_bounds = layout["unionBounds"]

        graph = defaultdict(lambda: [])
        for node_idx in range(len(node_names)):
            parent_idx = parent[node_idx]
            if parent_idx != -1:
                graph[parent_idx].append(node_idx)

        def dfs(idx: int) -> str:
            node_name = strings[node_names[idx]].lower().strip()
            can_skip = "#" in node_name or "::" in node_name

            inner_text = ""
            node_value_idx = node_value[idx]
            if node_value_idx >= 0 and node_value_idx < len(strings):
                inner_text = " ".join(strings[node_value_idx].split())
            node_attributes = [strings[i] for i in attributes[idx]]
            node_attributes_str = ""
            for i in range(0, len(node_attributes), 2):
                a = node_attributes[i]
                b = node_attributes[i + 1]
                b = " ".join(b.split())
                node_attributes_str += f'{a}="{b}" '
            node_attributes_str = node_attributes_str.strip()

            html = ""
            if not can_skip:
                html += f"<{node_name}"
                if {node_attributes_str}:
                    html += f" {node_attributes_str}"
                html += f">{inner_text}"
            else:
                html += f"{inner_text}"

            for child_idx in graph[idx]:
                if child_idx in layout_node_cursor:
                    cursor = layout_node_cursor.index(child_idx)
                    union_bound = union_bounds[cursor]
                    if not self.partially_in_viewport(
                        union_bound, info["config"]
                    ):
                        continue
                    html += dfs(child_idx)

            if not can_skip:
                html += f"</{node_name}>"

            return html

        html = dfs(0)

        return html

    @beartype
    def fetch_page_accessibility_tree(
        self, info: BrowserInfo, client: CDPSession
    ) -> AccessibilityTree:
        accessibility_tree: AccessibilityTree = client.send(
            "Accessibility.getFullAXTree", {}
        )["nodes"]

        print("```````````````````````````````````````````````")
        accessibility_tree_json = json.dumps(accessibility_tree, indent=4)
        with open("accessibility_tree.json", "w", encoding="utf-8") as file:
            file.write(accessibility_tree_json)
        file.close()

        print("```````````````Done accessibility tree````````````````")

        image_data = client.send("Page.getLayoutMetrics")
        images = client.send(
            "Page.captureSnapshot", {"clipRect": image_data["contentSize"]}
        )

        html_content = images.get("data")
        html_content = html_content.replace("=\r\n", "")
        with open("image.html", "w", encoding="utf-8") as html_file:
            html_file.write(html_content)
        
        navigation_history = client.send("Page.getNavigationHistory")
        current_index = navigation_history['currentIndex']
        entries = navigation_history['entries']
        current_url = entries[current_index]['url']

        print("current_url : "+current_url) 
        soup = BeautifulSoup(html_content, "html.parser")

        # print(filtered_img_urls)
        img_tags = soup.find_all('img')

        filtered_img_urls = []
        
        flag = False
        
        for img in img_tags:
        
            
            # TODO: depend on the enviroment
            # suit for shopping
            img_url = img.get('src')
            # filtered_img_urls.append(img_url[3:-1])
            if "localhost:7770" in current_url:
                if "d10f0a40e8034" not in img_url:
                    filtered_img_urls.append(img_url[3:-1])
                    flag = True
            
                # print("img_url::::::::::::: ",img_url)
            
            # suit for wikipedia
            
            if "kiwix" in current_url:
                img_url = img_url.replace('\\', '')
                # print(img_url)
                if "/I/" in img_url:
                    if img_url[3]=='h' and img_url[4]==img_url[5] and img_url[6]=='p':
                        filtered_img_urls.append(img_url[3:-1])
                    else:
                        filtered_img_urls.append(img_url[1:-1])
                    flag = True
        
        print("filtered_img_urls")
        print(filtered_img_urls)
        # a few nodes are repeated in the accessibility tree
        seen_ids = set()
        _accessibility_tree = []
        for node in accessibility_tree:
            if node["nodeId"] not in seen_ids:
                _accessibility_tree.append(node)
                seen_ids.add(node["nodeId"])
        accessibility_tree = _accessibility_tree

        # add the bounding box of each node
        tree = info["DOMTree"]
        document = tree["documents"][0]
        nodes = document["nodes"]
        backend_node_id = nodes["backendNodeId"]
        node_names = nodes["nodeName"]

        layout = document["layout"]
        layout_node_cursor = layout["nodeIndex"]
        bounds = layout["bounds"]
        union_bounds = layout["unionBounds"]
        offsetrect_bounds = layout["offsetRects"]
        backend_id_to_bound = {}

        # get the mapping between backend node id and bounding box
        for idx in range(len(node_names)):
            if idx not in layout_node_cursor:
                continue
            cursor = layout_node_cursor.index(idx)
            node_bound = bounds[cursor]
            node_union_bound = union_bounds[cursor]
            node_offsetrect_bound = offsetrect_bounds[cursor]
            node_backend_id = backend_node_id[idx]
            backend_id_to_bound[node_backend_id] = [
                node_bound,
                node_union_bound,
                node_offsetrect_bound,
            ]

        parent_graph: dict[str, str] = {}
        refine_node_ids: list[str] = []
        for node in accessibility_tree:
            if "parentId" in node:
                parent_graph[node["nodeId"]] = node["parentId"]
            if "backendDOMNodeId" not in node:
                node["bound"] = None
                node["union_bound"] = None
                node["offsetrect_bound"] = None
            elif node["backendDOMNodeId"] not in backend_id_to_bound:
                refine_node_ids.append(node["nodeId"])
            else:
                node["bound"] = backend_id_to_bound[node["backendDOMNodeId"]][
                    0
                ]
                node["union_bound"] = backend_id_to_bound[
                    node["backendDOMNodeId"]
                ][1]
                node["offsetrect_bound"] = backend_id_to_bound[
                    node["backendDOMNodeId"]
                ][2]

        # refine the bounding box for nodes which only appear in the accessibility tree
        node_ids = [node["nodeId"] for node in accessibility_tree]
        for refine_node_id in refine_node_ids:
            child_id = refine_node_id
            parent_idx: None | int = None
            while child_id in parent_graph:
                parent_id = parent_graph[child_id]
                parent_idx = node_ids.index(parent_id)
                child_id = parent_id
                if accessibility_tree[parent_idx]["union_bound"] is not None:
                    break

            refine_node_idx = node_ids.index(refine_node_id)

            if parent_idx is not None:
                accessibility_tree[refine_node_idx][
                    "bound"
                ] = accessibility_tree[parent_idx]["bound"]
                accessibility_tree[refine_node_idx][
                    "union_bound"
                ] = accessibility_tree[parent_idx]["union_bound"]
                accessibility_tree[refine_node_idx][
                    "offsetrect_bound"
                ] = accessibility_tree[parent_idx]["offsetrect_bound"]
            else:
                accessibility_tree[refine_node_idx]["bound"] = None
                accessibility_tree[refine_node_idx]["union_bound"] = None
                accessibility_tree[refine_node_idx]["offsetrect_bound"] = None

        # just a test

        # print("***********************************************")
        global cnt
        cnt = 0
        # for node in accessibility_tree:
        #     if "role" in node and "value" in node["role"] and node["role"]["value"] == "img":
        #         node.setdefault("url", filtered_img_urls[cnt])
        #         node["name"]["value"] = filtered_img_urls[cnt]
        #         cnt += 1
        node_id_to_idx = {}
        for idx, node in enumerate(accessibility_tree):
            node_id_to_idx[node["nodeId"]] = idx

        def dfs(idx: int, obs_node_id: str, depth: int, now: int):
            global cnt
            node = accessibility_tree[idx]
            valid_node = True
            check = False
            if node["role"]["value"] == "img":
                node["name"]["value"] = filtered_img_urls[cnt]
                cnt += 1

            for _, child_node_id in enumerate(node["childIds"]):
                if child_node_id not in node_id_to_idx:
                    continue
                child_depth = depth + 1 if valid_node else depth
                dfs(
                    node_id_to_idx[child_node_id],
                    child_node_id,
                    child_depth,
                    now + 1,
                )

        if len(filtered_img_urls) > 0 and flag:
            dfs(0, accessibility_tree[0]["nodeId"], 0, 0)

        print("count: %d, %d" % (cnt, len(filtered_img_urls)))
        print("**************Done add img***************")

        print("```````````````````````````````````````````````")
        accessibility_tree_clean_json = json.dumps(
            accessibility_tree, indent=4
        )
        with open(
            "accessibility_tree_clean.json", "w", encoding="utf-8"
        ) as file:
            file.write(accessibility_tree_clean_json)
        file.close()

        print("**************Done clean accessibility tree***************")
        # with open("filtered_img_urls.txt", "w") as f:
        #     for url in filtered_img_urls:
        #         f.write(url + "\n")
        # f.close()
        # print("**************Done filtered_img_urls.txt******************")

        return accessibility_tree

    @beartype
    def current_viewport_accessibility_tree(
        self,
        info: BrowserInfo,
        accessibility_tree: AccessibilityTree,
    ) -> AccessibilityTree:
        config = info["config"]
        subtree = []
        for node in accessibility_tree:
            if not node["union_bound"]:
                continue

            [x, y, width, height] = node["union_bound"]
            elem_left_bound = x
            elem_top_bound = y
            elem_right_bound = x + width
            elem_lower_bound = y + height

            ok = (
                elem_left_bound < config["win_right_bound"]
                and elem_right_bound >= config["win_left_bound"]
                and elem_top_bound < config["win_lower_bound"]
                and elem_lower_bound >= config["win_upper_bound"]
            )

            if ok:
                subtree.append(node)

        return subtree

    @beartype
    @staticmethod
    def parse_accessibility_tree(
        accessibility_tree: AccessibilityTree,
    ) -> tuple[str, dict[str, Any]]:
        """Parse the accessibility tree into a string text"""
        node_id_to_idx = {}
        for idx, node in enumerate(accessibility_tree):
            node_id_to_idx[node["nodeId"]] = idx

        obs_nodes_info = {}

        def dfs(idx: int, obs_node_id: str, depth: int) -> str:
            tree_str = ""
            node = accessibility_tree[idx]
            indent = "\t" * depth
            valid_node = True
            try:
                role = node["role"]["value"]
                name = node["name"]["value"]
                node_str = f"[{obs_node_id}] {role} {repr(name)}"
                properties = []
                for property in node.get("properties", []):
                    try:
                        if property["name"] in IGNORED_ACTREE_PROPERTIES:
                            continue
                        if not (name == "img" or role =="img"):
                            properties.append(
                                f'{property["name"]}: {property["value"]["value"]}'
                            )
                    except KeyError:
                        pass

                if properties:
                    node_str += " " + " ".join(properties)

                # check valid
                if not node_str.strip():
                    valid_node = False

                # empty generic node
                if not name.strip():
                    if not properties:
                        if role in [
                            "generic",
                            "img",
                            "list",
                            "strong",
                            "paragraph",
                            "banner",
                            "navigation",
                            "Section",
                            "LabelText",
                            "Legend",
                            "listitem",
                        ]:
                            valid_node = False
                    elif role in ["listitem"]:
                        valid_node = False

                if valid_node:
                    tree_str += f"{indent}{node_str}"
                    obs_nodes_info[obs_node_id] = {
                        "backend_id": node["backendDOMNodeId"],
                        "bound": node["bound"],
                        "union_bound": node["union_bound"],
                        "offsetrect_bound": node["offsetrect_bound"],
                        "text": node_str,
                    }

            except Exception as e:
                valid_node = False

            for _, child_node_id in enumerate(node["childIds"]):
                if child_node_id not in node_id_to_idx:
                    continue
                # mark this to save some tokens
                child_depth = depth + 1 if valid_node else depth
                child_str = dfs(
                    node_id_to_idx[child_node_id], child_node_id, child_depth
                )
                if child_str.strip():
                    if tree_str.strip():
                        tree_str += "\n"
                    tree_str += child_str

            return tree_str

        tree_str = dfs(0, accessibility_tree[0]["nodeId"], 0)
        return tree_str, obs_nodes_info

    @beartype
    @staticmethod
    def clean_accesibility_tree(tree_str: str) -> str:
        """further clean accesibility tree"""
        clean_lines: list[str] = []
        for line in tree_str.split("\n"):
            if "statictext" in line.lower():
                prev_lines = clean_lines[-3:]
                pattern = r"\[\d+\] StaticText '([^']+)'"

                match = re.search(pattern, line)
                if match:
                    static_text = match.group(1)
                    if all(
                        static_text not in prev_line
                        for prev_line in prev_lines
                    ):
                        clean_lines.append(line)
            else:
                clean_lines.append(line)

        return "\n".join(clean_lines)

    @beartype
    def process(self, page: Page, client: CDPSession, cnt1: int, cnt2:int) -> str:
        # get the tab info
        open_tabs = page.context.pages
        try:
            tab_titles = [tab.title() for tab in open_tabs]
            current_tab_idx = open_tabs.index(page)
            for idx in range(len(open_tabs)):
                if idx == current_tab_idx:
                    tab_titles[
                        idx
                    ] = f"Tab {idx} (current): {open_tabs[idx].title()}"
                else:
                    tab_titles[idx] = f"Tab {idx}: {open_tabs[idx].title()}"
            tab_title_str = " | ".join(tab_titles)
        except Exception:
            tab_title_str = " | ".join(
                ["Tab {idx}" for idx in range(len(open_tabs))]
            )

        print("***********************page + client************************")
        print(page)
        print(client)
        print("************************************************************")

        try:
            browser_info = self.fetch_browser_info(page, client)
        except Exception:
            page.wait_for_load_state("load", timeout=500)
            browser_info = self.fetch_browser_info(page, client)

        if self.current_viewport_only:
            self.retrieve_viewport_info(browser_info)

        if self.observation_type == "html":
            if self.current_viewport_only:
                html = self.current_viewport_html(browser_info)
                content = html
            else:
                content = page.content()
        elif self.observation_type == "accessibility_tree":
            accessibility_tree = self.fetch_page_accessibility_tree(
                browser_info, client
            )  # browser_info["DOMTree"]
            if self.current_viewport_only:
                accessibility_tree = self.current_viewport_accessibility_tree(
                    browser_info, accessibility_tree
                )
            content, obs_nodes_info = self.parse_accessibility_tree(
                accessibility_tree
            )
            content = self.clean_accesibility_tree(content)
            self.obs_nodes_info = obs_nodes_info
            self.meta_data["obs_nodes_info"] = obs_nodes_info
        else:
            raise ValueError(
                f"Invalid observatrion type: {self.observation_type}"
            )

        img_urls = []
        text = []
        node_id_to_idx = {}
        for idx, node in enumerate(accessibility_tree):
            node_id_to_idx[node["nodeId"]] = idx

        def dfs(idx: int, obs_node_id: str, depth: int, now: int):
            node = accessibility_tree[idx]
            valid_node = True
            if node["role"]["value"] == "img":
                img_url = node["name"]["value"]
                '''
                # fetch data from http
                if img_url.startswith("http"):
                    response = requests.get(img_url)
                    if img_url[-3:] == "svg":
                        # Fetch the SVG data from the URL
                        # Convert the SVG data to a PNG image using cairosvg
                        png_img = cairosvg.svg2png(bytestring=response.content)  # png_data
                        # Save the PNG image to a file
                        image = Image.open(BytesIO(png_img)).convert('RGB')
                        # png_file_path = ("/home/data2/stian/MMInA/sampling_data/imgs_caption/" + str(idx) + ".png")
                    else:
                        image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
                '''

                # img_url = image_unifier(img_url, idx)
                img_urls.append(img_url)
                text.append(node["nodeId"]+".png")
                
                # Captioning with Blip
                if self.caption:
                    try:
                        image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
                        inputs = self.blip_processor(image, return_tensors="pt").to(self.device, torch.float16)
                        generated_ids = self.blip_model.generate(**inputs, max_new_tokens=20)
                        generated_text = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                        
                        # add caption to the image
                        draw = ImageDraw.Draw(image)
                        # Specify the font, size, and color
                        font = ImageFont.truetype(r"/home/data2/stian/MMInA/browser_env/Arial.ttf", 15)
                        text_color = (237, 230, 211)  # RGB
                        # Calculate the width of the text and the width of the image
                        text_width = draw.textlength(generated_text, font=font)
                        image_width, _ = image.size                
                        text_position = ((image_width - text_width) // 2, image.height)
                        
                        draw.text(text_position, generated_text, font=font, fill=text_color)
                        
                        # Save the image with the caption
                        file_path = "/home/data2/stian/MMInA/sampling_data/imgs_caption/" + str(idx) + ".png"
                        image.save(file_path)
                        print("************Done save captioned image *****************")
                    except:
                        generated_text = "\nCurrent webpage has no image provided, please refer to the webpage content for more information.\n"

                    
                    # Save caption
                    # root = "/home/data2/stian/MMInA/sampling_data/caption/"
                    # file_name = self.caption_name
                    # full_file_path = os.path.join(root, file_name)
                    # # Check if the file already exists, and if so, delete it
                    # if os.path.exists(full_file_path):
                    #     os.remove(full_file_path)
                    # # Ensure the directory exists
                    # os.makedirs(os.path.dirname(full_file_path), exist_ok=True)
                    
                    with open(self.full_file_path, "a") as f:
                        f.write(generated_text + "\n")
            
                    print(f"```````````````Done Captioning: {generated_text}````````````````")
                    

                

            for _, child_node_id in enumerate(node["childIds"]):
                if child_node_id not in node_id_to_idx:
                    continue
                child_depth = depth + 1 if valid_node else depth
                dfs(
                    node_id_to_idx[child_node_id],
                    child_node_id,
                    child_depth,
                    now + 1,
                )

        dfs(0, accessibility_tree[0]["nodeId"], 0, 0)
        # merge_img(img_urls, text, "/home/data2/stian/MMInA/sampling_data/imgs_gemini/output_img.jpg")
        print(img_urls)
        # imgs = MergeImage()
        # imgs.merge(img_urls,text)  #TODO: cannot identify image file
        import shutil
        
        def delete_files_in_folder(folder):
            if not os.path.exists(folder):
                print(f"Folder '{folder}' does not exist.")
                return

            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)

                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')


        def download_images(img_list, folder):
            if not os.path.exists(folder):
                os.makedirs(folder)

            delete_files_in_folder(folder)
            
            for i, img_url in enumerate(img_list):
                try:
                    response = requests.get(img_url)
                    response.raise_for_status()
                    img_name = f"image_{i}.jpg"
                    img_path = os.path.join(folder, img_name)

                    with open(img_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded {img_name}")
                except requests.RequestException as e:
                    print(f"Failed to download {img_url}: {e}")


        folder = "/home/data2/stian/MMInA/imgbin"
        download_images(img_urls, folder)
        self.browser_config = browser_info["config"]
        content = f"{tab_title_str}\n\n{content}"
        return content

    @beartype
    def get_element_center(self, element_id: str) -> tuple[float, float]:
        node_info = self.obs_nodes_info[element_id]
        node_bound = node_info["bound"]
        x, y, width, height = node_bound
        browser_config = self.browser_config
        b_x, b_y = (
            browser_config["win_left_bound"],
            browser_config["win_upper_bound"],
        )
        center_x = (x - b_x) + width / 2
        center_y = (y - b_y) + height / 2
        return (
            center_x / self.viewport_size["width"],
            center_y / self.viewport_size["height"],
        )


class ImageObservationProcessor(ObservationProcessor):
    def __init__(self, observation_type: str):
        self.observation_type = observation_type
        self.observation_tag = "image"
        self.meta_data = create_empty_metadata()

    def process(self, page: Page, client: CDPSession) -> npt.NDArray[np.uint8]:
        try:
            screenshot = png_bytes_to_numpy(page.screenshot())
        except:
            page.wait_for_event("load")
            screenshot = png_bytes_to_numpy(page.screenshot())
        return screenshot


class ObservationHandler:
    """Main entry point to access all observation processor"""

    def __init__(
        self,
        main_observation_type: str,
        text_observation_type: str,
        image_observation_type: str,
        current_viewport_only: bool,
        viewport_size: ViewportSize,
        caption: bool,
        caption_name: str,
    ) -> None:
        self.main_observation_type = main_observation_type
        self.text_processor = TextObervationProcessor(
            text_observation_type, current_viewport_only, viewport_size, caption, caption_name
        )
        self.image_processor = ImageObservationProcessor(
            image_observation_type
        )
        self.viewport_size = viewport_size
        self.caption = caption
        self.caption_name = caption_name

    @beartype
    def get_observation_space(self) -> spaces.Dict:
        text_space = spaces.Text(
            min_length=0,
            max_length=UTTERANCE_MAX_LENGTH,
            charset=ASCII_CHARSET + FREQ_UNICODE_CHARSET,
        )

        image_space = spaces.Box(
            # Each position stores the RGB values. Note the swapped axes (height first).
            np.zeros(
                (self.viewport_size["height"], self.viewport_size["width"], 3),
                dtype=np.uint8,
            ),
            np.ones(
                (self.viewport_size["height"], self.viewport_size["width"], 3),
                dtype=np.uint8,
            )
            * 255.0,
            dtype=np.uint8,
        )

        return spaces.Dict({"text": text_space, "image": image_space})

    @beartype
    def get_observation(
        self, page: Page, client: CDPSession, cnt1: int, cnt2:int
    ) -> dict[str, Observation]:
        text_obs = self.text_processor.process(page, client, cnt1,cnt2)
        image_obs = self.image_processor.process(page, client) # screenshot of the weboage
        return {"text": text_obs, "image": image_obs, "current_url": page.url} 

    @beartype
    def get_observation_metadata(self) -> dict[str, ObservationMetadata]:
        return {
            "text": self.text_processor.meta_data,
            "image": self.image_processor.meta_data,
        }

    @property
    def action_processor(self) -> ObservationProcessor:
        """Return the main processor that is associated with the action space"""
        if self.main_observation_type == "text":
            return self.text_processor
        elif self.main_observation_type == "image":
            return self.image_processor
        else:
            raise ValueError("Invalid main observation type")
