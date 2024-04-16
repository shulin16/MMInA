import json  # Function to recursively build the HTML from the DOM tree nodes


def build_html(node, strings):
    tag = strings[node["nodeName"]]
    attributes = ""

    if "attributes" in node:
        # Attributes are stored in pairs; [name, value, name, value, ...]
        attributes = "".join(
            f' {strings[node["attributes"][i]]}="{strings[node["attributes"][i+1]]}"'
            for i in range(0, len(node["attributes"]), 2)
        )
        children = ""
        if "childNodeIndexes" in node:
            children = "".join(
                build_html(nodes[child_index], strings)
                for child_index in node["childNodeIndexes"]
            )

        if tag == "#text":

            return strings[node["nodeValue"]]

        elif tag == "#comment":

            return f'<!-- {strings[node["nodeValue"]]} -->'

        else:

            return f"<{tag}`{attributes}>{children}</{tag}>"  # Load the DOM tree from the provided JSON example


with open("../tree.json", "r") as f:
    dom_tree_json = f.read()
dom_tree = json.loads(
    dom_tree_json
)  # Assuming the first document is the one we want to convert

document = dom_tree["documents"][0]

nodes = document["nodes"]
print(nodes.keys())  # Extract the nodes and strings from the document
print(nodes["isClickable"])
clickable_nodes = nodes["isClickable"]


strings = dom_tree[
    "strings"
]  # Assuming the root node is the first node in the nodes list

# root_node = nodes[0] # Build the cleaned HTML

cleaned_html = build_html(root_node, strings)  # Output the cleaned HTML

print(cleaned_html)
