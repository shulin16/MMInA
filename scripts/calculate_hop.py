import json

# Function to read lines containing the pattern from a .txt file
def read_lines_with_pattern(file_path, pattern):
    try:
        matching_lines = []
        with open(file_path, 'r') as file:
            for line in file:
                if pattern in line:
                    matching_lines.append(line.strip())
        return matching_lines
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage:
file_path = '/home/data2/stian/webarena/output_gpv_mh.txt'  # Replace with the path to your .txt file
pattern = "INFO - [Result] (PASS) "

matching_lines = read_lines_with_pattern(file_path, pattern)

pass_list = []
if matching_lines:
    print(f"Lines containing the pattern '{pattern}' in '{file_path}':")
    for line in matching_lines:
        # print(line)
        path = line.split(pattern)[1]
        pass_list.append(path)

for config_file in pass_list:
    # get intent
    with open(config_file) as f:
        _c = json.load(f)
        hop_cnt = _c["hop_cnt"]
        # cnt_hop = _c["cnt_hop"]
        print(hop_cnt)
        

