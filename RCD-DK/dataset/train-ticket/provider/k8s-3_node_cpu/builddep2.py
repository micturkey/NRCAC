import pandas as pd
import ast

# 读取节点依赖关系的TXT文件
with open('dependency.txt', 'r') as file:
    nodes_data = file.read()
    nodes_list = ast.literal_eval(nodes_data)

# 将节点依赖关系转换为DataFrame
nodes_df = pd.DataFrame(nodes_list, columns=['source', 'target'])

# 读取列名的CSV文件
columns_df = pd.read_csv('normal.csv')

# 获取列名
columns = columns_df.columns

# 提取节点名并创建一个字典以加快匹配速度
node_names = set()
node_to_columns = {}
for col in columns:
    node_name = col.split('_')[0]
    node_names.add(node_name)
    if node_name not in node_to_columns:
        node_to_columns[node_name] = []
    node_to_columns[node_name].append(col)

# 生成列名依赖关系
column_dependencies = []

for index, row in nodes_df.iterrows():
    source_node = row['source']
    target_node = row['target']
    
    # 查找与节点相关的列名
    source_columns = node_to_columns.get(source_node, [])
    target_columns = node_to_columns.get(target_node, [])
    
    # 创建列名之间的依赖关系
    for source_col in source_columns:
        for target_col in target_columns:
            column_dependencies.append([source_col, target_col])

# 将结果转换为TXT格式字符串
column_dependencies_str = str(column_dependencies)

# 保存为TXT文件
with open('dependenciy_list.txt', 'w') as file:
    file.write(column_dependencies_str)

print("列名之间的依赖关系已保存到 dependenciy_list.txt 文件中。")
