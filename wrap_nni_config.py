import yaml
import argparse
import uuid
from os import system
def format_str_to_list(space):
    if isinstance(space, dict):
        for key,value in space.items():
            if key == '_name':
                continue
            elif isinstance(value, str) or isinstance(value, int) or isinstance(value, float):
                space[key] = [value]
            elif isinstance(value, list):
                space[key] = [format_str_to_list(v) for v in value]
    return space
def format_list_to_choice(space):
    # 首先判断传入的搜索空间是否合法，若不是dict或不包含_name键，则直接返回
    if not isinstance(space, dict) or '_name' not in space:
        return space
    # 若是list类型，则对其进行递归处理，将其中的所有元素都转换为_choice类型
    for key,value in space.items():
        if isinstance(value, list):
            format_value = [format_list_to_choice(v) for v in value]
            space[key] = {'_type': 'choice', '_value': format_value}
    # 对已经转换的部分进行返回
    return space

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    space = cfg['search_space']

    space = format_str_to_list(space) 

    space['_name'] = 'Exp'
    space = format_list_to_choice(space)
    space.pop('_name')

    cfg['search_space'] = space
    filename = '/tmp/nni_exp_config_' + str(uuid.uuid4()) + '.yml'
    yaml.dump(cfg, open(filename, 'w'))
    print(filename)

main()