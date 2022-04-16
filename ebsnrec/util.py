import hashlib
import json
import logging
import os
import time
from hashlib import md5
import yaml


def check_and_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def set_log(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', log_data_dir=None):
    if log_data_dir:
        log_file = os.path.join(log_data_dir, 'info.log')
        logging.basicConfig(level=level,
                            format=format,
                            handlers=[logging.FileHandler(log_file, mode='w'),
                                      logging.StreamHandler()])
    else:
        logging.basicConfig(level=level,
                            format=format,
                            handlers=[logging.StreamHandler()])


def parse_spec_cols(spec_cols):
    idx = 0
    for k in ["label_col", "group_col", "sequence_col"]:
        if k in spec_cols:
            spec_cols[k]["idx"] = idx
            idx += 1
    if idx != len(spec_cols):
        raise RuntimeError


def get_dict_from_config_path(config_path:str)->dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"{config_path}")
    with open(config_path, "r") as fp:
        return yaml.full_load(fp)


def update_dict_by_dict_recursively(target_dict:dict, source_dict:dict)->None:
    for k in source_dict:
        if k in target_dict:
            if isinstance(target_dict[k], dict):
                update_dict_by_dict_recursively(target_dict[k], source_dict[k])
            else:
                target_dict[k] = source_dict[k]
        else:
            target_dict[k] = source_dict[k]


def create_signature(*args)->str:
    x = '.'.join((json.dumps(arg) for arg in args))
    md = hashlib.md5()
    md.update(x.encode('utf-8'))
    return md.hexdigest()


def save_signature(signature,dir_name):
    check_and_mkdir(dir_name)
    file = os.path.join(dir_name, "signature.txt")
    with open(file, 'w', encoding='utf8') as fp:
        fp.write(signature)


def get_signature(dir_name):
    file = os.path.join(dir_name, "signature.txt")
    if not os.path.exists(file):
        return None
    with open(file,'r',encoding='utf-8') as fp:
        return fp.read()



