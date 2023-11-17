# !/usr/bin/python3
# --coding:utf-8--
# @File: clean_util.py
# @Author:junru jin
# @Time: 2023年11月17日13
# @description: A code to clean version dirs in pretrain_model

import os
import shutil


def clean(path):
    root_path = path
    empty_dirs = []
    dir_names = os.listdir(root_path)
    for dir_name in dir_names:
        cur_path = os.path.join(root_path, dir_name)
        if os.path.isdir(cur_path):
            cur_versions = os.listdir(cur_path)
            for cur_version in cur_versions:
                file_names = os.listdir(os.path.join(cur_path, cur_version))
                if "checkpoints" not in file_names:
                    empty_dirs.append(os.path.join(cur_path, cur_version))

    print(empty_dirs)
    # if input("Delete these dirs? (y/n)") == 'y':
    #     for deleted_dir in empty_dirs:
    #         shutil.rmtree(deleted_dir)
    #     print("Deleted!")
    # else:
    #     print("Not deleted!")


def clean_pretrain(path):
    root_path = path
    empty_dirs = []
    dir_names = os.listdir(root_path)
    for dir_name in dir_names:
        # pubchem_250_data dir_name
        cur_path = os.path.join(root_path, dir_name)
        if os.path.isdir(cur_path):
            file_names = os.listdir(cur_path)
            print(file_names)
            if "checkpoint" not in file_names:
                empty_dirs.append(cur_path)

    print(empty_dirs)
    # if input("Delete these dirs? (y/n)") == 'y':
    #     for deleted_dir in empty_dirs:
    #         shutil.rmtree(deleted_dir)
    #     print("Deleted!")
    # else:
    #     print("Not deleted!")


if __name__ == '__main__':
    pretrain_task = ['cl', 'fg']
    pretrain_mode = ['cl', 'ad']
    for i in pretrain_task:
        for j in pretrain_mode:
            path = f'../pretrain_model/{i}/{j}'
            # 判断是否存在该路径
            if os.path.exists(path):
                clean_pretrain(path)