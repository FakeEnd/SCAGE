import os
import shutil

from tqdm import tqdm

def add_new_data(type):
    folder_path = '/mnt/8t/qjb/workspace/SCAGE_DATA/pretrain_data/pubchem_250_new/processed'

    target_names = os.listdir(folder_path)

    max = 0  # 用于计数符合条件的文件数量

    for file_name in tqdm(target_names):
        if file_name.endswith('.pth') and file_name != 'pre_transform.pth' and file_name != 'pre_filter.pth':
            file_name, file_extension = os.path.splitext(file_name)
            index = int(file_name.split('_')[1])
            max = index if index > max else max
    print(max)
    count = max + 1
    # oriange=2574334  finetune=163047  cliff=48707

    # oriange=2574334  finetune=163045  cliff=48707

    source_path = f'/mnt/8t/qjb/workspace/SCAGE_DATA/{type}_data_sum/'
    file_names = os.listdir(source_path)
    for file_name in tqdm(file_names):
        if file_name.endswith('.pth') and file_name != 'pre_transform.pth' and file_name != 'pre_filter.pth':
            source_file_path = os.path.join(source_path, file_name)
            destination_file_name = f'data_{count}.pth'
            destination_file_path = os.path.join(folder_path, destination_file_name)

            # 复制文件
            shutil.copy(source_file_path, destination_file_path)
            count += 1




def cp_files(source_path, target_path):
    file_names = os.listdir(source_path)
    task_name_list = []
    for file_name in file_names:
        task_name_list.append(file_name)
    print(task_name_list)
    # task_name_list = ['CHEMBL204_Ki']
    index = 0
    for task_name in task_name_list:
        print(f'now processing {task_name}')
        source_folder = os.path.join(source_path, task_name)

        file_names = os.listdir(source_folder)
        for file_name in tqdm(file_names):
            if file_name.endswith('.pth') and file_name != 'pre_transform.pth' and file_name != 'pre_filter.pth':
                source_file_path = os.path.join(source_folder, file_name)
                destination_file_name = f'data_{index}.pth'
                destination_file_path = os.path.join(target_path, destination_file_name)

                # 复制文件
                shutil.copy(source_file_path, destination_file_path)
                index += 1

def cp_(source_path, target_path):
    file_names = os.listdir(source_path)
    for file_name in tqdm(file_names):
        if file_name.endswith('.pth') and file_name != 'pre_transform.pth' and file_name != 'pre_filter.pth':
            source_file_path = os.path.join(source_path, file_name)
            destination_file_path = os.path.join(target_path, file_name)
            shutil.copy(source_file_path, destination_file_path)

def re_label():
    source_path = '/mnt/8t/qjb/workspace/SCAGE_DATA/finetune_data_sum_cp/'
    target_path = '/mnt/8t/qjb/workspace/SCAGE_DATA/finetune_data_sum/'

    index = 0
    file_names = os.listdir(source_path)
    for file_name in tqdm(file_names):
        if file_name.endswith('.pth') and file_name != 'pre_transform.pth' and file_name != 'pre_filter.pth':
            source_file_path = os.path.join(source_path, file_name)
            destination_file_name = f'data_{index}.pth'
            destination_file_path = os.path.join(target_path, destination_file_name)

            # 复制文件
            shutil.copy(source_file_path, destination_file_path)
            index += 1

def stats_files():
    path = '/root/autodl-tmp/SCAGE2.0/pretrain_data/pubchem_250_trick/processed/'
    file_names = os.listdir(path)
    # index = 2574334
    # nums = len(file_names) - index
    # for _ in tqdm(range(nums)):
    #     file_name = f'data_{index}.pth'
    #     file_path = os.path.join(path, file_name)
    #     os.remove(file_path)
    #     index += 1
    max = 0  # 用于计数符合条件的文件数量

    for file_name in tqdm(file_names):
        if file_name.endswith('.pth') and file_name != 'pre_transform.pth' and file_name != 'pre_filter.pth':
            file_name, file_extension = os.path.splitext(file_name)
            index = int(file_name.split('_')[1])
            max = index if index > max else max
    print(max)

def rename():
    path = '/root/autodl-tmp/data/cliff_data_sum/'
    file_names = os.listdir(path)
    for file_name in tqdm(file_names):
        if file_name.endswith('.pth') and file_name != 'pre_transform.pth' and file_name != 'pre_filter.pth':
            file_name_a, file_extension = os.path.splitext(file_name)
            index = int(file_name_a.split('_')[1])
            os.rename(os.path.join(path, file_name), os.path.join(path, f'data_{index+2737379}.pth'))

# def cp_files():
#     source_path = '/root/autodl-tmp/data/cliff_data_sum/'
#     target_path = '/root/autodl-tmp/SCAGE2.0/pretrain_data/pubchem_250_trick/processed/'
#     file_names = os.listdir(source_path)
#     for file_name in tqdm(file_names):
#         if file_name.endswith('.pth') and file_name != 'pre_transform.pth' and file_name != 'pre_filter.pth':
#             source_file_path = os.path.join(source_path, file_name)
#             destination_file_path = os.path.join(target_path, file_name)
#             # 复制文件
#             shutil.copy(source_file_path, destination_file_path)






if __name__ == '__main__':
    source_path = '/mnt/solid/tss/Project-2023/SCAGE2.0/data/pretrain_data/pubchem_1000/processed'
    target_path = '/mnt/sde/qjb/SCAGE/data/pubchem_1000_new/processed'
    # source_path = '/mnt/8t/qjb/workspace/SCAGE_DATA/cliff_data_split'
    # target_path = '/mnt/8t/qjb/workspace/SCAGE_DATA/cliff_data_split_sum/'
    # cp_files(source_path, target_path)

    # add_new_data('cliff')
    # cp_(source_path, target_path)

    stats_files()

    # import os
    #
    # directory_path = '/mnt/8t/qjb/workspace/SCAGE_DATA/cliff_data_split_cp'  # 将'/your/directory/path'替换为实际目录路径
    #
    # # 获取目录中的所有文件
    # file_names = os.listdir(directory_path)
    #
    # # 使用列表推导式来筛选以'.pth'结尾的文件
    # pth_files = [file_name for file_name in file_names if file_name.endswith('.pth')]
    #
    # # 遍历并删除这些文件
    # for pth_file in pth_files:
    #     file_path = os.path.join(directory_path, pth_file)
    #     os.remove(file_path)
    #
    # print(f"已删除 {len(pth_files)} 个以'.pth'结尾的文件")