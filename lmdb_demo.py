import lmdb
import os, sys
import torch
import io
from tqdm import tqdm
import pickle

def insert_pyg(data_path, db_path):
    env = lmdb.open(db_path, map_size=1024*1024*1024*1024)
    txn = env.begin(write=True)
    keys = []
    for file_name in tqdm(os.listdir(data_path)):
            if file_name.endswith('.pth') and file_name != 'pre_transform.pth' and file_name != 'pre_filter.pth':
                #  print(file_name)
                fname, fext = os.path.splitext(file_name)
                idx = fname.split('_')[1]
                # print(idx)
                # data = torch.load(os.path.join(data_path, 'data_{}.pth'.format(idx)))
                # print(data)
                file = open(os.path.join(data_path, 'data_{}.pth'.format(idx)), 'rb')
                # buffer = io.BytesIO(file)
                # data = torch.load(file)
                # print(data)
                keys.append(str(idx).encode())
                txn.put(str(idx).encode(), file.read())
    txn.commit()
    # print(keys)
    with env.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys))
        txn.put(b'__len__', str(len(keys)).encode())


def read_pyg(db_path, idx):
    env = lmdb.open(db_path)
    txn = env.begin()
    # print(txn.get(str(idx).encode()))

    # env.close()
    return txn.get(str(idx).encode())

def update_pyg(env, idx, data):
    txn = env.begin(write=True)
    txn.put(str(idx).encode(), data.read())


def get_len(db_path):
    env = lmdb.open(db_path)
    txn = env.begin()
    return txn.get(b'__len__')

def get_keys(db_path):
    env = lmdb.open(db_path)
    txn = env.begin()
    return txn.get(b'__keys__')


if __name__ == '__main__':
    data_path = '/mnt/8t/qjb/workspace/SCAGE_DATA/pretrain_data/pubchem_demo/processed'
    db_path = '/mnt/8t/qjb/workspace/SCAGE_DATA/lmdb_dir/ttt'
    # length = get_len(db_path)
    # print(length.decode())
    # keys = get_keys(db_path)
    # print(keys)
    # print(pickle.loads(keys))
    insert_pyg(data_path, db_path)
    # print(torch.load(os.path.join(data_path, 'data_{}.pth'.format(1))))
    # data = read_pyg(db_path, 0)
    # buffer = io.BytesIO(data)
    # data = torch.load(buffer)
    # print(data)

# def initialize():
#     env = lmdb.open("./lmdb_dir/pubchem_demo5")
#     return env

# def insert(env, sid, name):
#     txn = env.begin(write=True)
#     txn.put(str(sid).encode(), name.encode())
#     txn.commit()

# def delete(env, sid):
#     txn = env.begin(write=True)
#     txn.delete(str(sid).encode())
#     txn.commit()

# def update(env, sid, name):
#     txn = env.begin(write=True)
#     txn.put(str(sid).encode(), name.encode())
#     txn.commit()

# def search(env, sid):
#     txn = env.begin()
#     name = txn.get(str(sid).encode())
#     return name

# def display(env):
#     txn = env.begin()
#     cur = txn.cursor()
#     for key, value in cur:
#         print(key, value)


# env = initialize()

# print("Insert 3 records.")
# insert(env, 1, "Alice")
# insert(env, 2, "Bob")
# insert(env, 3, "Peter")
# display(env)

# print("Delete the record where sid = 1.")
# delete(env, 1)
# display(env)

# print("Update the record where sid = 3.")
# update(env, 3, "Mark")
# display(env)

# print("Get the name of student whose sid = 3.")
# name = search(env, 3)
# print(name)

# 最后需要关闭关闭lmdb数据库
# env.close()

# 执行系统命令
# os.system("rm -r lmdb_dir")