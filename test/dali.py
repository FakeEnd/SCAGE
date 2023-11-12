from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import os
images_dir = '/mnt/solid/tss/Project-2023/SCAGE2.0/data/pretrain_data/pubchem_250_new'
@pipeline_def(num_threads=4, device_id=0)
def get_dali_pipeline():
    images, _ = fn.readers.file(file_root=images_dir, random_shuffle=True, name="Reader")
    # images, labels = fn.readers.file(
    #     file_root=images_dir, random_shuffle=True, name="Reader")
    # # decode data on the GPU
    # images = fn.decoders.image_random_crop(
    #     images, device="mixed", output_type=types.RGB)
    return images


train_data = DALIGenericIterator(
    [get_dali_pipeline(batch_size=16)],
    ['data', 'label'],
    reader_name='Reader'
)


for i, data in enumerate(train_data):
    x, y = data[0]['data'], data[0]['label']
    print(x)