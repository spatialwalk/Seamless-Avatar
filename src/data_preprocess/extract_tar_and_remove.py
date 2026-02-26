import os
import subprocess
from icecream import ic
import tqdm
from tqdm.contrib.concurrent import process_map


def untar_file(tar_file_path):
    command = ['tar', '-xvf', os.path.basename(tar_file_path)]
    subprocess.run(command, cwd=os.path.dirname(tar_file_path), check=True)
    return


root_dir = 'datasets/seamless_smplx_dataset'

sub_folder_list = [
    'asr_annos',
    'audios_loudnorm_16k',
    'smplx_npz_annos',
]


tar_file_list = []
for sub_folder in sub_folder_list:
    sub_folder_path = os.path.join(root_dir, sub_folder)
    for tar_file in os.listdir(sub_folder_path):
        if tar_file.endswith('.tar'):
            tar_file_list.append(os.path.join(sub_folder_path, tar_file))

assert len(tar_file_list) == 27

process_map(untar_file, tar_file_list, desc='Extracting tar files')

for tar_file in tqdm.tqdm(tar_file_list, desc='Removing tar files'):
    os.remove(tar_file)


