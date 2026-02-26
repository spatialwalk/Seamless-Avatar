import os

N_MOTIONS_FOR_DIT = 240  # 8s
FPS = 30
SAMPLE_RATE = 16000


# paths configuration
DATASET_ROOT_DIR = 'datasets/seamless_smplx_dataset'
SMPLX_PARAMS_STATS_SAVE_PATH = os.path.join(
    DATASET_ROOT_DIR, 'smplx_params_stats.npz')
SMPLX_NPZ_FOLDER = os.path.join(DATASET_ROOT_DIR, 'smplx_npz_annos')
AUDIOS_FOLDER = os.path.join(DATASET_ROOT_DIR, 'audios_loudnorm_16k')


N_FRAMES_DELTA_THRESHOLD = 2

TMP_DIR = 'outputs/tmp'
OUTPUT_ROOT_DIR = 'outputs'
