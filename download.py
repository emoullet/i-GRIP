import argparse
import os
import zipfile
import wget
import logging
import shutil
from pathlib import Path
import subprocess
from cosypose.config import PROJECT_DIR, LOCAL_DATA_DIR, BOP_DS_DIR
from cosypose.utils.logging import get_logger
from i_grip.config import _COSYPOSE_DATASET_PATH, _FILES_DIR, _MEDIAPIPE_MODEL_PATH, TLESS_COARSE_ESTIMATOR_ID, TLESS_REFINER_ESTIMATOR_ID, YCVB_COARSE_ESTIMATOR_ID, YCVB_REFINER_ESTIMATOR_ID, TLESS_DETECTOR_ID, YCVB_DETECTOR_ID
logger = get_logger(__name__)


RCLONE_CFG_PATH = (PROJECT_DIR / 'rclone.conf')
RCLONE_ROOT = 'cosypose:'

#get default download directory
DEFAULT_DOWNLOAD_DIR = Path(os.getenv('HOME')) / 'Downloads'
DOWNLOAD_DIR = Path('/home/emoullet/Téléchargements') / 'temp'
DOWNLOAD_DIR = _COSYPOSE_DATASET_PATH
DOWNLOAD_DIR.mkdir(exist_ok=True)
EXTRACT_DIR = Path('/home/emoullet/Téléchargements') / 'temp' / 'extracted'
EXTRACT_DIR = _COSYPOSE_DATASET_PATH    

BOP_SRC = 'https://bop.felk.cvut.cz/media/data/bop_datasets/'
BOP_DESTINATION = _COSYPOSE_DATASET_PATH

MEDIAPIPE_SRC = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
MEDIAPIPE_DESTINATION = _MEDIAPIPE_MODEL_PATH

BOP_DS_NAMES = ('tless','ycbv', 'both')


def main():
    parser = argparse.ArgumentParser('CosyPose download utility')
    parser.add_argument('--bop_dataset', default='both', type=str, choices=BOP_DS_NAMES)
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
        
    if args.bop_dataset == 'both':
        download_bop_original('tless')
        download_bop_original('ycbv')
        gdrive_download(f'urdfs/tless.cad', LOCAL_DATA_DIR / 'urdfs')
        gdrive_download(f'urdfs/ycbv', LOCAL_DATA_DIR / 'urdfs')
        expe_list = [TLESS_COARSE_ESTIMATOR_ID, TLESS_REFINER_ESTIMATOR_ID, YCVB_COARSE_ESTIMATOR_ID, YCVB_REFINER_ESTIMATOR_ID, TLESS_DETECTOR_ID, YCVB_DETECTOR_ID]
        for expe_id in expe_list:
            gdrive_download(f'experiments/{expe_id}', LOCAL_DATA_DIR / 'experiments')
        
    elif args.bop_dataset == 'tless':
        download_bop_original('tless')
        gdrive_download(f'urdfs/tless.cad', LOCAL_DATA_DIR / 'urdfs')
        expe_list = [TLESS_COARSE_ESTIMATOR_ID, TLESS_REFINER_ESTIMATOR_ID, TLESS_DETECTOR_ID]
        for expe_id in expe_list:
            gdrive_download(f'experiments/{expe_id}', LOCAL_DATA_DIR / 'experiments')
    elif args.bop_dataset == 'ycbv':
        download_bop_original('ycbv')
        gdrive_download(f'urdfs/ycbv', LOCAL_DATA_DIR / 'urdfs')
        expe_list = [YCVB_COARSE_ESTIMATOR_ID, YCVB_REFINER_ESTIMATOR_ID, YCVB_DETECTOR_ID]
        for expe_id in expe_list:
            gdrive_download(f'experiments/{expe_id}', LOCAL_DATA_DIR / 'experiments')
        
    wget_download(MEDIAPIPE_SRC, MEDIAPIPE_DESTINATION)
        

def download_bop_original(ds_name):
    filename = f'{ds_name}_base.zip'
    wget_download_and_extract(BOP_SRC + filename, BOP_DESTINATION)
    wget_download_and_extract(BOP_SRC + f'{ds_name}_models.zip', BOP_DESTINATION / ds_name)
    useless_suffixes = ['eval', 'fine', 'reconst']
    for suffix in useless_suffixes:
        try:
            print(f'rm {BOP_DESTINATION / ds_name / f"models_{suffix}"}')
            shutil.rmtree(str(BOP_DESTINATION / ds_name / f'models_{suffix}'))
        except FileNotFoundError:
            pass

def wget_download_and_extract(url,  out):
    tmp_path = DOWNLOAD_DIR / url.split('/')[-1]
    print(f'url: {url}')    
    if tmp_path.exists():
        logger.info(f'{url} already downloaded: {tmp_path}...')
    else:
        logger.info(f'Download {url} at {tmp_path}...')
        wget.download(url, out=tmp_path.as_posix())
    logger.info(f'Extracting {tmp_path} at {out}.')
    zipfile.ZipFile(tmp_path).extractall(out)
    #delete the zip file
    # tmp_path.unlink()
    
def wget_download(url, out): 
    if out.exists():
        logger.info(f'{url} already downloaded: {out}...')
    else:
        logger.info(f'Download {url} at {out}...')
        wget.download(url, out=out.as_posix())


def run_rclone(cmd, args, flags):
    rclone_cmd = ['rclone', cmd] + args + flags + ['--config', str(RCLONE_CFG_PATH)]
    logger.debug(' '.join(rclone_cmd))
    subprocess.run(rclone_cmd)


def gdrive_download(gdrive_path, local_path):
    gdrive_path = Path(gdrive_path)
    if gdrive_path.name != local_path.name:
        local_path = local_path / gdrive_path.name
    rclone_path = RCLONE_ROOT+str(gdrive_path)
    local_path = str(local_path)
    logger.info(f'Copying {rclone_path} to {local_path}')
    run_rclone('copyto', [rclone_path, local_path], flags=['-P'])

if __name__ == '__main__':
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    main()
