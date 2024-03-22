from cosypose.config import LOCAL_DATA_DIR
import i_grip

from pathlib import Path

_COSYPOSE_DATASET_PATH = LOCAL_DATA_DIR / 'bop_datasets'
_TLESS_MESH_PATH = _COSYPOSE_DATASET_PATH / 'tless' / 'models_cad'
_YCVB_MESH_PATH = _COSYPOSE_DATASET_PATH / 'ycbv' / 'models'
_TLESS_URDF_PATH = LOCAL_DATA_DIR / 'urdfs' / 'tless.cad' 
_YCVB_URDF_PATH = LOCAL_DATA_DIR / 'urdfs' / 'ycbv'

_FILES_DIR = Path(i_grip.__path__[0]) 
_FILES_DIR = Path(i_grip.__file__).parent.parent / 'files'
_MEDIAPIPE_MODEL_PATH = _FILES_DIR / 'hand_landmarker.task'

_YCVB_TEST_PICTURES_PATH = _FILES_DIR / 'YCBV_test_pictures'
_DEFAULT_YCBV_TEST_PICTURES = [str(_YCVB_TEST_PICTURES_PATH / 'bleach.png'), str(_YCVB_TEST_PICTURES_PATH / 'mustard_front.png')]

TLESS_DETECTOR_ID = 'detector-bop-ycbv-synt+real--292971'
TLESS_COARSE_ESTIMATOR_ID = 'coarse-bop-tless-synt+real--160982'
TLESS_REFINER_ESTIMATOR_ID = 'refiner-bop-tless-synt+real--881314'

YCVB_DETECTOR_ID =  'detector-bop-tless-synt+real--452847'
YCVB_COARSE_ESTIMATOR_ID = 'coarse-bop-ycbv-synt+real--822463'
YCVB_REFINER_ESTIMATOR_ID = 'refiner-bop-ycbv-synt+real--631598'