'''
NeuFace optimization script
'''
import os, sys
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"] = "12"
import numpy as np
import warnings
warnings.simplefilter("ignore", UserWarning)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
np.random.seed(0)


def main(cfg):
   # start optimization
    from neuface.deca import DECA_EFT
    from neuface.pipeline import NeuFacePipeline

    deca = DECA_EFT(cfg)
    eft = NeuFacePipeline(model=deca, config=cfg)

    eft.fit(seq_path=cfg.test_seq_path)
    # eft.recover_and_make_video(seq_path=cfg.test_seq_path, flame_path='/local_data_2/urp25su_jspark/neuface/neuface_noUV_noNormal/MEAD/M013_angry_level_1_001')
    """
    M013_angry_level_1_001
    M019_happy_level_3_003
    M022_sad_level_2_003
    W024_surprised_level_2_003
    """


    # eft.save_vectices_numpy(flame_path='/local_data_2/urp25su_jspark/neuface/nersemble_uv100.0_n100.0_l10.0/Nersemble/N017')

if __name__ == '__main__':
    from neuface.utils.config import parse_args
    cfg = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_device
    main(cfg)
