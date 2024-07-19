from utils.video_augmentation import *
from dataset.vtn_hc_dataset import VTN_HC_Dataset



def build_dataset(dataset_cfg, split,model = None,**kwargs):
    dataset = None

    if dataset_cfg['model_name'] == "vtn_hc":
        dataset = VTN_HC_Dataset(dataset_cfg['base_url'],split,dataset_cfg,**kwargs)

    assert dataset is not None
    return dataset



    