import os

from config import cfg
from utils.utils import VisAmtHelper
from utils.keyboard_helper.seghand import SegHand






if __name__ == '__main__':
    # img_path=cfg.SightToSound_paper_path
    img_path = cfg.Tencent_path
    # img_path = cfg.Record_path
    img_paths=[os.path.join(img_path,x) for x in os.listdir(img_path)]
    img_paths.sort()
    VisAmt = VisAmtHelper()
    VisAmt.init_model_load()
    for path in img_paths:
        # if not os.path.basename(path) in cfg.file_loc:continue
        if not os.path.basename(path) == 'level_4_no_02': continue
        print(path)
        # get_key_nums(path)
        VisAmt.process_img_dir(path)