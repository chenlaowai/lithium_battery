from mmdet.datasets import CocoDataset
from mmdet.datasets.api_wrappers import COCO
from mmdet.registry import DATASETS as DETDATASETS
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS as SEGDATASETS


# train
#  - images
#  - annotations
#  annotations.json
# val
#  - images
#  - annotations
#  annotations.json

@SEGDATASETS.register_module()
class DefectSegDataset(BaseSegDataset):
    """Defect dataset.
    """
    METAINFO = dict(
        #classes=('background', 'ear_fold_up', 'ear_fold_horizontal', 'ear_unweld'),
        #palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0]]
        classes = ('background', 'tab_question', 'foreign_object', 'metal_chip', 'ear_foldup', 'ear_damaged', 'weld_crack',),
              palette = [[0, 0, 0], [244, 108, 59], [0, 255, 0], [0, 85, 255], [255, 255, 0], [255, 85, 255], [151, 255, 248], ]
                    )

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)


@DETDATASETS.register_module()
class DefectDetDataset(CocoDataset):
    """Dataset for Litchi."""
    METAINFO = dict(
        classes=('ear_fold_up', 'ear_fold_horizontal', 'ear_unweld',),
        palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142)],
    )

    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True
