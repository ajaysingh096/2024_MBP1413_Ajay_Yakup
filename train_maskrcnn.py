from ubteacher import add_ubteacher_config
from ubteacher.engine.trainer import UBTeacherTrainer, UBRCNNTeacherTrainer, BaselineTrainer
from detectron2.checkpoint import DetectionCheckpointer
from ubteacher.modeling import EnsembleTSModel
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data import DatasetCatalog, MetadataCatalog

import os
import json

def split_dataset(cfg):
    """Function to split a dataset into 'train' and 'val' sets.
    Args:
    dataset_dicts: a list of dicts in detectron2 dataset format
    """
    with open(cfg.DATASET_DICTS, 'r') as f:
        data = json.load(f)
        train_set = data['train']
        val_set = data['val']
        return train_set, val_set
    
def register_dataset(dset_type, dataset_dicts):
        """Helper function to register a new dataset to detectron2's
        Datasetcatalog and Metadatacatalog.

        Args:
        dataset_dicts -- list of dicts in detectron2 dataset format
        cat_map -- dictionary to map categories to ids, e.g. {'ROI':0, 'JUNK':1}
        """
        reg_name = dset_type
        
        # Register dataset to DatasetCatalog
        print(f"working on '{reg_name}'...")
        
        DatasetCatalog.register(
            reg_name,
            lambda d=dset_type: dataset_dicts
        )
        MetadataCatalog.get(reg_name).set(
            thing_classes='0',
        )
        
        return MetadataCatalog

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.set_new_allowed(True) #allows custom cfg keys
    add_ubteacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg
    
def main(args):
    
    cfg = setup(args)
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # split and register
    with open(cfg.DATASET_DICTS, 'r') as f:
        train_labeled, val = split_dataset(cfg)
        register_dataset("train", train_labeled)
        register_dataset("val", val)      
            
    # train
    if cfg.SEMISUPNET.Trainer == "ubteacher":
        Trainer = UBTeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "ubteacher_rcnn":
        Trainer = UBRCNNTeacherTrainer
    else:
        Trainer = BaselineTrainer #Combined from ubteacher v1

    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "ubteacher":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            res = Trainer.test(cfg, ensem_ts_model.modelTeacher)
        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()

    
if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
