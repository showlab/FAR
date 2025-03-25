import importlib
from os import path as osp

from far.utils.misc import scandir
from far.utils.registry import TRAINER_REGISTRY

__all__ = ['build_trainer']

# automatically scan and import model modules for registry
# scan all the files under the 'models' folder and collect files ending with
# '_model.py'
trainer_folder = osp.dirname(osp.abspath(__file__))
trainer_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(trainer_folder)
    if v.startswith('trainer_')
]
# import all the model modules
_trainer_modules = [
    importlib.import_module(f'far.trainers.{file_name}')
    for file_name in trainer_filenames
]


def build_trainer(trainer_type):
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            model_type (str): Model type.
    """
    trainer = TRAINER_REGISTRY.get(trainer_type)
    return trainer
