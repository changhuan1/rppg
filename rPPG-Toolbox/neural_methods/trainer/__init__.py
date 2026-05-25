import importlib


def _optional_import(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        print(f"Optional trainer import skipped: {module_name} ({exc})")
        return None


BaseTrainer = _optional_import("neural_methods.trainer.BaseTrainer")
RT_MRCPNetTrainer = _optional_import("neural_methods.trainer.RT_MRCPNetTrainer")
