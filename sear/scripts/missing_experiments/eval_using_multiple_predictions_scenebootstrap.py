import gc

import torch
import tyro

from sear import logger
from sear.scripts.eval_using_predictions_scenebootstrap import (
    EvalParameters,
    main,
)
from sear.scripts.missing_experiments.eval_using_multiple_predictions import (
    EvalParametersMultiple,
)


def eval_one_by_one(params: EvalParametersMultiple) -> None:
    methods = sorted(list(params.method_predictions_root.iterdir()))
    for i, method_name in enumerate(methods):
        if not method_name.is_dir() or (
            params.subset is not None and method_name.name not in params.subset
        ):
            logger.info(f"Skipping {method_name.name}: {i + 1}/{len(methods)}")
            continue

        params_single = EvalParameters(
            method_predictions_folder=params.method_predictions_root / method_name.name,
            depth_eps=params.depth_eps,
            output_folder_root=params.output_folder_root,
            num_bootstrap=params.num_bootstrap,
            custom_aggregation_file_path=params.custom_aggregation_file_path,
        )
        main(params_single)

        logger.info(f"Done {method_name.name}: {i + 1}/{len(methods)}")

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    params = tyro.cli(EvalParametersMultiple)
    eval_one_by_one(params=params)
