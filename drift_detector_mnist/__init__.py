"""Package to create dataset, build training and prediction pipelines.

This file should define or import all the functions needed to operate the
methods defined at drift_detector_mnist/api.py. Complete the TODOs
with your own code or replace them importing your own functions.
For example:
```py
from your_module import your_function as predict
from your_module import your_function as training
```
"""

import logging
from pathlib import Path
from drift_detector_mnist import config

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


def warm(**kwargs):
    """Main/public method to start up the model"""
    # if necessary, start the model
    pass


def predict(detector_name, input_file, **options):
    """Main/public method to perform prediction"""
    sample = None  # TODO: Load sample from input_file
    detector = None  # TODO: Load the detector
    mmd, callbacks_logs = detector.compare(X=sample)
    p_value = callbacks_logs["permutation_test"]["p_value"]
    logger.debug(f"MMD: {mmd}")
    logger.debug(f"p_value: %s", round(p_value, 8))
    return {"MMD": mmd.distance, "p_value": p_value}
