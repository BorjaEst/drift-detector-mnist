"""Generic tests environment configuration. This file implement all generic
fixtures to simplify model and api specific testing.

Modify this file only if you need to add new fixtures or modify the existing
related to the environment and generic tests.
"""

# pylint: disable=redefined-outer-name
import inspect
import os
import pathlib
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec, patch

import numpy as np
import pytest
from frouros.detectors import concept_drift, data_drift

import api


@pytest.fixture(scope="session", autouse=True)
def original_datapath():
    """Fixture to generate a original directory path for datasets."""
    return pathlib.Path(api.config.DATA_PATH).absolute()


@pytest.fixture(scope="session", autouse=True)
def original_modelspath():
    """Fixture to generate a original directory path for datasets."""
    return pathlib.Path(api.config.MODELS_PATH).absolute()


@pytest.fixture(scope="session", params=os.listdir("tests/configurations"))
def config_file(request):
    """Fixture to provide each deepaas configuration path."""
    config_str = f"tests/configurations/{request.param}"
    return pathlib.Path(config_str).absolute()


@pytest.fixture(scope="module", name="testdir")
def create_testdir():
    """Fixture to generate a temporary directory for each test module."""
    with tempfile.TemporaryDirectory() as testdir:
        os.chdir(testdir)
        yield testdir


@pytest.fixture(scope="module", autouse=True)
def copytree_data(testdir, original_datapath):
    """Fixture to copy the original data directory to the test directory."""
    shutil.copytree(original_datapath, f"{testdir}/{api.config.DATA_PATH}")


def generate_signature(names, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD):
    """Function to generate dynamically signatures."""
    parameters = [inspect.Parameter(name, kind) for name in names]
    return inspect.Signature(parameters=parameters)


def generate_fields_fixture(signature):
    """Function to generate dynamically fixtures with dynamic arguments."""
    def fixture_function(**options):  # fmt: skip
        return {k: v for k, v in options.items() if v is not None}
    fixture_function.__signature__ = signature
    return pytest.fixture(scope="module")(fixture_function)


@pytest.fixture(scope="module")
def metadata():
    """Fixture to return get_metadata to assert properties."""
    return api.get_metadata()


# Generate and inject fixtures for predict arguments
fields_predict = api.schemas.PredArgsSchema().fields
signature = generate_signature(fields_predict.keys())
globals()["predict_kwds"] = generate_fields_fixture(signature)


@pytest.fixture(scope="module")
def predictions(predict_kwds):
    """Fixture to return predictions to assert properties."""
    comparative_result = (
        SimpleNamespace(distance=0.28),
        {
            "permutation_test": {
                "observed_statistic": 0.28,
                "permuted_statistics": 0.01 * np.random.rand(10),
                "p_value": 0.01,
            }
        },
    )
    detector = create_autospec(
        data_drift.MMD, compare=create_autospec(data_drift.MMD.compare)
    )
    detector.encoder = MagicMock()  # Mock encoder forward method
    detector.compare.return_value = comparative_result
    with patch("torch.load", autospec=True) as load:
        load.return_value = detector
        return api.predict(**predict_kwds)
