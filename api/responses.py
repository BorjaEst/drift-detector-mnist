"""Module for defining custom API response parsers and content types.
This module is used by the API server to convert the output of the requested
method into the desired format. 

The module shows simple but efficient example functions. However, you may
need to modify them for your needs.
"""

import logging

from detector.api import config, utils  # pylint: disable=E0611,E0401

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


def json_response(result, **options):
    """Converts the prediction results into json return format.

    Arguments:
        result -- Result value from call, expected either dict or str
          (see https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/stable/user/v2-api.html).
        options -- Not used, added for illustration purpose.

    Raises:
        RuntimeError: Unsupported response type.

    Returns:
        Converted result into json dictionary format.
    """
    logger.debug("Response result type: %d", type(result))
    logger.debug("Response result: %d", result)
    logger.debug("Response options: %d", options)
    try:
        return utils.convert_to_serializable(result)
    except Exception as err:
        logger.error("Error converting result to json: %s", err)
        raise RuntimeError("Unsupported response type") from err


content_types = {
    "application/json": json_response,
}
