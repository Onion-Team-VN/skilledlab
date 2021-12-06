from typing import Union, List, Tuple

from skilledlab import logger
from skilledlab.internal.util.colors import StyleCode
from skilledlab.logger import Text

def skilledlab_notice(message: Union[str, List[Union[str, Tuple[str, StyleCode]]]], *, is_danger=False, is_warn=True):
    log = [('\n' + '-' * 50, Text.subtle)]
    if is_danger:
        log.append(('\SKILLEDLAB ERROR\n', [Text.danger, Text.title]))
    elif is_warn:
        log.append(('\SKILLEDLAB WARNING\n', [Text.warning, Text.title]))
    else:
        log.append(('\SKILLEDLAB MESSAGE\n', [Text.title]))

    if isinstance(message, str):
        log.append(message)
    else:
        log += message

    log.append(('\n' + '-' * 50, Text.subtle))
    logger.log(log)