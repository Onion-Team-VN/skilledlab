from typing import List

from skilledlab.internal.logger.destinations import Destination

def create_destination() -> List[Destination]:
    from skilledlab.internal.logger.destinations.console import ConsoleDestination
    return [ConsoleDestination(True)]