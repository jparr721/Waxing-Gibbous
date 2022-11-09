import sys

PLATFORM_MAC = "darwin"
PLATFORM_LINUX = "linux"


def platform_is(platform: str) -> bool:
    return sys.platform == platform
