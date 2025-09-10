#!/usr/bin/env python3
"""dl_utils.py in src/argusdp/fileio/download."""
# noqa: S311
import random


MOZILLA_VERSIONS = [
    "Mozilla/4.0",
    "Mozilla/5.0",
    "Mozilla/5.01",
    "Mozilla/5.02",
    "Mozilla/5.5",
]

PLATFORMS = [
    "(Windows NT 10.0; Win64; x64)",
    "(Windows NT 6.1; Win64; x64)",
    "(Windows NT 6.2; Win64; x64)",
    "(Windows NT 6.3; Win64; x64)",
    "(X11; Linux x86_64)",
    "(X11; Ubuntu; Linux x86_64)",
    "(X11; Linux i586)",
    "(iPhone; CPU iPhone OS 12_2 like Mac OS X)",
    "(iPhone; CPU iPhone OS 13_3 like Mac OS X)",
    "(iPad; CPU OS 12_2 like Mac OS X)",
    "(Macintosh; Intel Mac OS X 10_14_5)",
    "(Macintosh; Intel Mac OS X 10_15_3)",
    "(Linux; Android 11; SM-G960U)",
    "(Linux; Android 10; SM-G960U)",
    "(Linux; Android 9; SM-G960U)",
    "(Linux; U; Android 4.0.3; en-us; Xoom Build/IML77)",
    "(BlackBerry; U; BlackBerry 9800; en)",
]

ENGINES = [
    "AppleWebKit/537.36 (KHTML, like Gecko)",
    "AppleWebKit/605.1.15 (KHTML, like Gecko)",
    "AppleWebKit/602.1.50 (KHTML, like Gecko)",
    "AppleWebKit/601.1 (KHTML, like Gecko)",
    "Gecko/20100101 Firefox/73.0",
    "Gecko/20100101 Firefox/74.0",
    "Presto/2.12.388 Version/12.16",
    "Trident/7.0; rv:11.0",
]

BROWSERS = [
    "Chrome/91.0.4472.124 Safari/537.36",
    "Chrome/92.0.4515.107 Safari/537.36",
    "Chrome/90.0.4430.212 Safari/537.36",
    "Chrome/89.0.4389.72 Mobile Safari/537.36",
    "Mobile/15E148",
    "OPR/66.0.3515.36",
    "Version/12.1 Mobile/15E148 Safari/604.1",
    "Version/13.0.3 (Macintosh; Intel Mac OS X 10_15_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Safari/605.1.15",
    "Firefox/73.0",
    "Firefox/74.0",
    "Edge/18.18362",
    "Edge/17.17134",
]


def get_useragent() -> dict:
    """Generate a random User-Agent dictionary.

    Returns:
        dict: A dictionary containing the User-Agent header.

    Examples:
        >>> get_useragent()
        {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    """
    mozilla_version = random.choice(MOZILLA_VERSIONS)
    platform = random.choice(PLATFORMS)
    engine = random.choice(ENGINES)
    browser = random.choice(BROWSERS)
    user_agent = f"{mozilla_version} {platform} {engine} {browser}"

    return {"User-Agent": user_agent}
