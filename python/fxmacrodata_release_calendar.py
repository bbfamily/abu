"""FXMacroData release-calendar helper for macro-aware backtests.

The public USD calendar endpoint does not require an API key. Strategy code can
use the returned top-tier dates to avoid opening new risk before major releases.
"""

from __future__ import print_function

import json
from datetime import date, timedelta

try:
    from urllib.parse import urlencode
    from urllib.request import urlopen
except ImportError:
    from urllib import urlencode
    from urllib2 import urlopen


BASE_URL = "https://fxmacrodata.com/api/v1/calendar/{currency}"


def fetch_calendar(currency="USD", start_date=None, end_date=None, timeout=20):
    today = date.today()
    start_date = start_date or today.isoformat()
    end_date = end_date or (today + timedelta(days=14)).isoformat()
    query = urlencode({"start_date": start_date, "end_date": end_date})

    with urlopen("{}?{}".format(BASE_URL.format(currency=currency), query), timeout=timeout) as response:
        payload = json.load(response)

    return payload.get("data", [])


def top_tier_blackout_dates(events):
    blackout_dates = set()
    for event in events:
        if event.get("top_tier_for_currency") or event.get("market_tier") == 1:
            timestamp = event.get("announcement_datetime_utc") or event.get("announcement_datetime_local")
            blackout_dates.add((timestamp or event.get("date", ""))[:10])
    return sorted(date_value for date_value in blackout_dates if date_value)


if __name__ == "__main__":
    calendar = fetch_calendar(start_date="2026-07-01", end_date="2026-07-20")
    print("Top-tier USD macro blackout dates:")
    for blackout_date in top_tier_blackout_dates(calendar):
        print("  - {}".format(blackout_date))
