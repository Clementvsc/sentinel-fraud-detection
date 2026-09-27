"""Seeded synthetic bank activity, tuned to be *hard*.

Design goals versus a naive generator:

* fraud amounts and merchants overlap heavily with legitimate spend, so the
  model has to use context (velocity, geo, entity risk, login history), not a
  cut-off on the raw amount;
* amounts are denominated in INR, tuned to look like real UPI/card spend in
  India (per-transaction amounts roughly ₹50-₹10,000+, with a long tail for
  big-ticket buys and fraud spikes);
* three explicit **adversarial** playbooks that try to evade obvious rules:
  ``amount_just_under``, ``slow_drip``, ``geo_consistent_ato``;
* **fraud rings** — several victims whose fraud shares a device or a mule
  ``beneficiary`` account, which is what the graph features are meant to catch;
* plenty of scary-looking *legitimate* activity (real travel, big-ticket buys).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List

from .. import config

# name, country, lat, lon
WORLD = [
    ("New York", "US", 40.71, -74.01), ("Chicago", "US", 41.88, -87.63),
    ("Los Angeles", "US", 34.05, -118.24), ("Austin", "US", 30.27, -97.74),
    ("London", "GB", 51.51, -0.13), ("Manchester", "GB", 53.48, -2.24),
    ("Berlin", "DE", 52.52, 13.40), ("Paris", "FR", 48.86, 2.35),
    ("Mumbai", "IN", 19.08, 72.88), ("Singapore", "SG", 1.35, 103.82),
    ("Lagos", "NG", 6.52, 3.38), ("Kyiv", "UA", 50.45, 30.52),
    ("Sao Paulo", "BR", -23.55, -46.63),
]
HOME_CITIES = WORLD[:8]
FAR_CITIES = WORLD[8:]

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "transport", "utilities", "retail",
    "entertainment", "travel", "electronics", "crypto", "gift_card", "wire_transfer",
]
LEGIT_MCC_W = [0.24, 0.20, 0.12, 0.08, 0.15, 0.09, 0.05, 0.05, 0.006, 0.006, 0.008]
CHANNELS = ["pos", "online", "atm", "transfer"]
LEGIT_CHANNEL_W = [0.52, 0.34, 0.10, 0.04]


@dataclass
class Customer:
    cust_id: str
    home_city: str
    home_country: str
    home_lat: float
    home_lon: float
    account_open: datetime
    spend_mean: float
    daily_lambda: float
    device_id: str
    card_bin: str
    favorites: Dict[str, List[str]] = field(default_factory=dict)


def _jitter(lat, lon, rng, km=12.0):
    d = km / 111.0
    return lat + rng.uniform(-d, d), lon + rng.uniform(-d, d)


def generate_customers(n: int, seed: int) -> List[Customer]:
    rng = random.Random(seed)
    start = datetime(2025, 1, 1)
    out: List[Customer] = []
    for i in range(n):
        city, country, lat, lon = rng.choice(HOME_CITIES)
        c = Customer(
            cust_id=f"C{i:05d}", home_city=city, home_country=country,
            home_lat=lat, home_lon=lon,
            account_open=start - timedelta(days=rng.randint(90, 2400)),
            # INR: mean per-transaction spend ~₹150-₹1,000 (realistic UPI/card range)
            spend_mean=round(math.exp(rng.uniform(5.0, 6.9)), 2),
            daily_lambda=round(rng.uniform(1.4, 6.0), 2),
            device_id=f"dev-{i:05d}-a",
            card_bin=f"{rng.randint(400000, 499999)}",
        )
        c.favorites = {
            mcc: [f"{mcc}_{c.cust_id}_{k}" for k in range(rng.randint(2, 5))]
            for mcc in MERCHANT_CATEGORIES
        }
        out.append(c)
    return out


def _mk_txn(cust, ts, *, amount, mcc, channel, merchant_id, country, city, lat, lon,
            device_id, label=0, beneficiary="", scenario=None):
    return {
        "type": "txn", "cust_id": cust.cust_id, "ts": ts,
        "amount": round(float(amount), 2), "mcc": mcc, "channel": channel,
        "merchant_id": merchant_id, "beneficiary": beneficiary,
        "country": country, "city": city,
        "lat": round(lat, 4), "lon": round(lon, 4),
        "device_id": device_id, "card_bin": cust.card_bin, "label": int(label),
        **({"scenario": scenario} if scenario else {}),
    }


def sample_legit_txn(cust: Customer, ts: datetime, rng: random.Random,
                     city: tuple | None = None) -> dict:
    mcc = rng.choices(MERCHANT_CATEGORIES, weights=LEGIT_MCC_W, k=1)[0]
    channel = rng.choices(CHANNELS, weights=LEGIT_CHANNEL_W, k=1)[0]
    if mcc == "wire_transfer":
        channel = "transfer"
    if channel in ("pos", "atm") and mcc in ("crypto", "wire_transfer"):
        mcc = "grocery"
    scale = {"utilities": 2.2, "travel": 4.0, "electronics": 3.5, "crypto": 3.0,
             "wire_transfer": 3.5, "gift_card": 1.6}.get(mcc, 1.0)
    amount = max(1.0, rng.lognormvariate(math.log(cust.spend_mean * scale), 0.55))
    if rng.random() < 0.85 and cust.favorites[mcc]:
        merchant = rng.choice(cust.favorites[mcc])
    else:
        merchant = f"{mcc}_new_{rng.randint(0, 9999)}"
    c_name, c_country, c_lat, c_lon = city or (
        cust.home_city, cust.home_country, cust.home_lat, cust.home_lon)
    lat, lon = _jitter(c_lat, c_lon, rng)
    benef = f"payee_{cust.cust_id}_{rng.randint(0, 3)}" if channel == "transfer" else ""
    return _mk_txn(cust, ts, amount=amount, mcc=mcc, channel=channel,
                   merchant_id=merchant, country=c_country, city=c_name,
                   lat=lat, lon=lon, device_id=cust.device_id, beneficiary=benef)


def _txn_time(day, rng):
    hour = rng.choices(range(24), weights=[
        1, 1, 1, 1, 1, 2, 4, 7, 9, 9, 8, 10, 12, 10, 8, 8, 9, 11, 12, 10, 7, 5, 3, 2], k=1)[0]
    return day + timedelta(hours=hour, minutes=rng.randint(0, 59), seconds=rng.randint(0, 59))


# --------------------------------------------------------------------------- #
# fraud playbooks  ->  list[event]     (ring_device / ring_beneficiary optional)
# --------------------------------------------------------------------------- #
def _fraud_account_takeover(cust, t0, rng, ring_device=None, ring_beneficiary=None):
    ev = []
    quiet = rng.random() < 0.45
    bad_device = ring_device or (cust.device_id if quiet else f"dev-atk-{rng.randint(1000, 9999)}")
    n_fail = rng.randint(2, 4) if quiet else rng.randint(6, 14)
    for k in range(n_fail):
        ev.append({"type": "login", "cust_id": cust.cust_id,
                   "ts": t0 + timedelta(seconds=45 * k), "success": 0, "device_id": bad_device})
    t_ok = t0 + timedelta(minutes=rng.randint(12, 20))
    ev.append({"type": "login", "cust_id": cust.cust_id, "ts": t_ok, "success": 1,
               "device_id": bad_device})
    dest = ((cust.home_city, cust.home_country, cust.home_lat, cust.home_lon)
            if quiet or rng.random() < 0.5 else rng.choice(FAR_CITIES))
    t = t_ok + timedelta(minutes=rng.randint(2, 6))
    for _ in range(rng.randint(3, 6)):
        lat, lon = _jitter(dest[2], dest[3], rng)
        mult = rng.uniform(2.0, 4.5) if quiet else rng.uniform(3.5, 9.0)
        benef = ring_beneficiary or f"mule_{rng.randint(0, 99999)}"
        ev.append(_mk_txn(cust, t, amount=cust.spend_mean * mult,
                          mcc=rng.choice(["wire_transfer", "crypto", "gift_card"]),
                          channel=rng.choice(["transfer", "online"]),
                          merchant_id=benef, beneficiary=benef,
                          country=dest[1], city=dest[0], lat=lat, lon=lon,
                          device_id=bad_device, label=1, scenario="account_takeover"))
        t += timedelta(minutes=rng.randint(2, 9))
    return ev


def _fraud_card_testing(cust, t0, rng, ring_device=None, ring_beneficiary=None):
    ev = []
    bad_device = ring_device or f"dev-bot-{rng.randint(1000, 9999)}"
    slow = rng.random() < 0.35
    t = t0
    for _ in range(rng.randint(9, 20)):
        lat, lon = _jitter(cust.home_lat, cust.home_lon, rng, km=5)
        ev.append(_mk_txn(cust, t, amount=rng.uniform(0.4, 9.0),
                          mcc=rng.choice(["retail", "gift_card", "entertainment"]),
                          channel="online", merchant_id=f"probe_{rng.randint(0, 99999)}",
                          country=cust.home_country, city=cust.home_city,
                          lat=lat, lon=lon, device_id=bad_device, label=1,
                          scenario="card_testing"))
        t += timedelta(seconds=rng.randint(45, 110) if slow else rng.randint(8, 40))
    return ev


def _fraud_stolen_card_geo(cust, t0, rng, ring_device=None, ring_beneficiary=None):
    ev = []
    # the card is physically used at home first, then abroad ~25 min later:
    # the classic impossible-travel signature
    hlat, hlon = _jitter(cust.home_lat, cust.home_lon, rng, km=6)
    ev.append(_mk_txn(cust, t0, amount=cust.spend_mean * rng.uniform(0.6, 1.4),
                      mcc="grocery", channel="pos",
                      merchant_id=rng.choice(cust.favorites["grocery"]),
                      country=cust.home_country, city=cust.home_city,
                      lat=hlat, lon=hlon, device_id=cust.device_id, label=0))
    dest = rng.choice(FAR_CITIES)
    t = t0 + timedelta(minutes=rng.randint(20, 40))
    for _ in range(rng.randint(3, 8)):
        lat, lon = _jitter(dest[2], dest[3], rng)
        ev.append(_mk_txn(cust, t, amount=cust.spend_mean * rng.uniform(1.2, 4.0),
                          mcc=rng.choice(["electronics", "retail", "restaurant", "travel"]),
                          channel="pos", merchant_id=f"{dest[0].lower()}_shop_{rng.randint(0, 999)}",
                          country=dest[1], city=dest[0], lat=lat, lon=lon,
                          device_id=cust.device_id, label=1, scenario="stolen_card_geo"))
        t += timedelta(minutes=rng.randint(20, 90))
    return ev


def _fraud_bust_out(cust, t0, rng, ring_device=None, ring_beneficiary=None):
    ev, n = [], rng.randint(14, 28)
    for i in range(n):
        ramp = 1.0 + 6.0 * (i / n)
        lat, lon = _jitter(cust.home_lat, cust.home_lon, rng, km=60)
        ev.append(_mk_txn(cust, t0 + timedelta(minutes=rng.randint(0, 55) + i * 30),
                          amount=cust.spend_mean * ramp * rng.uniform(0.8, 1.2),
                          mcc=rng.choice(["electronics", "retail", "gift_card", "travel"]),
                          channel=rng.choice(["online", "pos"]),
                          merchant_id=f"bust_{rng.randint(0, 99999)}",
                          country=cust.home_country, city=cust.home_city,
                          lat=lat, lon=lon, device_id=cust.device_id, label=1,
                          scenario="bust_out"))
    return ev


# ---- adversarial: built to slip past naive rules --------------------------
def _fraud_amount_just_under(cust, t0, rng, ring_device=None, ring_beneficiary=None):
    """Every charge sits at 90-99% of the customer's personal record, spaced out
    enough to dodge velocity rules. Beats a 'block if > personal max' rule."""
    ev = []
    dev = ring_device or cust.device_id
    t = t0
    for _ in range(rng.randint(4, 7)):
        lat, lon = _jitter(cust.home_lat, cust.home_lon, rng, km=30)
        benef = ring_beneficiary or f"mule_{rng.randint(0, 99999)}"
        ev.append(_mk_txn(cust, t, amount=cust.spend_mean * rng.uniform(3.0, 4.2),
                          mcc=rng.choice(["electronics", "retail", "wire_transfer"]),
                          channel=rng.choice(["online", "transfer"]),
                          merchant_id=benef, beneficiary=benef,
                          country=cust.home_country, city=cust.home_city,
                          lat=lat, lon=lon, device_id=dev, label=1,
                          scenario="amount_just_under"))
        t += timedelta(hours=rng.uniform(1.5, 4.0))
    return ev


def _fraud_slow_drip(cust, t0, rng, ring_device=None, ring_beneficiary=None):
    """Small, in-profile amounts to a mule, once or twice a day for a week —
    each txn looks normal; only the sustained new-beneficiary outflow betrays it."""
    ev = []
    dev = ring_device or cust.device_id
    benef = ring_beneficiary or f"mule_{rng.randint(0, 99999)}"
    for d in range(rng.randint(5, 9)):
        for _ in range(rng.randint(1, 2)):
            t = t0 + timedelta(days=d, hours=rng.randint(8, 22), minutes=rng.randint(0, 59))
            lat, lon = _jitter(cust.home_lat, cust.home_lon, rng, km=15)
            ev.append(_mk_txn(cust, t, amount=cust.spend_mean * rng.uniform(0.8, 1.6),
                              mcc="wire_transfer", channel="transfer",
                              merchant_id=benef, beneficiary=benef,
                              country=cust.home_country, city=cust.home_city,
                              lat=lat, lon=lon, device_id=dev, label=1,
                              scenario="slow_drip"))
    return ev


def _fraud_geo_consistent_ato(cust, t0, rng, ring_device=None, ring_beneficiary=None):
    """Takeover with NO geo or device tell: home city, the customer's own device,
    only 1-2 failed logins, moderate amounts. Pure behavioural-context catch."""
    ev = []
    dev = ring_device or cust.device_id
    for k in range(rng.randint(1, 2)):
        ev.append({"type": "login", "cust_id": cust.cust_id,
                   "ts": t0 + timedelta(seconds=50 * k), "success": 0, "device_id": dev})
    ev.append({"type": "login", "cust_id": cust.cust_id,
               "ts": t0 + timedelta(minutes=4), "success": 1, "device_id": dev})
    t = t0 + timedelta(minutes=rng.randint(5, 12))
    for _ in range(rng.randint(3, 5)):
        lat, lon = _jitter(cust.home_lat, cust.home_lon, rng, km=10)
        benef = ring_beneficiary or f"mule_{rng.randint(0, 99999)}"
        ev.append(_mk_txn(cust, t, amount=cust.spend_mean * rng.uniform(2.2, 3.6),
                          mcc=rng.choice(["wire_transfer", "crypto"]),
                          channel="transfer", merchant_id=benef, beneficiary=benef,
                          country=cust.home_country, city=cust.home_city,
                          lat=lat, lon=lon, device_id=dev, label=1,
                          scenario="geo_consistent_ato"))
        t += timedelta(minutes=rng.randint(6, 18))
    return ev


FRAUD_PLAYBOOKS = {
    "account_takeover": _fraud_account_takeover,
    "card_testing": _fraud_card_testing,
    "stolen_card_geo": _fraud_stolen_card_geo,
    "bust_out": _fraud_bust_out,
    "amount_just_under": _fraud_amount_just_under,
    "slow_drip": _fraud_slow_drip,
    "geo_consistent_ato": _fraud_geo_consistent_ato,
}
ADVERSARIAL = {"amount_just_under", "slow_drip", "geo_consistent_ato"}


class SyntheticSource:
    def __init__(self, n_customers: int | None = None, days: int | None = None,
                 seed: int | None = None, inject_fraud: bool = True):
        self.n = n_customers or config.TRAIN_CUSTOMERS
        self.days = days or config.TRAIN_DAYS
        self.seed = config.TRAIN_SEED if seed is None else seed
        self.inject_fraud = inject_fraud
        self.customers = generate_customers(self.n, self.seed)

    def events(self) -> List[dict]:
        rng = random.Random(self.seed + 1)
        start = datetime(2025, 1, 1)
        events: List[dict] = []

        for cust in self.customers:
            trip = None
            for d in range(self.days):
                day = start + timedelta(days=d)
                if trip is None and rng.random() < 0.010:
                    trip = (rng.choice(WORLD), day + timedelta(days=rng.randint(3, 6)))
                if trip and day > trip[1]:
                    trip = None
                city = trip[0] if trip else None
                k = _poisson(cust.daily_lambda, rng)
                if k <= 0:
                    continue
                first = None
                for _ in range(k):
                    ts = _txn_time(day, rng)
                    first = ts if first is None else min(first, ts)
                    events.append(sample_legit_txn(cust, ts, rng, city))
                if rng.random() < 0.006:      # scary-looking but legit
                    ts = day + timedelta(hours=rng.randint(0, 23), minutes=rng.randint(0, 59))
                    first = ts if first is None else min(first, ts)
                    far = rng.random() < 0.4
                    dest = rng.choice(WORLD) if far else (
                        cust.home_city, cust.home_country, cust.home_lat, cust.home_lon)
                    lat, lon = _jitter(dest[2], dest[3], rng)
                    events.append(_mk_txn(cust, ts, amount=cust.spend_mean * rng.uniform(4.0, 9.0),
                                          mcc=rng.choice(["electronics", "travel", "retail"]),
                                          channel=rng.choice(["online", "pos"]),
                                          merchant_id=f"newbig_{rng.randint(0, 99999)}",
                                          country=dest[1], city=dest[0], lat=lat, lon=lon,
                                          device_id=cust.device_id, label=0))
                events.append({"type": "login", "cust_id": cust.cust_id,
                               "ts": first - timedelta(minutes=rng.randint(1, 8)),
                               "success": 1, "device_id": cust.device_id})

        if self.inject_fraud:
            self._inject(events, rng)
        events.sort(key=lambda e: e["ts"])
        return events

    def _inject(self, events: List[dict], rng: random.Random) -> None:
        start = datetime(2025, 1, 1)
        n_vict = max(2, int(len(self.customers) * config.FRAUD_VICTIM_RATE))
        victims = rng.sample(self.customers, k=n_vict)

        # ~30% of victims belong to a ring that shares a device or a mule account
        rings: list[list[Customer]] = []
        pool = victims[:]
        while len(pool) >= 3 and rng.random() < 0.55 and len(rings) < 4:
            size = rng.randint(3, 5)
            rings.append([pool.pop() for _ in range(min(size, len(pool)))])

        def rnd_start():
            # spread uniformly across (almost) the whole window, incl. the tail,
            # so a time-ordered holdout still contains fraud
            return start + timedelta(days=rng.randint(5, max(6, self.days - 1)),
                                     hours=rng.randint(0, 23), minutes=rng.randint(0, 59))

        for ring in rings:
            scen = rng.choice(["account_takeover", "amount_just_under",
                               "slow_drip", "geo_consistent_ato"])
            shared_dev = f"dev-ring-{rng.randint(0, 9999)}" if rng.random() < 0.5 else None
            shared_ben = f"mule-ring-{rng.randint(0, 9999)}" if shared_dev is None else None
            for cust in ring:
                events.extend(FRAUD_PLAYBOOKS[scen](
                    cust, rnd_start(), rng, ring_device=shared_dev, ring_beneficiary=shared_ben))

        for cust in pool:      # lone-wolf victims (some hit more than once)
            for _ in range(1 + (rng.random() < 0.3)):
                scen = rng.choice(list(FRAUD_PLAYBOOKS))
                events.extend(FRAUD_PLAYBOOKS[scen](cust, rnd_start(), rng))


def _poisson(lam: float, rng: random.Random) -> int:
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1
