#!/usr/bin/env python3
"""
gold_vrp_feed.py - publish gold's options-implied volatility as a small JSON feed.

WHY THIS EXISTS
    MetaTrader 5 has no access to options data. Implied volatility, the market's
    forward-looking expectation of movement, does not exist inside the terminal. This
    script computes it outside, from a free public option chain, and writes one small
    JSON object that any MQL5 program can read over HTTP.

WHY IT SOLVES FOR IMPLIED VOLATILITY INSTEAD OF READING IT
    Option chains usually ship an implied volatility column. For gold that column
    reported 0.20 percent, and on some contracts 0.00 percent, while the prices of those
    same contracts implied about 22 percent. So we ignore that column and solve
    Black-Scholes backwards from the price the market is actually paying, which is what
    implied volatility means in the first place.

WHAT IT COMPUTES
    spot                  current price of GLD, the gold ETF whose options we read
    iv_atm                at-the-money implied volatility of the nearest usable expiry
    iv_30d                the same, interpolated to a 30-day horizon
    iv_30d_interpolated   true when a real interpolation happened
    rv_20d_gld            realized volatility of GLD over the last 20 sessions
    vrp_reference         iv_30d - rv_20d_gld, in volatility points, as a sanity check
    plus the expiry used, days to expiry, and a UTC timestamp

USAGE
    pip install yfinance pandas numpy
    python gold_vrp_feed.py                  print the JSON
    python gold_vrp_feed.py --out feed.json  also write it to a file
    python gold_vrp_feed.py --debug          show the contracts and prices used
"""

import argparse
import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

TICKER = "GLD"          # gold ETF: liquid options, no dividend, quoted in dollars
RV_WINDOW = 20          # sessions used for realized volatility
TRADING_DAYS = 252      # annualization factor for realized volatility
RISK_FREE = 0.04        # short-rate approximation; at-the-money IV barely moves with it
MIN_OI = 10             # ignore contracts with almost no open interest
IV_FLOOR, IV_CEIL = 0.02, 2.00   # sanity bounds: 2% to 200% annualized
MAX_EXPIRIES = 8        # stop scanning after this many usable expiries


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(spot, strike, t_years, vol, rate, is_call) -> float:
    """Black-Scholes price of a European option on a non-dividend-paying asset."""
    if t_years <= 0 or vol <= 0:
        intrinsic = (spot - strike) if is_call else (strike - spot)
        return max(intrinsic, 0.0)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * t_years) / (vol * math.sqrt(t_years))
    d2 = d1 - vol * math.sqrt(t_years)
    if is_call:
        return spot * norm_cdf(d1) - strike * math.exp(-rate * t_years) * norm_cdf(d2)
    return strike * math.exp(-rate * t_years) * norm_cdf(-d2) - spot * norm_cdf(-d1)


def implied_vol(market_price, spot, strike, t_years, rate, is_call):
    """
    Solve Black-Scholes backwards for volatility, by bisection.

    An option's price rises monotonically with volatility, so we can narrow a bracket
    until the model price matches the market price. Bisection is slower than Newton's
    method and far more reliable, which is the right trade for a feed that must not
    return nonsense.
    """
    intrinsic = max((spot - strike) if is_call else (strike - spot), 0.0)
    if market_price <= intrinsic + 1e-6:
        return None                      # no time value left, nothing to imply

    low, high = IV_FLOOR, IV_CEIL
    if bs_price(spot, strike, t_years, high, rate, is_call) < market_price:
        return None                      # price above what 200% volatility can explain

    for _ in range(100):
        mid = 0.5 * (low + high)
        if bs_price(spot, strike, t_years, mid, rate, is_call) < market_price:
            low = mid
        else:
            high = mid
        if high - low < 1e-6:
            break
    return 0.5 * (low + high)


def mid_price(row):
    """Prefer the bid/ask midpoint; fall back to the last trade."""
    bid = float(row.get("bid") or 0.0)
    ask = float(row.get("ask") or 0.0)
    if bid > 0 and ask > 0 and ask >= bid:
        return 0.5 * (bid + ask)
    last = float(row.get("lastPrice") or 0.0)
    return last if last > 0 else None


def realized_volatility(closes: pd.Series, window: int = RV_WINDOW) -> float:
    """Annualized standard deviation of daily log returns."""
    logret = np.log(closes / closes.shift(1)).dropna()
    if len(logret) < window:
        return float("nan")
    return float(logret.tail(window).std(ddof=1) * math.sqrt(TRADING_DAYS))


def atm_iv_for_expiry(tk, expiry, spot, t_years, debug=False):
    """
    Average the implied volatility of the call and the put nearest the money.

    Using both sides cancels most of the skew between puts and calls, so the result
    reads as expected movement rather than as directional positioning.
    """
    chain = tk.option_chain(expiry)
    results = []
    for frame, is_call in ((chain.calls, True), (chain.puts, False)):
        if frame is None or frame.empty:
            continue
        usable = frame[frame["openInterest"].fillna(0) >= MIN_OI]
        if usable.empty:
            usable = frame
        row = usable.iloc[(usable["strike"] - spot).abs().argsort().iloc[0]]
        price = mid_price(row)
        if price is None:
            continue
        vol = implied_vol(price, spot, float(row["strike"]), t_years, RISK_FREE, is_call)
        if vol is None or not (IV_FLOOR < vol < IV_CEIL):
            continue
        results.append(vol)
        if debug:
            kind = "call" if is_call else "put "
            print(f"    {kind} strike {float(row['strike']):8.2f}  price {price:7.2f}  "
                  f"solved IV {vol * 100:6.2f}%  (chain field said "
                  f"{float(row.get('impliedVolatility') or 0) * 100:.2f}%)")
    if not results:
        return None
    return sum(results) / len(results)


def build_feed(debug=False) -> dict:
    tk = yf.Ticker(TICKER)

    history = tk.history(period="6mo", interval="1d")
    if history.empty:
        raise RuntimeError(f"no price history returned for {TICKER}")
    spot = float(history["Close"].iloc[-1])
    rv = realized_volatility(history["Close"])

    today = datetime.now(timezone.utc).date()
    candidates = []
    for expiry in tk.options:
        days = (datetime.strptime(expiry, "%Y-%m-%d").date() - today).days
        if days < 7 or days > 90:
            continue
        if debug:
            print(f"  expiry {expiry} ({days} days):")
        vol = atm_iv_for_expiry(tk, expiry, spot, days / 365.0, debug)
        if vol:
            candidates.append((days, expiry, vol))
        # stop once the 30-day horizon is bracketed on both sides
        bracketed = (any(c[0] <= 30 for c in candidates)
                     and any(c[0] >= 30 for c in candidates))
        if bracketed or len(candidates) >= MAX_EXPIRIES:
            break
    if not candidates:
        raise RuntimeError("no expiry produced a usable at-the-money implied volatility")

    candidates.sort()
    near_days, near_expiry, near_iv = candidates[0]

    # interpolate linearly in time between the expiries straddling 30 days
    iv_30d = near_iv
    interpolated = False
    before = [c for c in candidates if c[0] <= 30]
    after = [c for c in candidates if c[0] >= 30]
    if before and after:
        lo_days, _, lo_iv = before[-1]
        hi_days, _, hi_iv = after[0]
        if hi_days > lo_days:
            weight = (30 - lo_days) / (hi_days - lo_days)
            iv_30d = lo_iv + weight * (hi_iv - lo_iv)
            interpolated = True
        else:
            iv_30d = lo_iv

    return {
        "symbol": TICKER,
        "spot": round(spot, 4),
        "iv_atm": round(near_iv, 6),
        "iv_30d": round(iv_30d, 6),
        "iv_30d_interpolated": interpolated,
        "rv_20d_gld": None if math.isnan(rv) else round(rv, 6),
        "vrp_reference": None if math.isnan(rv) else round(iv_30d - rv, 6),
        "expiry": near_expiry,
        "days_to_expiry": near_days,
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "option chain, implied volatility solved from market prices",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish gold implied volatility as JSON")
    parser.add_argument("--out", help="also write the JSON to this file")
    parser.add_argument("--debug", action="store_true", help="show contracts and prices used")
    args = parser.parse_args()

    if args.debug:
        print("solving implied volatility from market prices:")
    feed = build_feed(args.debug)
    text = json.dumps(feed, indent=2)
    print(("\n" if args.debug else "") + text)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(f"\nwritten to {args.out}")

    iv = feed["iv_30d"] * 100
    rv = feed["rv_20d_gld"]
    horizon = "interpolated to 30 days" if feed["iv_30d_interpolated"] else \
              f"nearest expiry, {feed['days_to_expiry']} days"
    print(f"\n30-day implied volatility: {iv:.2f}%  ({horizon})")
    if rv is not None:
        print(f"20-day realized volatility (GLD): {rv * 100:.2f}%")
        print(f"reference premium: {feed['vrp_reference'] * 100:+.2f} volatility points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
