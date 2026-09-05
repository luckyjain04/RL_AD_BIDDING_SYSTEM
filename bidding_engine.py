"""
bidding_engine.py
-----------------
Real-time bidding (RTB) simulation.

Architecture
------------
  Auction  →  BidRequest  →  [N Bidders]  →  bids  →  Auction resolves
                                                          ↓
                                                    winner notified
                                                    (pay 2nd-price)

Bidding strategies implemented
-------------------------------
1. FixedBidder       – constant bid (naive baseline)
2. CTRBidder         – bid = base_cpm × predicted_ctr
3. ValueBidder       – bid = expected_value = ctr × conversion_value
4. BudgetPacingBidder– ValueBidder with spend-rate throttling
5. ThresholdBidder   – only bid if pCTR ≥ threshold
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BidRequest:
    """One ad-impression opportunity sent to all bidders."""
    impression_id: str
    user_profile:  str
    ad_category:   str
    features:      np.ndarray          # pre-extracted feature vector
    floor_price:   float = 0.10        # minimum CPM ($)
    timestamp:     float = 0.0


@dataclass
class BidResponse:
    bidder_name:  str
    bid_cpm:      float                 # $ per 1000 impressions
    p_click:      float                 # predicted click probability
    p_conv:       float = 0.0
    expected_value: float = 0.0


@dataclass
class AuctionResult:
    impression_id: str
    winner:        Optional[str]        # bidder name
    winning_bid:   float                # 1st-price cleared
    clearing_price: float               # 2nd-price (actual charge)
    all_bids:      list[BidResponse] = field(default_factory=list)
    clicked:       bool = False         # ground-truth outcome
    converted:     bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Bidder implementations
# ─────────────────────────────────────────────────────────────────────────────
class BaseBidder:
    def __init__(self, name: str, daily_budget: float = 1000.0):
        self.name          = name
        self.daily_budget  = daily_budget
        self.spent         = 0.0
        self.won_auctions  = 0
        self.total_clicks  = 0
        self.total_convs   = 0

    def bid(self, request: BidRequest, model) -> Optional[BidResponse]:
        raise NotImplementedError

    def notify_win(self, price: float, clicked: bool, converted: bool):
        self.spent        += price / 1000   # CPM → per-impression cost
        self.won_auctions += 1
        self.total_clicks += int(clicked)
        self.total_convs  += int(converted)

    def remaining_budget(self) -> float:
        return max(0.0, self.daily_budget - self.spent)

    def stats(self) -> dict:
        ctr  = self.total_clicks / max(1, self.won_auctions)
        cvr  = self.total_convs  / max(1, self.won_auctions)
        cpc  = self.spent / max(1, self.total_clicks)
        return {
            "name":          self.name,
            "won":           self.won_auctions,
            "spent_$":       round(self.spent, 4),
            "clicks":        self.total_clicks,
            "conversions":   self.total_convs,
            "ctr":           round(ctr, 4),
            "cvr":           round(cvr, 4),
            "cpc_$":         round(cpc, 4),
            "budget_used_%": round(self.spent / self.daily_budget * 100, 1),
        }


class FixedBidder(BaseBidder):
    """Baseline: always bid the same CPM regardless of pCTR."""

    def __init__(self, fixed_cpm: float = 2.0, **kw):
        super().__init__(**kw)
        self.fixed_cpm = fixed_cpm

    def bid(self, request: BidRequest, model) -> Optional[BidResponse]:
        if self.remaining_budget() <= 0:
            return None
        p_click = float(model.predict_proba(request.features.reshape(1, -1))[0])
        return BidResponse(
            bidder_name=self.name,
            bid_cpm=self.fixed_cpm,
            p_click=p_click,
        )


class CTRBidder(BaseBidder):
    """Bid proportional to predicted CTR."""

    def __init__(self, base_cpm: float = 30.0, **kw):
        super().__init__(**kw)
        self.base_cpm = base_cpm

    def bid(self, request: BidRequest, model) -> Optional[BidResponse]:
        if self.remaining_budget() <= 0:
            return None
        p_click = float(model.predict_proba(request.features.reshape(1, -1))[0])
        bid_cpm = self.base_cpm * p_click
        bid_cpm = max(bid_cpm, request.floor_price)
        return BidResponse(
            bidder_name=self.name,
            bid_cpm=bid_cpm,
            p_click=p_click,
        )


class ValueBidder(BaseBidder):
    """
    Bid = pCTR × pConv × conversion_value  (expected-value bidding).
    This is the industry-standard oCPM/CPA strategy.
    """

    def __init__(self, conversion_value: float = 50.0,
                 conv_rate_prior: float = 0.03, **kw):
        super().__init__(**kw)
        self.conversion_value  = conversion_value
        self.conv_rate_prior   = conv_rate_prior

    def bid(self, request: BidRequest, model) -> Optional[BidResponse]:
        if self.remaining_budget() <= 0:
            return None
        p_click = float(model.predict_proba(request.features.reshape(1, -1))[0])
        p_conv  = p_click * self.conv_rate_prior
        ev      = p_click * p_conv * self.conversion_value * 1000  # → CPM
        bid_cpm = max(ev, request.floor_price)
        return BidResponse(
            bidder_name=self.name,
            bid_cpm=bid_cpm,
            p_click=p_click,
            p_conv=p_conv,
            expected_value=ev,
        )


class BudgetPacingBidder(ValueBidder):
    """
    ValueBidder + smooth spend pacing.
    Throttles bid when spend-rate is ahead of the daily target.
    """

    def __init__(self, elapsed_fraction: float = 0.0, **kw):
        super().__init__(**kw)
        self._elapsed_frac = elapsed_fraction   # updated externally

    def set_elapsed(self, fraction: float):
        self._elapsed_frac = max(0.0, min(1.0, fraction))

    def bid(self, request: BidRequest, model) -> Optional[BidResponse]:
        resp = super().bid(request, model)
        if resp is None:
            return None
        # pacing ratio: if spent_fraction > elapsed_fraction → throttle
        spent_frac = self.spent / (self.daily_budget + 1e-8)
        target_frac = self._elapsed_frac
        if target_frac > 0 and spent_frac > target_frac * 1.1:
            pace_factor = target_frac / (spent_frac + 1e-8)
            resp.bid_cpm *= pace_factor
        resp.bid_cpm = max(resp.bid_cpm, request.floor_price)
        return resp


class ThresholdBidder(CTRBidder):
    """Skip auction if pCTR is below threshold (avoids wasted impressions)."""

    def __init__(self, pclick_threshold: float = 0.05, **kw):
        super().__init__(**kw)
        self.threshold = pclick_threshold

    def bid(self, request: BidRequest, model) -> Optional[BidResponse]:
        if self.remaining_budget() <= 0:
            return None
        p_click = float(model.predict_proba(request.features.reshape(1, -1))[0])
        if p_click < self.threshold:
            return None     # pass on this impression
        bid_cpm = self.base_cpm * p_click
        bid_cpm = max(bid_cpm, request.floor_price)
        return BidResponse(
            bidder_name=self.name,
            bid_cpm=bid_cpm,
            p_click=p_click,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Second-price auction engine
# ─────────────────────────────────────────────────────────────────────────────
class AuctionEngine:
    """
    Simulates a standard second-price (Vickrey) RTB auction.

    Clearing price = second-highest bid (or floor_price if only one bid).
    """

    def __init__(self, bidders: list[BaseBidder], model):
        self.bidders = bidders
        self.model   = model
        self.history : list[AuctionResult] = []

    def run_auction(self, request: BidRequest) -> AuctionResult:
        responses: list[BidResponse] = []
        for bidder in self.bidders:
            resp = bidder.bid(request, self.model)
            if resp is not None and resp.bid_cpm >= request.floor_price:
                responses.append(resp)

        result = AuctionResult(
            impression_id=request.impression_id,
            winner=None,
            winning_bid=0.0,
            clearing_price=request.floor_price,
            all_bids=responses,
        )

        if not responses:
            self.history.append(result)
            return result

        # sort descending by bid
        responses.sort(key=lambda r: r.bid_cpm, reverse=True)
        winner_resp = responses[0]
        clearing = responses[1].bid_cpm if len(responses) > 1 else request.floor_price

        result.winner        = winner_resp.bidder_name
        result.winning_bid   = winner_resp.bid_cpm
        result.clearing_price = clearing

        # simulate outcome
        result.clicked   = np.random.random() < winner_resp.p_click
        result.converted = result.clicked and (
            np.random.random() < winner_resp.p_conv
        )

        # notify winner
        winning_bidder = next(b for b in self.bidders
                              if b.name == winner_resp.bidder_name)
        winning_bidder.notify_win(clearing, result.clicked, result.converted)

        self.history.append(result)
        return result

    def run_simulation(self, requests: list[BidRequest],
                       verbose: bool = True) -> pd.DataFrame:
        """Run all requests through the auction loop."""
        n = len(requests)
        for i, req in enumerate(requests):
            # update pacing bidders' elapsed fraction
            for b in self.bidders:
                if isinstance(b, BudgetPacingBidder):
                    b.set_elapsed(i / n)
            self.run_auction(req)

        if verbose:
            self._print_summary()
        return self.results_df()

    def results_df(self) -> pd.DataFrame:
        rows = []
        for r in self.history:
            bids_str = " | ".join(
                f"{b.bidder_name}={b.bid_cpm:.2f}" for b in r.all_bids
            )
            rows.append({
                "impression_id":  r.impression_id,
                "winner":         r.winner,
                "winning_bid":    r.winning_bid,
                "clearing_price": r.clearing_price,
                "clicked":        r.clicked,
                "converted":      r.converted,
                "n_bidders":      len(r.all_bids),
                "all_bids":       bids_str,
            })
        return pd.DataFrame(rows)

    def _print_summary(self):
        print("\n" + "=" * 60)
        print("  AUCTION SIMULATION SUMMARY")
        print("=" * 60)
        total = len(self.history)
        contested = sum(1 for r in self.history if r.winner)
        print(f"  Auctions run    : {total:,}")
        print(f"  Impressions won : {contested:,}  ({contested/total*100:.1f}%)")
        print()
        for b in self.bidders:
            s = b.stats()
            print(f"  {s['name']:25s}  won={s['won']:4d}  "
                  f"CTR={s['ctr']:.3f}  spent=${s['spent_$']:.2f}  "
                  f"budget={s['budget_used_%']}%")


# ─────────────────────────────────────────────────────────────────────────────
# Build default bidder set for the demo
# ─────────────────────────────────────────────────────────────────────────────
def build_default_bidders() -> list[BaseBidder]:
    return [
        FixedBidder(    name="FixedBidder",    fixed_cpm=1.5, daily_budget=500),
        CTRBidder(      name="CTRBidder",      base_cpm=30.0, daily_budget=800),
        ValueBidder(    name="ValueBidder",    conversion_value=60.0,
                        daily_budget=1000),
        BudgetPacingBidder(name="PacingBidder", conversion_value=60.0,
                           daily_budget=600),
        ThresholdBidder(name="ThresholdBidder", pclick_threshold=0.06,
                        base_cpm=25.0, daily_budget=700),
    ]


if __name__ == "__main__":
    # quick smoke test with a dummy model
    class _DummyModel:
        def predict_proba(self, X):
            return np.random.uniform(0.01, 0.30, size=(len(X), 1))

    rng = np.random.default_rng(42)
    requests = [
        BidRequest(
            impression_id=str(i),
            user_profile="tech_enthusiast",
            ad_category="electronics",
            features=rng.standard_normal(43),
            floor_price=0.10,
            timestamp=float(i),
        )
        for i in range(500)
    ]

    bidders = build_default_bidders()
    engine  = AuctionEngine(bidders, _DummyModel())
    df      = engine.run_simulation(requests)
    print(df.head())
