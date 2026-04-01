from __future__ import annotations

import pandas as pd


def build_purchase_plan(current_holdings: pd.DataFrame, budget: float) -> tuple[pd.DataFrame, float]:
    stocks = current_holdings[current_holdings["ticker"] != "CASH"].copy()
    if stocks.empty or budget <= 0:
        return pd.DataFrame(columns=[
            "ticker",
            "target_weight",
            "latest_price",
            "target_value",
            "shares_to_buy",
            "actual_value",
            "actual_weight",
            "weight_gap",
        ]), float(max(budget, 0.0))

    stocks = stocks[["ticker", "weight", "latest_price"]].copy()
    stocks = stocks[stocks["latest_price"] > 0].reset_index(drop=True)
    total_weight = float(stocks["weight"].sum())
    if total_weight <= 0:
        stocks["weight"] = 1.0 / len(stocks)
    else:
        stocks["weight"] = stocks["weight"] / total_weight

    stocks["target_value"] = budget * stocks["weight"]
    stocks["shares_to_buy"] = (stocks["target_value"] / stocks["latest_price"]).astype(int)
    spent = float((stocks["shares_to_buy"] * stocks["latest_price"]).sum())
    cash_left = float(budget - spent)

    while True:
        stocks["actual_value"] = stocks["shares_to_buy"] * stocks["latest_price"]
        stocks["gap_value"] = stocks["target_value"] - stocks["actual_value"]
        affordable = stocks[stocks["latest_price"] <= cash_left]
        if affordable.empty:
            break
        ticker_to_add = affordable.sort_values(by="gap_value", ascending=False).iloc[0]["ticker"]
        price = float(stocks.loc[stocks["ticker"] == ticker_to_add, "latest_price"].iloc[0])
        if price > cash_left:
            break
        stocks.loc[stocks["ticker"] == ticker_to_add, "shares_to_buy"] += 1
        cash_left -= price

    stocks["actual_value"] = stocks["shares_to_buy"] * stocks["latest_price"]
    total_invested = float(stocks["actual_value"].sum())
    denominator = total_invested + cash_left
    stocks["actual_weight"] = stocks["actual_value"] / denominator if denominator > 0 else 0.0
    stocks["weight_gap"] = stocks["actual_weight"] - stocks["weight"]
    stocks = stocks.rename(columns={"weight": "target_weight"})
    columns = [
        "ticker",
        "target_weight",
        "latest_price",
        "target_value",
        "shares_to_buy",
        "actual_value",
        "actual_weight",
        "weight_gap",
    ]
    return stocks[columns].sort_values(by="actual_value", ascending=False).reset_index(drop=True), float(cash_left)
