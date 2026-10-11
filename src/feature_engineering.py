import pandas as pd

def create_order_item_features(order_items: pd.DataFrame) -> pd.DataFrame:
    order_item_features = (order_items.groupby('order_id').agg(item_count = ('order_item_id', 'count'),
                                                               unique_products = ('product_id', 'nunique'),
                                                               product_revenue = ('price', 'sum'),
                                                               freight_cost = ('freight_value', 'sum')
                                                               ).reset_index()
                           )
    return order_item_features

def create_payment_features(payments: pd.DataFrame) -> pd.DataFrame:
    payment_features = (payments.groupby('order_id').agg(payment_count = ('payment_sequential', 'count'),
                                                         total_payment_value = ('payment_value', 'sum'),
                                                         max_installments = ('payment_installments', 'max')).reset_index()
    )
    return payment_features

def create_review_features(reviews: pd.DataFrame) -> pd.DataFrame:
    reviews = reviews.copy()
    reviews['negative_review'] = (reviews['review_score'] <= 2).astype(int)
    review_features= (reviews.groupby('order_id').agg(
        review_count = ('review_score', 'count'),
        average_review_score = ('review_score', 'mean'),
        negative_review = ('negative_review', 'max')
        ).reset_index()
    )
    return review_features


def create_order_level_dataset(
    orders,
    order_item_features,
    payment_features,
    review_features
):
    order_level = orders.copy()

    order_level = order_level.merge(
        order_item_features,
        on="order_id",
        how="left",
        validate="one_to_one"
    )

    order_level = order_level.merge(
        payment_features,
        on="order_id",
        how="left",
        validate="one_to_one"
    )

    order_level = order_level.merge(
        review_features,
        on="order_id",
        how="left",
        validate="one_to_one"
    )

    return order_level



def create_analytics_dataset(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    order_items: pd.DataFrame,
    payments: pd.DataFrame,
    reviews: pd.DataFrame
) -> pd.DataFrame:
    """
    Build a validated order-level analytical dataset.

    Aggregates order items, payments, and reviews before merging
    to prevent row multiplication. Preserves all orders.
    """

    # 1. Copy inputs so the original DataFrames remain unchanged
    orders = orders.copy()
    customers = customers.copy()

    # 2. Convert order timestamps to datetime
    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]

    for column in date_columns:
        if column in orders.columns:
            orders[column] = pd.to_datetime(
                orders[column],
                errors="coerce"
            )

    # 3. Aggregate order items: one row per order
    item_metrics = create_order_item_features(order_items)

    # 4. Aggregate payments: one row per order
    payment_metrics = create_payment_features(payments)

    # 5. Aggregate reviews: one row per order
    review_metrics = create_review_features(reviews)

    # 6. Prepare customer lookup
    customer_columns = [
        "customer_id",
        "customer_unique_id",
        "customer_zip_code_prefix"
    ]

    customer_lookup = customers[
        customer_columns
    ].drop_duplicates(subset=["customer_id"])

    # 7. Merge customer information
    analytics_orders = orders.merge(
        customer_lookup,
        on="customer_id",
        how="left",
        validate="many_to_one"
    )

    # 8. Merge aggregated item features
    analytics_orders = analytics_orders.merge(
        item_metrics,
        on="order_id",
        how="left",
        validate="one_to_one"
    )

    # 9. Merge aggregated payment features
    analytics_orders = analytics_orders.merge(
        payment_metrics,
        on="order_id",
        how="left",
        validate="one_to_one"
    )

    # 10. Merge aggregated review features
    analytics_orders = analytics_orders.merge(
        review_metrics,
        on="order_id",
        how="left",
        validate="one_to_one"
    )

    # 11. Fill missing count and monetary features where
    # no corresponding aggregate record exists
    zero_fill_columns = [
        "item_count",
        "unique_products",
        "product_revenue",
        "freight_cost",
        "payment_count",
        "total_payment_value",
        "max_installments",
        "review_count",
        "negative_review"
    ]

    for column in zero_fill_columns:
        if column in analytics_orders.columns:
            analytics_orders[column] = (
                analytics_orders[column].fillna(0)
            )

    # 12. Calculate delivery delay in days.
    # Positive = late; negative = early.
    if (
        "order_delivered_customer_date" in analytics_orders.columns
        and "order_estimated_delivery_date" in analytics_orders.columns
    ):
        analytics_orders["delivery_delay_days"] = (
            analytics_orders["order_delivered_customer_date"]
            - analytics_orders["order_estimated_delivery_date"]
        ).dt.total_seconds() / 86400

    # 13. Validate final dataset
    if len(analytics_orders) != len(orders):
        raise ValueError(
            "Order count changed during merging. "
            "Check the source keys and merge relationships."
        )

    if analytics_orders["order_id"].duplicated().any():
        raise ValueError(
            "Duplicate order IDs detected in analytical dataset."
        )

    return analytics_orders


def analyze_category_contribution(
    order_items: pd.DataFrame,
    orders: pd.DataFrame,
    products: pd.DataFrame,
    category_translation: pd.DataFrame,
    target_date,
    baseline_start,
    baseline_end
) -> pd.DataFrame:
    """
    Compare category-level product revenue on a target date
    against average daily revenue during a historical baseline.

    Dates should be supplied as YYYY-MM-DD strings or timestamps.
    The baseline end date is exclusive.
    """

    required_order_columns = {
        "order_id", "order_purchase_timestamp"
    }
    required_item_columns = {
        "order_id", "product_id", "price"
    }

    if not required_order_columns.issubset(orders.columns):
        raise ValueError("Orders table is missing required columns.")

    if not required_item_columns.issubset(order_items.columns):
        raise ValueError("Order items table is missing required columns.")

    # Prepare order dates
    order_dates = orders[
        ["order_id", "order_purchase_timestamp"]
    ].copy()

    order_dates["order_purchase_timestamp"] = pd.to_datetime(
        order_dates["order_purchase_timestamp"],
        errors="coerce"
    )

    order_dates["purchase_date"] = (
        order_dates["order_purchase_timestamp"].dt.normalize()
    )

    target_date = pd.Timestamp(target_date).normalize()
    baseline_start = pd.Timestamp(baseline_start).normalize()
    baseline_end = pd.Timestamp(baseline_end).normalize()

    if baseline_start >= baseline_end:
        raise ValueError("Baseline start must precede baseline end.")

    if baseline_start <= target_date < baseline_end:
        raise ValueError("Target date must not overlap the baseline.")

    # Attach purchase dates to order items
    item_data = order_items[
        ["order_id", "product_id", "price"]
    ].merge(
        order_dates[["order_id", "purchase_date"]],
        on="order_id",
        how="left",
        validate="many_to_one"
    )

    # Attach category names to products
    product_categories = products[
        ["product_id", "product_category_name"]
    ].merge(
        category_translation[
            ["product_category_name", "product_category_name_english"]
        ],
        on="product_category_name",
        how="left",
        validate="many_to_one"
    )

    item_data = item_data.merge(
        product_categories,
        on="product_id",
        how="left",
        validate="many_to_one"
    )

    item_data["category"] = (
        item_data["product_category_name_english"]
        .fillna(item_data["product_category_name"])
        .fillna("unknown")
    )

    # Restrict to the target date and baseline period
    target = item_data[
        item_data["purchase_date"] == target_date
    ]

    baseline = item_data[
        (item_data["purchase_date"] >= baseline_start)
        & (item_data["purchase_date"] < baseline_end)
    ]

    if target.empty:
        raise ValueError("No order items found for the target date.")

    if baseline.empty:
        raise ValueError("No order items found in the baseline period.")

    # Daily category revenue during the baseline
    baseline_daily = (
        baseline.groupby(["purchase_date", "category"])["price"]
        .sum()
        .unstack(fill_value=0)
    )

    # Include zero-revenue days in the daily average
    baseline_dates = pd.date_range(
        baseline_start,
        baseline_end - pd.Timedelta(days=1),
        freq="D"
    )

    baseline_daily = baseline_daily.reindex(
        baseline_dates,
        fill_value=0
    )

    baseline_average = baseline_daily.mean(axis=0)

    target_revenue = target.groupby("category")["price"].sum()

    result = pd.concat(
        [
            target_revenue.rename("target_revenue"),
            baseline_average.rename("baseline_avg_daily_revenue")
        ],
        axis=1
    ).fillna(0).reset_index()

    result["revenue_difference"] = (
        result["target_revenue"]
        - result["baseline_avg_daily_revenue"]
    )

    result["target_revenue_share_pct"] = (
        result["target_revenue"]
        / result["target_revenue"].sum() * 100
    )

    return result.sort_values(
        "revenue_difference",
        ascending=False
    ).reset_index(drop=True)



def analyze_seller_contribution(
    order_items: pd.DataFrame,
    orders: pd.DataFrame,
    sellers: pd.DataFrame,
    target_date,
    baseline_start,
    baseline_end
) -> pd.DataFrame:
    """
    Compare seller-level product revenue on a target date
    with average daily revenue during a historical baseline.
    The baseline end date is exclusive.
    """

    order_dates = orders[
        ["order_id", "order_purchase_timestamp"]
    ].copy()

    order_dates["purchase_date"] = pd.to_datetime(
        order_dates["order_purchase_timestamp"],
        errors="coerce"
    ).dt.normalize()

    target_date = pd.Timestamp(target_date).normalize()
    baseline_start = pd.Timestamp(baseline_start).normalize()
    baseline_end = pd.Timestamp(baseline_end).normalize()

    if baseline_start >= baseline_end:
        raise ValueError("Baseline start must precede baseline end.")

    if baseline_start <= target_date < baseline_end:
        raise ValueError("Target date must not overlap the baseline.")

    # Join order dates to items without duplicating item records
    data = order_items[
        ["order_id", "seller_id", "price"]
    ].merge(
        order_dates[["order_id", "purchase_date"]],
        on="order_id",
        how="left",
        validate="many_to_one"
    )

    target = data[data["purchase_date"] == target_date]
    baseline = data[
        (data["purchase_date"] >= baseline_start)
        & (data["purchase_date"] < baseline_end)
    ]

    if target.empty:
        raise ValueError("No order items found for the target date.")
    if baseline.empty:
        raise ValueError("No order items found in the baseline period.")

    baseline_daily = (
        baseline.groupby(["purchase_date", "seller_id"])["price"]
        .sum()
        .unstack(fill_value=0)
    )

    baseline_dates = pd.date_range(
        baseline_start,
        baseline_end - pd.Timedelta(days=1),
        freq="D"
    )

    baseline_daily = baseline_daily.reindex(
        baseline_dates,
        fill_value=0
    )

    baseline_average = baseline_daily.mean(axis=0)
    target_revenue = target.groupby("seller_id")["price"].sum()

    result = pd.concat(
        [
            target_revenue.rename("target_revenue"),
            baseline_average.rename("baseline_avg_daily_revenue")
        ],
        axis=1
    ).fillna(0).reset_index()

    result["revenue_difference"] = (
        result["target_revenue"]
        - result["baseline_avg_daily_revenue"]
    )

    result = result.merge(
        sellers[["seller_id", "seller_zip_code_prefix"]],
        on="seller_id",
        how="left",
        validate="one_to_one"
    )

    return result.sort_values(
        "revenue_difference",
        ascending=False
    ).reset_index(drop=True)



def analyze_delivery_and_reviews(
    analytics_orders: pd.DataFrame
) -> pd.DataFrame:
    """
    Summarize delivery performance and customer review scores
    for early, on-time, and late deliveries.
    """

    df = analytics_orders.copy()

    df["delivery_status"] = "Unknown"

    valid_delivery = (
        df["order_delivered_customer_date"].notna()
        & df["order_estimated_delivery_date"].notna()
    )

    df.loc[
        valid_delivery & (df["delivery_delay_days"] < 0),
        "delivery_status"
    ] = "Early"

    df.loc[
        valid_delivery & (df["delivery_delay_days"] == 0),
        "delivery_status"
    ] = "On time"

    df.loc[
        valid_delivery & (df["delivery_delay_days"] > 0),
        "delivery_status"
    ] = "Late"

    summary = (
        df.groupby("delivery_status", as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            average_delay_days=("delivery_delay_days", "mean"),
            average_review_score=("average_review_score", "mean"),
            reviewed_orders=("average_review_score", "count")
        )
    )

    return summary


def investigate_sales_anomaly(
    order_items: pd.DataFrame,
    orders: pd.DataFrame,
    products: pd.DataFrame,
    category_translation: pd.DataFrame,
    sellers: pd.DataFrame,
    target_date,
    baseline_days: int = 28
) -> dict:
    """
    Investigate a sales anomaly using category and seller contributions.

    The baseline consists of the baseline_days immediately before
    target_date. The target date is excluded from the baseline.
    """

    target_date = pd.Timestamp(target_date).normalize()

    if baseline_days < 1:
        raise ValueError("baseline_days must be at least 1.")

    baseline_start = target_date - pd.Timedelta(days=baseline_days)
    baseline_end = target_date

    category_results = analyze_category_contribution(
        order_items=order_items,
        orders=orders,
        products=products,
        category_translation=category_translation,
        target_date=target_date,
        baseline_start=baseline_start,
        baseline_end=baseline_end
    )

    seller_results = analyze_seller_contribution(
        order_items=order_items,
        orders=orders,
        sellers=sellers,
        target_date=target_date,
        baseline_start=baseline_start,
        baseline_end=baseline_end
    )

    # Calculate target-day totals directly from order items
    order_dates = orders[
        ["order_id", "order_purchase_timestamp"]
    ].copy()

    order_dates["purchase_date"] = pd.to_datetime(
        order_dates["order_purchase_timestamp"],
        errors="coerce"
    ).dt.normalize()

    target_order_ids = order_dates.loc[
        order_dates["purchase_date"] == target_date,
        "order_id"
    ]

    target_items = order_items[
        order_items["order_id"].isin(target_order_ids)
    ]

    summary = {
        "target_date": target_date.strftime("%Y-%m-%d"),
        "baseline_start": baseline_start.strftime("%Y-%m-%d"),
        "baseline_end_exclusive": baseline_end.strftime("%Y-%m-%d"),
        "baseline_days": baseline_days,
        "target_orders": int(target_order_ids.nunique()),
        "target_product_revenue": float(target_items["price"].sum()),
        "top_category": (
            category_results.iloc[0]["category"]
            if not category_results.empty else None
        ),
        "top_category_revenue_difference": (
            float(category_results.iloc[0]["revenue_difference"])
            if not category_results.empty else None
        ),
        "top_seller": (
            seller_results.iloc[0]["seller_id"]
            if not seller_results.empty else None
        ),
        "top_seller_revenue_difference": (
            float(seller_results.iloc[0]["revenue_difference"])
            if not seller_results.empty else None
        )
    }

    return {
        "summary": summary,
        "category_contributions": category_results,
        "seller_contributions": seller_results
    }


def prioritize_sales_anomalies(
    daily_sales: pd.DataFrame,
    anomaly_results: pd.DataFrame,
    baseline_days: int = 28
) -> pd.DataFrame:
    """
    Rank detected sales anomalies using historical deviations
    in daily orders and product revenue.

    The impact values are estimated deviations from a trailing
    baseline, not confirmed financial losses.
    """

    if baseline_days < 7:
        raise ValueError("Use at least 7 baseline days.")

    daily = daily_sales.copy()
    results = anomaly_results.copy()

    daily["purchase_date"] = pd.to_datetime(
        daily["purchase_date"], errors="coerce"
    ).dt.normalize()

    results["purchase_date"] = pd.to_datetime(
        results["purchase_date"], errors="coerce"
    ).dt.normalize()

    daily = daily.sort_values("purchase_date")

    metrics = ["total_orders", "product_revenue"]

    baseline = daily[["purchase_date"] + metrics].copy()

    # Use only prior dates; today's values never enter their baseline.
    baseline = baseline.set_index("purchase_date")

    baseline_stats = pd.DataFrame(index=baseline.index)

    for metric in metrics:
        baseline_stats[f"{metric}_baseline"] = (
            baseline[metric]
            .shift(1)
            .rolling(
                window=baseline_days,
                min_periods=7
            )
            .mean()
        )

        baseline_stats[f"{metric}_std"] = (
            baseline[metric]
            .shift(1)
            .rolling(
                window=baseline_days,
                min_periods=7
            )
            .std()
        )

    results = results.merge(
        baseline_stats.reset_index(),
        on="purchase_date",
        how="left",
        validate="many_to_one"
    )

    results["order_deviation"] = (
        results["total_orders"]
        - results["total_orders_baseline"]
    )

    results["revenue_deviation"] = (
        results["product_revenue"]
        - results["product_revenue_baseline"]
    )

    results["order_z_score"] = (
        results["order_deviation"]
        / results["total_orders_std"].replace(0, float("nan"))
    )

    results["revenue_z_score"] = (
        results["revenue_deviation"]
        / results["product_revenue_std"].replace(0, float("nan"))
    )

    results["estimated_revenue_deviation"] = (
        results["revenue_deviation"]
    )

    # Prioritize by absolute standardized deviation when available.
    results["severity_score"] = results[
        ["order_z_score", "revenue_z_score"]
    ].abs().max(axis=1)

    results["severity"] = "Needs review"

    valid = results["severity_score"].notna()

    results.loc[
        valid & (results["severity_score"] >= 3),
        "severity"
    ] = "High"

    results.loc[
        valid
        & (results["severity_score"] >= 2)
        & (results["severity_score"] < 3),
        "severity"
    ] = "Medium"

    results.loc[
        valid & (results["severity_score"] < 2),
        "severity"
    ] = "Low"

    return results.sort_values(
        "severity_score",
        ascending=False,
        na_position="last"
    ).reset_index(drop=True)


def build_anomaly_investigation_report(
    prioritized_anomalies: pd.DataFrame,
    order_items: pd.DataFrame,
    orders: pd.DataFrame,
    products: pd.DataFrame,
    category_translation: pd.DataFrame,
    sellers: pd.DataFrame,
    baseline_days: int = 28
) -> pd.DataFrame:
    """
    Build one evidence-based investigation summary per anomaly date.
    """

    if prioritized_anomalies.empty:
        return pd.DataFrame()

    report_rows = []

    for _, anomaly in prioritized_anomalies.iterrows():
        target_date = pd.Timestamp(
            anomaly["purchase_date"]
        ).normalize()

        investigation = investigate_sales_anomaly(
            order_items=order_items,
            orders=orders,
            products=products,
            category_translation=category_translation,
            sellers=sellers,
            target_date=target_date,
            baseline_days=baseline_days
        )

        summary = investigation["summary"]

        report_rows.append({
            "anomaly_date": target_date,
            "severity": anomaly.get("severity", "Needs review"),
            "severity_score": anomaly.get("severity_score"),
            "observed_orders": anomaly.get("total_orders"),
            "observed_product_revenue": anomaly.get("product_revenue"),
            "estimated_revenue_deviation": anomaly.get(
                "estimated_revenue_deviation"
            ),
            "top_category": summary["top_category"],
            "top_category_revenue_difference": summary[
                "top_category_revenue_difference"
            ],
            "top_seller": summary["top_seller"],
            "top_seller_revenue_difference": summary[
                "top_seller_revenue_difference"
            ],
            "investigation_status": "Evidence generated; review required"
        })

    return (
        pd.DataFrame(report_rows)
        .sort_values(
            "severity_score",
            ascending=False,
            na_position="last"
        )
        .reset_index(drop=True)
    )
