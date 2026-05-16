"""Query related table/document metadata for the find-resource workflow."""

import json
import re
from typing import Optional

from langchain_core.tools import tool

from app.core.logging import logger

# Mock metadata until ES retrieval is available.
_MOCK_TABLE_METADATA: list[dict] = [
    {
        "catalog": "hive",
        "schema": "ods",
        "table": "ods_item_sku_df",
        "layer": "原始数据",
        "domain": "商品域",
        "description": "商品 SKU 原始快照，含类目、品牌、上下架状态",
        "columns": ["item_id", "sku_id", "item_name", "category_id", "brand_id", "status"],
    },
    {
        "catalog": "hive",
        "schema": "dwd",
        "table": "dwd_item_sku_wide_df",
        "layer": "明细",
        "domain": "商品域",
        "description": "商品 SKU 明细宽表，关联类目与品牌维度",
        "columns": ["sku_id", "item_name", "category_name", "brand_name", "price", "stock_qty"],
    },
    {
        "catalog": "hive",
        "schema": "dws",
        "table": "dws_item_sales_agg_1d",
        "layer": "宽表",
        "domain": "商品域",
        "description": "商品域日粒度销售汇总宽表",
        "columns": ["sku_id", "order_cnt", "gmv", "buyer_cnt", "dt"],
    },
    {
        "catalog": "hive",
        "schema": "ads",
        "table": "ads_item_top_sales_report",
        "layer": "报表",
        "domain": "商品域",
        "description": "商品销量 TOP 报表层输出表",
        "columns": ["rank_no", "sku_id", "item_name", "gmv", "dt"],
    },
    {
        "catalog": "hive",
        "schema": "ods",
        "table": "ods_trade_order_di",
        "layer": "原始数据",
        "domain": "交易域",
        "description": "交易订单原始增量表",
        "columns": ["order_id", "buyer_id", "shop_id", "order_status", "pay_amt", "dt"],
    },
    {
        "catalog": "hive",
        "schema": "dwd",
        "table": "dwd_trade_order_detail_di",
        "layer": "明细",
        "domain": "交易域",
        "description": "订单明细事实表，行级子订单",
        "columns": ["order_id", "sub_order_id", "sku_id", "pay_amt", "discount_amt", "dt"],
    },
    {
        "catalog": "hive",
        "schema": "dws",
        "table": "dws_trade_shop_order_1d",
        "layer": "宽表",
        "domain": "交易域",
        "description": "店铺交易日汇总宽表",
        "columns": ["shop_id", "order_cnt", "pay_amt", "refund_amt", "dt"],
    },
    {
        "catalog": "hive",
        "schema": "ads",
        "table": "ads_trade_gmv_report_di",
        "layer": "报表",
        "domain": "交易域",
        "description": "交易 GMV 日报表",
        "columns": ["dt", "gmv", "order_cnt", "pay_buyer_cnt"],
    },
    {
        "catalog": "hive",
        "schema": "ods",
        "table": "ods_user_register_di",
        "layer": "原始数据",
        "domain": "用户域",
        "description": "用户注册原始日志",
        "columns": ["user_id", "register_channel", "register_time", "dt"],
    },
    {
        "catalog": "hive",
        "schema": "dwd",
        "table": "dwd_user_profile_df",
        "layer": "明细",
        "domain": "用户域",
        "description": "用户画像明细表",
        "columns": ["user_id", "gender", "age_range", "city", "member_level"],
    },
    {
        "catalog": "hive",
        "schema": "dws",
        "table": "dws_user_active_1d",
        "layer": "宽表",
        "domain": "用户域",
        "description": "用户日活跃汇总宽表",
        "columns": ["user_id", "pv", "visit_cnt", "last_visit_time", "dt"],
    },
    {
        "catalog": "hive",
        "schema": "ods",
        "table": "ods_shop_info_df",
        "layer": "原始数据",
        "domain": "店铺域",
        "description": "店铺主数据原始全量表",
        "columns": ["shop_id", "shop_name", "shop_type", "main_category", "open_date"],
    },
    {
        "catalog": "hive",
        "schema": "dwd",
        "table": "dwd_shop_operation_di",
        "layer": "明细",
        "domain": "店铺域",
        "description": "店铺经营明细（流量、转化、成交）",
        "columns": ["shop_id", "uv", "order_cnt", "pay_amt", "dt"],
    },
    {
        "catalog": "hive",
        "schema": "ads",
        "table": "ads_shop_score_report",
        "layer": "报表",
        "domain": "店铺域",
        "description": "店铺评分与服务质量报表",
        "columns": ["shop_id", "shop_name", "score", "complaint_rate", "dt"],
    },
]

_LAYER_ALIASES = {
    "原始": "原始数据",
    "ods": "原始数据",
    "明细": "明细",
    "dwd": "明细",
    "宽表": "宽表",
    "dws": "宽表",
    "汇总": "宽表",
    "报表": "报表",
    "ads": "报表",
}


def _normalize_layer(layer: str) -> str:
    text = layer.strip()
    if text in ("不限", "全部", "all", "*"):
        return "不限"
    return _LAYER_ALIASES.get(text.lower(), text)


def _normalize_domain(domain: str) -> str:
    text = domain.strip()
    if text in ("不限", "全部", "all", "*"):
        return "不限"
    if not text.endswith("域"):
        text = f"{text}域"
    return text


def _keyword_tokens(keywords: Optional[str]) -> list[str]:
    if not keywords:
        return []
    return [k.strip().lower() for k in re.split(r"[,，\s]+", keywords) if k.strip()]


def _match_keywords(record: dict, tokens: list[str]) -> bool:
    if not tokens:
        return True
    haystack = " ".join(
        [
            record.get("table", ""),
            record.get("schema", ""),
            record.get("description", ""),
            " ".join(record.get("columns", [])),
        ]
    ).lower()
    return any(token in haystack for token in tokens)


@tool
async def query_related_document(
    layer: str,
    domain: str,
    keywords: Optional[str] = None,
    limit: int = 10,
) -> str:
    """检索与分层、业务域相关的数据表元数据（当前为 mock 数据，待 ES 链路接入后替换）。

    Args:
        layer: 数仓分层，如：原始数据、明细、宽表、报表；传「不限」表示不限制分层。
        domain: 业务域，如：商品域、交易域、用户域、店铺域；传「不限」表示不限制域。
        keywords: 可选，逗号分隔检索关键词，用于在表名、描述、字段名中过滤。
        limit: 返回条数上限，默认 10，最大 50。

    Returns:
        str: JSON 字符串，包含匹配的表元数据列表。
    """
    norm_layer = _normalize_layer(layer)
    norm_domain = _normalize_domain(domain)
    tokens = _keyword_tokens(keywords)
    cap = max(1, min(limit, 50))

    matched: list[dict] = []
    for record in _MOCK_TABLE_METADATA:
        if norm_layer != "不限" and record["layer"] != norm_layer:
            continue
        if norm_domain != "不限" and record["domain"] != norm_domain:
            continue
        if not _match_keywords(record, tokens):
            continue
        matched.append(record)

    matched = matched[:cap]

    logger.info(
        "query_related_document_mock",
        layer=norm_layer,
        domain=norm_domain,
        keyword_count=len(tokens),
        result_count=len(matched),
    )

    return json.dumps(
        {
            "source": "mock",
            "layer": norm_layer,
            "domain": norm_domain,
            "keywords": keywords,
            "limit": cap,
            "count": len(matched),
            "tables": matched,
        },
        ensure_ascii=False,
    )
