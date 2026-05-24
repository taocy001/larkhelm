"""larkhelm · agent_hub.builtin.calendar_agent — Feishu Calendar via lark_oapi.

Uses the existing ``lark_client.client`` singleton (APP_ID + APP_SECRET).
Requires the app to have calendar scopes enabled in the Feishu developer console:
  calendar:calendar:read   calendar:event:read   calendar:event:write
  calendar:freebusy:read

The primary calendar ID for "my calendar" is "primary".

Flow:
  1. Detect intent (list events / search / create / freebusy check).
  2. Call lark_oapi calendar.v4 API.
  3. Format structured response as a context block.
  4. Pass to _do_query so AI composes a human-friendly reply.

Supported intents (detected from ctx.text):
  - 查看/列出今天/本周/最近的日程    → ListCalendarEvent
  - 搜索日程 <keyword>              → SearchCalendarEvent
  - 创建/新建/安排 日程              → CreateCalendarEvent (AI extracts fields)
  - 查询空闲时间 / 检查日历          → ListFreebusy
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from larkhelm.agent_hub.agent_base import AgentExecutor
from larkhelm.agent_hub.intent_types import AgentContext, AgentResult, IntentResult

_MAX_EVENTS = 10
_MAX_OUTPUT_CHARS = 3000
_PRIMARY_CALENDAR_ID = "primary"


# ── Intent detection ─────────────────────────────────────────────────


def _detect_calendar_intent(text: str) -> str:
    """Return one of: list, search, create, freebusy."""
    tl = text.lower()
    if re.search(r"(创建|新建|安排|add|create|schedule).{0,15}(日程|会议|提醒|事件|event|meeting)", tl):
        return "create"
    if re.search(r"(空闲|有空|freebusy|availability|available|忙碌时间)", tl):
        return "freebusy"
    if re.search(r"(搜索|查找|找|search).{0,10}日程", tl):
        return "search"
    return "list"


def _time_range_from_text(text: str) -> tuple[str, str]:
    """Return (start_time, end_time) ISO8601 strings for the user's request."""
    tl = text.lower()
    now = datetime.now(tz=timezone.utc)
    if "今天" in tl or "today" in tl:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
    elif "明天" in tl or "tomorrow" in tl:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        end = start + timedelta(days=1)
    elif "本周" in tl or "this week" in tl:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=7)
    elif "下周" in tl or "next week" in tl:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=7)
        end = start + timedelta(days=7)
    elif "本月" in tl or "this month" in tl:
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = (start + timedelta(days=32)).replace(day=1)
    else:
        # Default: next 7 days
        start = now
        end = now + timedelta(days=7)
    return start.isoformat(), end.isoformat()


def _format_event(ev: dict) -> str:
    """Convert a CalendarEvent dict to a readable summary line."""
    summary = ev.get("summary") or ev.get("description", "（无标题）")[:40]
    start_t = (ev.get("start_time") or {}).get("timestamp", "")
    end_t = (ev.get("end_time") or {}).get("timestamp", "")
    try:
        start_dt = datetime.fromtimestamp(int(start_t), tz=timezone.utc).strftime("%m-%d %H:%M") if start_t else "?"
        end_dt = datetime.fromtimestamp(int(end_t), tz=timezone.utc).strftime("%H:%M") if end_t else "?"
        time_str = f"{start_dt}–{end_dt}"
    except Exception:
        time_str = f"{start_t}–{end_t}"
    return f"• {time_str}  {summary}"


# ── API calls ─────────────────────────────────────────────────────────


def _list_events(client, start_time: str, end_time: str) -> str:
    from lark_oapi.api.calendar.v4 import ListCalendarEventRequestBuilder
    try:
        req = (
            ListCalendarEventRequestBuilder()
            .calendar_id(_PRIMARY_CALENDAR_ID)
            .start_time(start_time)
            .end_time(end_time)
            .page_size(_MAX_EVENTS)
            .build()
        )
        resp = client.calendar.v4.calendar_event.list(req)
        if not resp.success():
            return f"[Calendar API error: {resp.msg}]"
        events = (resp.data.items or []) if resp.data else []
        if not events:
            return "（该时间段内无日程）"
        lines = [_format_event(ev.__dict__ if hasattr(ev, "__dict__") else ev) for ev in events]
        return "\n".join(lines[:_MAX_EVENTS])
    except Exception as e:
        return f"[list_events failed: {e}]"


def _search_events(client, query: str) -> str:
    from lark_oapi.api.calendar.v4 import (
        SearchCalendarEventRequestBuilder, SearchCalendarEventRequestBodyBuilder,
    )
    try:
        body = (
            SearchCalendarEventRequestBodyBuilder()
            .query(query)
            .page_size(_MAX_EVENTS)
            .build()
        )
        req = (
            SearchCalendarEventRequestBuilder()
            .calendar_id(_PRIMARY_CALENDAR_ID)
            .request_body(body)
            .build()
        )
        resp = client.calendar.v4.calendar_event.search(req)
        if not resp.success():
            return f"[Search API error: {resp.msg}]"
        items = (resp.data.items or []) if resp.data else []
        if not items:
            return f"（未找到关键词「{query}」相关日程）"
        return "\n".join(_format_event(ev.__dict__ if hasattr(ev, "__dict__") else ev) for ev in items)
    except Exception as e:
        return f"[search_events failed: {e}]"


def _list_freebusy(client, start_time: str, end_time: str, open_id: str) -> str:
    from lark_oapi.api.calendar.v4 import (
        ListFreebusyRequestBuilder, ListFreebusyRequestBodyBuilder,
    )
    try:
        body = (
            ListFreebusyRequestBodyBuilder()
            .time_min(start_time)
            .time_max(end_time)
            .user_id(open_id)
            .build()
        )
        req = ListFreebusyRequestBuilder().request_body(body).build()
        resp = client.calendar.v4.freebusy.list(req)
        if not resp.success():
            return f"[Freebusy API error: {resp.msg}]"
        slots = ((resp.data.freebusy_list or {}).get(open_id, {}).get("busy", [])
                 if resp.data else [])
        if not slots:
            return "（该时间段无忙碌安排，日历显示空闲）"
        lines = []
        for s in slots[:_MAX_EVENTS]:
            ts = s.get("start_time", "")
            te = s.get("end_time", "")
            lines.append(f"• 忙碌: {ts} ~ {te}")
        return "\n".join(lines)
    except Exception as e:
        return f"[freebusy failed: {e}]"


# ── Agent ─────────────────────────────────────────────────────────────


class CalendarAgent(AgentExecutor):
    agent_type = "calendar"
    description = "飞书日历：查看日程、搜索日程、查询空闲时间（需 calendar 应用权限）"
    required_capabilities = ()

    def execute(self, intent: IntentResult, ctx: AgentContext) -> AgentResult:
        from larkhelm.handlers._query import _do_query
        from larkhelm.chat_state import _get_chat_model
        from larkhelm.log import _debug_log
        from larkhelm.lark_client import client as _lark_client
        import larkhelm.config as _cfg

        start = time.monotonic()
        try:
            cal_intent = _detect_calendar_intent(ctx.text)
            start_time, end_time = _time_range_from_text(ctx.text)
            _debug_log(f"[CalendarAgent] intent={cal_intent} range={start_time[:10]}~{end_time[:10]}")

            if cal_intent == "list":
                data_str = _list_events(_lark_client, start_time, end_time)
                label = "日程列表"
            elif cal_intent == "search":
                # Extract search keyword: everything after 搜/找/search
                kw_m = re.search(r"(?:搜索?|查找?|找|search)[：:：\s]*(.+)", ctx.text, re.IGNORECASE)
                kw = kw_m.group(1).strip()[:80] if kw_m else ctx.text[:80]
                data_str = _search_events(_lark_client, kw)
                label = f"搜索日程「{kw}」"
            elif cal_intent == "freebusy":
                open_id = getattr(_cfg, "LOGGED_IN_OPEN_ID", "") or getattr(_cfg, "DEFAULT_OWNER_OPEN_ID", "")
                data_str = _list_freebusy(_lark_client, start_time, end_time, open_id)
                label = "空闲时间"
            else:
                # create — let AI handle it with calendar context injected
                data_str = (
                    "（用户想创建一个日程，请提取标题、开始时间、结束时间、参与者，"
                    "然后告知用户你已收集到哪些信息，并提示缺少的字段。）"
                )
                label = "创建日程"

            parts = [
                f"[飞书日历 · {label}]\n",
                data_str[:_MAX_OUTPUT_CHARS],
                f"\n\n---\n**用户请求：** {ctx.text}\n",
                "请根据以上日历数据回答用户请求。",
            ]
            augmented = "\n".join(parts)
            model = _get_chat_model(ctx.chat_id)
            _do_query(
                chat_id=ctx.chat_id,
                message=augmented,
                model=model,
                user_msg_id=ctx.user_msg_id,
                parent_id=ctx.parent_id,
                force_backend_id=ctx.force_backend_id,
            )
            return AgentResult(success=True, duration_sec=time.monotonic() - start)
        except Exception as e:
            _debug_log(f"[CalendarAgent] execute failed: {e}")
            return AgentResult(
                success=False, duration_sec=time.monotonic() - start, error=str(e),
            )
