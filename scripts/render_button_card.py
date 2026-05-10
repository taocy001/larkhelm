#!/usr/bin/env python3
"""Side-by-side preview of button card rendering across all categories.

Historical context
------------------
Originally built to preview JSON 1.0 vs JSON 2.0 button cards before the
migration that landed in ``card_builder.py``. The "1.0" leg used the
production ``_make_card`` path (which routed buttons through the now-deleted
``_make_card_json10_dict``); the "2.0" leg used the hand-rolled prototype
in this script. After the migration, **both legs render via JSON 2.0** —
``_make_card_json10_dict`` no longer exists and ``_make_card`` always
produces ``schema:"2.0"``.

Why keep the script
-------------------
1. **Post-migration sanity check**: re-running it now sends 12 cards (6
   pairs) that should look visually identical (same body markdown, same
   buttons, same color). If they don't, that's a regression to flag.
2. **Future card-schema changes**: the per-category test fixtures (single
   button, 2-button confirm, 3-button perm, 4-button plan failure, etc.)
   exercise every real-world button shape. Useful as a smoke test if we
   ever need to revisit the schema.
3. **Manual rendering verification on Feishu staging**: the unit tests
   only assert JSON shape; this script is the only way to verify Feishu
   actually renders the cards correctly end-to-end.

Purpose (current)
-----------------
Send 6 paired cards to a target chat to verify all button categories
render correctly post-migration:

  * Same body markdown, same buttons, same color across the pair.
  * Both legs go through JSON 2.0 (one via production ``_make_card``,
    one via the hand-rolled prototype below). The pair should look
    identical; any visual divergence indicates a builder drift bug.
  * Cards are labeled "(JSON 1.0 — 当前)" / "(JSON 2.0 — 提议)" for
    historical reasons; both are now JSON 2.0 in practice.

The font-size mismatch the user reported (流式中间卡 vs 流完最终卡) comes
from JSON 1.0 using ``{"tag":"div","text":{"tag":"lark_md",...}}`` while
JSON 2.0 uses ``{"tag":"markdown",...}``. Migrating all buttons to JSON 2.0
makes the renderer consistent.

Usage
-----
    python3 scripts/render_button_card.py <chat_id>

Where ``<chat_id>`` is the Feishu chat to push to (oc_xxxxxxxx). The
``--config`` and ``--data-dir`` flags forward to ``_init_runtime``.

The buttons all carry ``cmd="noop:rb_<i>"`` payloads. The existing
``handle_card_action`` dispatcher walks ``perm: / cancel: / plan_*: / ...``
prefixes and falls through unknown commands silently — clicking the test
buttons does nothing destructive.

What to check on screen
-----------------------
1. **字号一致性**: 1.0 vs 2.0 should render the body markdown at the same
   font size. (This is the bug we're fixing.)
2. **Button 渲染**: 1.0 buttons sit inside a tightly-packed action row;
   2.0 buttons via column_set should look similar but may have slightly
   different gaps/padding. Note any gross misalignment.
3. **Type 颜色**: primary (blue) / danger (red) / default should look the
   same on 1.0 and 2.0.
4. **多 button 一行**: 2/3/4-button rows. 2.0 uses ``column_set`` with
   ``width:"auto"`` columns. Confirm they wrap onto one line on desktop
   and don't overflow.
5. **可点击性**: tap each test button. The cancel-button-style cmds will
   no-op (see above). If a button hangs or errors, that's a callback wiring
   regression and we should NOT proceed to migrate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make ``larkhelm`` importable when run from the project root or scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── JSON 2.0 button helpers (prototype; will move to card_builder.py later) ──

def _btn_type(label: str) -> str:
    """Mirror card_builder._btn_type so 1.0/2.0 cards have identical color logic."""
    if any(k in label for k in ("允许", "确认", "✅", "同意", "Yes", "OK", "继续", "▶")):
        return "primary"
    if any(k in label for k in ("拒绝", "取消", "删除", "❌", "No", "Deny", "🛑")):
        return "danger"
    return "default"


def _v2_button(label: str, cmd: str) -> dict:
    return {
        "tag": "button",
        "text": {"tag": "plain_text", "content": label},
        "type": _btn_type(label),
        "behaviors": [{"type": "callback", "value": {"cmd": cmd}}],
    }


def _v2_buttons_block(buttons: list[tuple[str, str]]) -> dict:
    """Single button → bare button element. Multi → column_set with width:auto columns."""
    if len(buttons) == 1:
        return _v2_button(*buttons[0])
    return {
        "tag": "column_set",
        "horizontal_spacing": "small",
        "columns": [
            {"tag": "column", "width": "auto",
             "elements": [_v2_button(label, cmd)]}
            for (label, cmd) in buttons
        ],
    }


def build_v2_card(title: str, body_md: str, color: str,
                  buttons: list[tuple[str, str]]) -> str:
    """Construct a JSON 2.0 card with body markdown + a button block."""
    elements = [{"tag": "markdown", "content": body_md}]
    elements.append(_v2_buttons_block(buttons))
    card = {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"template": color,
                   "title": {"tag": "plain_text", "content": title}},
        "body": {"elements": elements},
    }
    return json.dumps(card, ensure_ascii=False)


# ── Test cases — one per real-world button category ──────────────────────

# Body markdown deliberately includes: a paragraph, **bold**, `inline code`,
# a bullet list, and a code block — so the renderer's font-size for each
# construct is visible side-by-side.
_SAMPLE_BODY = (
    "**正在思考中**——本卡片用于对比 JSON 1.0 与 JSON 2.0 的渲染差异。\n\n"
    "- 普通段落文本：你应当看到这一行字号一致\n"
    "- 内联代码：`_make_card_dict()` 应当用等宽字体\n\n"
    "```python\nprint('hello world')\n```\n\n"
    "> 引用块：注意整段是否同字号"
)


CASES: list[dict] = [
    {
        "name": "1-button cancel (streaming card)",
        "color": "grey",
        "buttons": [("🛑 取消", "noop:rb_1_cancel")],
    },
    {
        "name": "2-button confirm/cancel (cmd_doc /doc write)",
        "color": "orange",
        "buttons": [("确认替换", "noop:rb_2_confirm"),
                    ("取消", "noop:rb_2_cancel")],
    },
    {
        "name": "2-button breakpoint (crew_card crew_bp)",
        "color": "yellow",
        "buttons": [("✅ 继续执行", "noop:rb_3_confirm"),
                    ("🛑 取消", "noop:rb_3_cancel")],
    },
    {
        "name": "3-button perm (allow/deny/yolo)",
        "color": "orange",
        "buttons": [("✅ 允许", "noop:rb_4_allow"),
                    ("❌ 拒绝", "noop:rb_4_deny"),
                    ("🚀 允许全部", "noop:rb_4_yolo")],
    },
    {
        "name": "4-button plan step-confirm (failed-step)",
        "color": "yellow",
        "buttons": [("🔄 重试本步", "noop:rb_5_retry"),
                    ("▶ 继续", "noop:rb_5_continue"),
                    ("⏭ 跳过下一步", "noop:rb_5_skip"),
                    ("🛑 取消", "noop:rb_5_cancel")],
    },
    {
        "name": "3-button /status or /help",
        "color": "turquoise",
        "buttons": [("♻️ 重置会话", "noop:rb_6_reset"),
                    ("🔗 接入终端", "noop:rb_6_pickup"),
                    ("切换 gemini", "noop:rb_6_switch")],
    },
]


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("chat_id", help="Target Feishu chat_id (oc_xxxxxxxx)")
    parser.add_argument("--config", help="Path to larkhelm config.json")
    parser.add_argument("--data-dir", help="Override DATA_DIR")
    parser.add_argument("--delay-ms", type=int, default=400,
                        help="Delay between sends to keep the timeline ordered")
    args = parser.parse_args()

    # Bootstrap config + lark client (mimicking bridge.main()).
    import larkhelm.config as _cfg
    _cfg._init_runtime(config_path=args.config, data_dir=args.data_dir)
    if not _cfg.APP_ID or not _cfg.APP_SECRET:
        print("ERROR: APP_ID / APP_SECRET not configured", file=sys.stderr)
        return 2

    import lark_oapi as lark
    import larkhelm.lark_client as _lc
    _lc.client = lark.Client.builder().app_id(_cfg.APP_ID).app_secret(_cfg.APP_SECRET).build()

    from larkhelm.card_builder import _make_card
    from larkhelm.lark_client import _send_card_raw

    # Header card explaining what to look for.
    intro = _make_card(
        "🧪 Button schema 对比：JSON 1.0 vs 2.0",
        (
            f"本次共发 **{len(CASES) * 2}** 张卡片对比，每对相邻两张卡片"
            f"分别用 JSON 1.0（当前生产路径）和 JSON 2.0（提议迁移目标）"
            f"渲染同一份内容。请逐对核对：\n\n"
            "1. **字号**：body markdown 是否一致（这是要修的 bug）\n"
            "2. **Button 渲染**：颜色 / 间距 / 是否一行排开\n"
            "3. **可点击**：每个按钮 tap 一下，应当 no-op 不报错\n\n"
            f"按 **{args.delay_ms}ms** 间隔依次发送，预计耗时 "
            f"{len(CASES) * 2 * args.delay_ms / 1000:.0f}s。"
        ),
        color="blue",
    )
    if not _send_card_raw(args.chat_id, intro):
        print(f"ERROR: failed to send intro card to {args.chat_id}", file=sys.stderr)
        return 2

    delay_s = args.delay_ms / 1000.0
    for i, case in enumerate(CASES, 1):
        # JSON 1.0 (current) — built via the production card_builder path.
        v1_title = f"({i}/{len(CASES)}) {case['name']} — JSON 1.0"
        v1_card = _make_card(v1_title, _SAMPLE_BODY,
                             color=case["color"], buttons=case["buttons"])
        if not _send_card_raw(args.chat_id, v1_card):
            print(f"WARN: send failed for case {i} v1.0", file=sys.stderr)
        time.sleep(delay_s)

        # JSON 2.0 (proposed) — hand-rolled via _v2_buttons_block.
        v2_title = f"({i}/{len(CASES)}) {case['name']} — JSON 2.0"
        v2_card = build_v2_card(v2_title, _SAMPLE_BODY,
                                color=case["color"], buttons=case["buttons"])
        if not _send_card_raw(args.chat_id, v2_card):
            print(f"WARN: send failed for case {i} v2.0", file=sys.stderr)
        time.sleep(delay_s)

        print(f"[{i}/{len(CASES)}] sent: {case['name']}")

    # Final summary card.
    summary = _make_card(
        "✅ 对比卡发送完成",
        (
            f"共 {len(CASES)} 对（{len(CASES) * 2} 张）卡片已发送。\n\n"
            "**评估口径**：\n"
            "- 全部 6 对字号一致 + 按钮渲染无明显回归 → 可安全迁移到 JSON 2.0\n"
            "- 任何一对存在视觉问题 → 在迁移前先解决该 case\n"
            "- 任何按钮 tap 报错 / 卡死 → callback 协议有兼容性问题，**不要**迁移\n\n"
            "**核对完后**：把发现的问题告诉我，下一步会动 `card_builder.py`。"
        ),
        color="green",
    )
    _send_card_raw(args.chat_id, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
