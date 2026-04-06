#!/usr/bin/env python3
"""
Feishu permission approval hook script — invoked by the Claude CLI PreToolUse hook.

Environment variables (injected by the bridge process):
  FEISHU_CHAT_ID      current Feishu chat_id
  FEISHU_PERM_SOCKET  Unix Socket path
  FEISHU_PERM_YOLO    "1" means this session has been granted Allow All; pass through immediately

Input  (stdin):  hook JSON supplied by Claude
Output (stdout): {"permissionDecision": "allow"} or empty (deny → stderr + exit 2)
"""
import json
import os
import socket
import sys


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}

    chat_id   = os.environ.get("FEISHU_CHAT_ID", "")
    sock_path = os.environ.get("FEISHU_PERM_SOCKET", "")

    # Allow All mode: pass through immediately
    if os.environ.get("FEISHU_PERM_YOLO") == "1":
        print(json.dumps({"permissionDecision": "allow"}))
        sys.exit(0)

    if not sock_path or not os.path.exists(sock_path):
        # Socket unavailable → fail-safe deny
        print("权限审批服务不可用", file=sys.stderr)
        sys.exit(2)

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(5)  # connection timeout: 5 seconds
        sock.connect(sock_path)

        request = json.dumps({
            "chat_id":     chat_id,
            "tool_name":   data.get("tool_name", "?"),
            "tool_input":  data.get("tool_input", {}),
            "tool_use_id": data.get("tool_use_id", ""),
        }) + "\n"
        sock.sendall(request.encode())

        # Block waiting for user decision (up to 5 minutes, aligned with perm.py server timeout)
        sock.settimeout(300)
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(512)
            if not chunk:
                break
            buf += chunk
        sock.close()

        resp = json.loads(buf.strip())
        if resp.get("decision") == "allow":
            print(json.dumps({"permissionDecision": "allow"}))
            sys.exit(0)
        else:
            reason = resp.get("reason", "用户拒绝了此操作")
            print(reason, file=sys.stderr)
            sys.exit(2)

    except Exception as e:
        print(f"权限 Hook 异常: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
