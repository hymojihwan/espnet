#!/usr/bin/env python3
"""Wait for a process to finish and send a local/email notification.

Examples:
    python3 local/notify_when_done.py \
        --pattern 'dump_logmel_features.py' \
        --title 'se_mel_asr1' \
        --message 'stage 3 finished'

SMTP email is optional and controlled by environment variables:
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM, SMTP_TO
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import smtplib
import subprocess
import sys
import time
from email.message import EmailMessage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--pattern", type=str, default=None)
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument("--title", type=str, default="ESPnet Job Finished")
    parser.add_argument("--message", type=str, default="The monitored job has finished.")
    parser.add_argument("--require-running", action="store_true")
    return parser.parse_args()


def pids_for_pattern(pattern: str) -> list[int]:
    proc = subprocess.run(
        ["pgrep", "-f", pattern],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return []
    pids = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid = int(line)
        except ValueError:
            continue
        if pid != os.getpid():
            pids.append(pid)
    return pids


def pid_exists(pid: int) -> bool:
    return os.path.exists(f"/proc/{pid}")


def wait_until_finished(args: argparse.Namespace) -> None:
    if args.pid is None and not args.pattern:
        raise SystemExit("Specify either --pid or --pattern")

    if args.pid is not None:
        if args.require_running and not pid_exists(args.pid):
            raise SystemExit(f"PID {args.pid} is not running")
        while pid_exists(args.pid):
            time.sleep(args.poll_interval)
        return

    assert args.pattern is not None
    if args.require_running and not pids_for_pattern(args.pattern):
        raise SystemExit(f"No running process matched pattern: {args.pattern}")
    while pids_for_pattern(args.pattern):
        time.sleep(args.poll_interval)


def send_desktop_notification(title: str, message: str) -> None:
    if shutil.which("notify-send") is None:
        return
    subprocess.run(["notify-send", title, message], check=False)


def send_terminal_bell() -> None:
    sys.stdout.write("\a")
    sys.stdout.flush()


def send_email(title: str, message: str) -> bool:
    required = ["SMTP_HOST", "SMTP_PORT", "SMTP_FROM", "SMTP_TO"]
    if not all(os.environ.get(key) for key in required):
        return False

    host = os.environ["SMTP_HOST"]
    port = int(os.environ["SMTP_PORT"])
    smtp_from = os.environ["SMTP_FROM"]
    smtp_to = os.environ["SMTP_TO"]
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")

    msg = EmailMessage()
    msg["Subject"] = title
    msg["From"] = smtp_from
    msg["To"] = smtp_to
    msg.set_content(message)

    with smtplib.SMTP(host, port, timeout=30) as server:
        server.ehlo()
        if port in (465, 587) or smtp_user or smtp_pass:
            try:
                server.starttls()
                server.ehlo()
            except smtplib.SMTPException:
                pass
        if smtp_user and smtp_pass:
            server.login(smtp_user, smtp_pass)
        server.send_message(msg)
    return True


def main() -> None:
    args = parse_args()
    wait_until_finished(args)

    sent_email = False
    try:
        sent_email = send_email(args.title, args.message)
    except Exception as exc:  # pragma: no cover
        print(f"[notify_when_done] Email send failed: {exc}", file=sys.stderr)

    send_desktop_notification(args.title, args.message)
    send_terminal_bell()

    mode = "email + desktop" if sent_email else "desktop"
    target = f"pid={args.pid}" if args.pid is not None else f"pattern={shlex.quote(args.pattern or '')}"
    print(f"[notify_when_done] Notification sent via {mode} for {target}")


if __name__ == "__main__":
    main()
