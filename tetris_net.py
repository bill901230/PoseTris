#!/usr/bin/env python3
"""
tetris_net.py
~~~~~~~~~~~~~
乾淨分離的 **Tetris 雙人連線模組**
使用單一 TCP 連線 + 行分割 JSON。

公開 API
--------
```
recv_q, send = create_link(mode: str, peer_ip: str | None = None, port: int = 9009)
```
* `mode`  : "host" | "peer" (必填)
* `peer_ip`: 當 `mode=="peer"` 時指定伺服器 IP
* `recv_q` : `queue.Queue`，另一個執行緒持續把對方 JSON dict 放進來
* `send(d)`: 將 dict 送給對手（自動換行與 encode），若連線斷線會安靜失敗

使用手冊
--------
```python
# 在 tetris.py 頂端
from tetris_net import create_link

# ── 建立連線 ──
if args.dual:                         # args.dual = [mode, ip?]
    mode = args.dual[0]
    peer = args.dual[1] if mode == "peer" else None
    recv_q, net_send = create_link(mode, peer)
else:
    recv_q, net_send = None, lambda *_: None
```
然後照先前教學：
* `net_send({"type":"attack",...})`
* `while not recv_q.empty(): ...` 讀取

本模組不依賴任何 pygame / mediapipe，可重用於 CLI 測試。"""
from __future__ import annotations
import socket, threading, queue, json, time
from typing import Tuple, Callable, Optional


def _recv_worker(sock: socket.socket, q: queue.Queue):
    """內部：阻塞讀 socket → 行分割 → JSON → queue"""
    buf = ""
    try:
        while True:
            data = sock.recv(1024)
            if not data:
                break  # 連線中斷
            buf += data.decode()
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                try:
                    q.put(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass  # socket 關閉
    finally:
        sock.close()


def create_link(mode: str, peer_ip: Optional[str] = None, port: int = 9009
                 ) -> Tuple[queue.Queue, Callable[[dict], None]]:
    """建立 host / peer 連線並回傳 (recv_queue, send_func)."""
    mode = mode.lower()
    if mode not in {"host", "peer"}:
        raise ValueError("mode 必須是 'host' 或 'peer'")
    if mode == "peer" and not peer_ip:
        raise ValueError("peer mode 需要 peer_ip")

    q: queue.Queue = queue.Queue()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    if mode == "host":
        sock.bind(("", port)); sock.listen(1)
        print(f"[NET] Host: listening on {port}…")
        conn, addr = sock.accept()
        print(f"[NET] Connected from {addr}")
    else:  # peer
        print(f"[NET] Peer: connect to {peer_ip}:{port}…")
        while True:
            try:
                sock.connect((peer_ip, port)); break
            except (ConnectionRefusedError, OSError):
                time.sleep(1)
        conn = sock
        print("[NET] Connected!")

    # 啟動接收執行緒
    threading.Thread(target=_recv_worker, args=(conn, q), daemon=True).start()

    def send(d: dict):
        try:
            conn.sendall((json.dumps(d) + "\n").encode())
        except (BrokenPipeError, OSError):
            pass  # 連線掛掉，忽略
    
    return q, send
