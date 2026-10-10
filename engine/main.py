"""PhotoVault — Main sidecar process entrypoint (T14).

Starts the HTTP REST / IPC API server and background job worker.
Emits a single JSON ready handshake to stdout for the Tauri shell:
    {"status": "ready", "port": <port>, "url": "<base_url>"}
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from engine.api import create_api_server
from engine.db.schema import init_db
from engine.queue import JobWorker


def main() -> int:
    parser = argparse.ArgumentParser(description="PhotoVault Core Engine Sidecar")
    parser.add_argument("--db", default="vault.db", help="Path to SQLite vault.db")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=0, help="Port to bind (0 for ephemeral)")
    args = parser.parse_args()

    # Ensure database schema is initialized
    conn = init_db(args.db)
    conn.close()

    # Start API server
    server = create_api_server(db_path=args.db, host=args.host, port=args.port)
    server.start()

    # Start background job worker
    worker = JobWorker(db_path=args.db, poll_interval=0.5)
    worker.start()

    # Emit JSON ready handshake to stdout
    port = server.server.server_address[1]
    handshake = {
        "status": "ready",
        "port": port,
        "url": server.base_url,
    }
    sys.stdout.write(json.dumps(handshake) + "\n")
    sys.stdout.flush()

    stop_flag = False

    def handle_signal(signum: int, frame: object) -> None:
        nonlocal stop_flag
        stop_flag = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        while not stop_flag:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        server.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
