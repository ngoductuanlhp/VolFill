"""Local preview server with HTTP Range support (needed to seek in the animation videos).

  python scripts/serve.py [--port 8000]

`python -m http.server` ignores Range headers, so browsers treat videos as unseekable;
GitHub Pages supports ranges, so this only matters locally.
"""
import argparse
import functools
import http.server
import os
import re
import socketserver
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    def send_head(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip()) if rng else None
        if not m or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        start = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2) or 0))
        end = int(m.group(2)) if m.group(1) and m.group(2) else size - 1
        if start >= size:
            self.send_error(416, "Requested Range Not Satisfiable")
            return None
        end = min(end, size - 1)
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "_remaining", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            try:
                outputfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                break
            remaining -= len(chunk)
        self._remaining = None

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()


class QuietRangeHandler(RangeHandler):
    def log_message(self, *args):
        pass


def make_server(port, quiet=False, bind="127.0.0.1"):
    handler = functools.partial(QuietRangeHandler if quiet else RangeHandler, directory=str(ROOT))
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    httpd = socketserver.ThreadingTCPServer((bind, port), handler)
    httpd.daemon_threads = True
    return httpd


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--bind", default="127.0.0.1")
    args = ap.parse_args()
    httpd = make_server(args.port, bind=args.bind)
    print(f"Serving {ROOT} at http://{args.bind}:{args.port}/")
    httpd.serve_forever()
