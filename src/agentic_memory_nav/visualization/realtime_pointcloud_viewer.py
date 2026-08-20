"""Browser-based live viewer: head-camera JPEG + 3D point-cloud map side by side.

Isaac Sim's WebRTC livestream (``omni.kit.livestream.webrtc``) requires the
proprietary Omniverse Streaming Client and cannot be embedded in a plain browser
page. This viewer serves the same head-camera frames as plain JPEG over HTTP
(the browser-achievable equivalent, mirroring ``LiveFrameServer``) alongside the
in-browser 3D point-cloud map, so a single page shows both the live camera view
and the reconstructed 3D map at once.

The page carries two panes:
    * left  -> live head-camera JPEG (polled every ~200 ms),
    * right -> 3D point-cloud map (polled every ~250 ms, orbit/zoom as before).
"""

from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover - PIL is available in the Isaac Sim runtime
    Image = None


def _placeholder_jpeg() -> bytes:
    """A 1x1 dark JPEG so the <img> shows a solid box, not a broken-image icon."""
    if Image is None:
        return b""
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), (8, 16, 24)).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>AgenticMemoryNav - Camera + 3D Map</title>
<style>
html,body{margin:0;height:100%;background:#081018;color:#d7e2ea;font:14px sans-serif}
.wrap{display:flex;height:100%}
.pane{position:relative;flex:1 1 50%;height:100%;min-width:0}
.pane.left{border-right:1px solid #31515f}
#cam{width:100%;height:100%;object-fit:contain;background:#000;display:block}
#view{width:100%;height:100%;display:block}
#status{position:absolute;left:16px;top:12px;padding:6px 10px;background:#10212bde;border:1px solid #31515f;border-radius:4px;font-size:12px;z-index:2}
.lbl{position:absolute;left:14px;bottom:10px;font-size:12px;color:#61d6c5;z-index:2}
</style></head>
<body><div class="wrap">
<div class="pane left"><img id="cam" src="/cam.jpg" alt="head camera"><div class="lbl">head camera (live)</div></div>
<div class="pane right"><div id="status">waiting for map</div><canvas id="view"></canvas>
<div class="lbl">3D point-cloud map - drag to orbit, wheel to zoom</div></div>
</div>
<script>
// ---- camera pane (poll the latest JPEG, like a plain MJPEG feed) ----
const cam=document.getElementById('cam');
function camPoll(){cam.src='/cam.jpg?'+Date.now()}
setInterval(camPoll, 200);
// ---- 3D map pane (orbit + point cloud) ----
const canvas=document.getElementById('view'), ctx=canvas.getContext('2d'), status=document.getElementById('status');
let cloud=[], robot=[0,0,0], yaw=0, az=0.65, el=0.52, zoom=1.0, dragging=false, last=[0,0];
function resize(){canvas.width=canvas.clientWidth*devicePixelRatio;canvas.height=canvas.clientHeight*devicePixelRatio;ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);draw()}
addEventListener('resize',resize);
canvas.addEventListener('mousedown',e=>{dragging=true;last=[e.clientX,e.clientY]});
addEventListener('mouseup',()=>dragging=false);
addEventListener('mousemove',e=>{if(!dragging)return;az+=(e.clientX-last[0])*.008;el+=(e.clientY-last[1])*.008;el=Math.max(-1.45,Math.min(1.45,el));last=[e.clientX,e.clientY];draw()});
canvas.addEventListener('wheel',e=>{zoom*=Math.exp(-e.deltaY*.001);zoom=Math.max(.15,Math.min(8,zoom));draw()});
function project(p){let x=p[0]-robot[0],y=p[1]-robot[1],z=p[2]-robot[2];let ca=Math.cos(az),sa=Math.sin(az),ce=Math.cos(el),se=Math.sin(el);let qx=ca*x-sa*y,qy=sa*x+ca*y;let sx=qx,sy=ce*z-se*qy,depth=se*z+ce*qy;let scale=Math.min(canvas.clientWidth,canvas.clientHeight)*.07*zoom/(1+Math.max(0,depth)*.015);return [canvas.clientWidth/2+sx*scale,canvas.clientHeight/2-sy*scale,depth]}
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#081018';ctx.fillRect(0,0,canvas.width,canvas.height);let pts=cloud.map(project).sort((a,b)=>a[2]-b[2]);for(const p of pts){let r=Math.max(1,Math.min(4,2.2-p[2]*.002));ctx.fillStyle=p[2]<0?'#61d6c5':'#f0b35a';ctx.fillRect(p[0],p[1],r,r)}let rp=project(robot), tip=project([robot[0]+Math.cos(yaw)*.7,robot[1]+Math.sin(yaw)*.7,robot[2]]);ctx.strokeStyle='#ff5b61';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(rp[0],rp[1]);ctx.lineTo(tip[0],tip[1]);ctx.stroke();ctx.fillStyle='#ff5b61';ctx.beginPath();ctx.arc(rp[0],rp[1],5,0,Math.PI*2);ctx.fill()}
async function poll(){try{let r=await fetch('/map?'+Date.now());let d=await r.json();cloud=d.points||[];robot=d.robot||[0,0,0];yaw=d.yaw||0;status.textContent=`${d.count||0} points | frame ${d.frame||'-'}`;draw()}catch(e){status.textContent='waiting for map'}setTimeout(poll,250)}
resize();poll();
</script></body></html>"""


class RealtimePointCloudViewer:
    """Serve a compact world-coordinate point cloud, robot pose, and a live head-camera JPEG."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8091, max_points: int = 30_000) -> None:
        self._lock = threading.Lock()
        self._max_points = max(100, max_points)
        self._payload: dict[str, Any] = {
            "points": [], "count": 0, "frame": None, "robot": [0.0, 0.0, 0.0], "yaw": 0.0
        }
        self._cam_jpeg: bytes = b""
        self._placeholder: bytes = _placeholder_jpeg()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path.startswith("/map"):
                    with owner._lock:
                        body = json.dumps(owner._payload, separators=(",", ":")).encode()
                    content_type = "application/json"
                    extra = {"Cache-Control": "no-store"}
                elif self.path.startswith("/cam") or "/cam.jpg" in self.path:
                    with owner._lock:
                        body = owner._cam_jpeg or owner._placeholder
                    content_type = "image/jpeg"
                    extra = {"Cache-Control": "no-store"}
                else:
                    body = _HTML.encode()
                    content_type = "text/html; charset=utf-8"
                    extra = {}
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                for key, value in extra.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (ConnectionError, BrokenPipeError):
                    # The client (browser) disconnected mid-request; ignore.
                    return

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                return

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def update(
        self,
        points: np.ndarray,
        *,
        frame_id: str,
        robot: tuple[float, float, float],
        yaw: float,
        rgb: "np.ndarray | None" = None,
    ) -> None:
        cloud = np.asarray(points, dtype=np.float32)
        if cloud.size == 0:
            cloud = np.empty((0, 3), dtype=np.float32)
        elif cloud.ndim == 1:
            if cloud.shape[0] == 3:
                cloud = cloud.reshape(1, 3)
            else:
                raise ValueError("Viewer point cloud must have shape (N, 3)")
        elif cloud.ndim != 2 or cloud.shape[1] != 3:
            raise ValueError("Viewer point cloud must have shape (N, 3)")
        finite = np.isfinite(cloud).all(axis=1)
        cloud = cloud[finite]
        if len(cloud) > self._max_points:
            indices = np.linspace(0, len(cloud) - 1, self._max_points, dtype=np.int64)
            cloud = cloud[indices]
        with self._lock:
            self._payload = {
                "points": np.round(cloud, 3).tolist(),
                "count": len(cloud),
                "frame": frame_id,
                "robot": list(robot),
                "yaw": float(yaw),
            }
            if rgb is not None and Image is not None:
                try:
                    buf = io.BytesIO()
                    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="JPEG", quality=85)
                    self._cam_jpeg = buf.getvalue()
                except Exception:
                    pass

    def close(self) -> None:
        self._server.shutdown()