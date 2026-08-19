"""Small browser-based live 3D point-cloud viewer."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np


_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>AgenticMemoryNav 3D Map</title>
<style>html,body{margin:0;height:100%;background:#081018;color:#d7e2ea;font:14px sans-serif}
canvas{width:100%;height:100%;display:block}#status{position:fixed;left:16px;top:14px;padding:8px 10px;background:#10212bde;border:1px solid #31515f;border-radius:4px}</style></head>
<body><div id="status">waiting for map</div><canvas id="view"></canvas>
<script>
const canvas=document.getElementById('view'), ctx=canvas.getContext('2d'), status=document.getElementById('status');
let cloud=[], robot=[0,0,0], yaw=0, az=0.65, el=0.52, zoom=1.0, dragging=false, last=[0,0];
function resize(){canvas.width=innerWidth*devicePixelRatio;canvas.height=innerHeight*devicePixelRatio;ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);draw()}
addEventListener('resize',resize); canvas.addEventListener('mousedown',e=>{dragging=true;last=[e.clientX,e.clientY]});
addEventListener('mouseup',()=>dragging=false); addEventListener('mousemove',e=>{if(!dragging)return;az+=(e.clientX-last[0])*.008;el+=(e.clientY-last[1])*.008;el=Math.max(-1.45,Math.min(1.45,el));last=[e.clientX,e.clientY];draw()});
canvas.addEventListener('wheel',e=>{zoom*=Math.exp(-e.deltaY*.001);zoom=Math.max(.15,Math.min(8,zoom));draw()});
function project(p){let x=p[0]-robot[0],y=p[1]-robot[1],z=p[2]-robot[2];let ca=Math.cos(az),sa=Math.sin(az),ce=Math.cos(el),se=Math.sin(el);let qx=ca*x-sa*y,qy=sa*x+ca*y;let sx=qx,sy=ce*z-se*qy,depth=se*z+ce*qy;let scale=Math.min(innerWidth,innerHeight)*.07*zoom/(1+Math.max(0,depth)*.015);return [innerWidth/2+sx*scale,innerHeight/2-sy*scale,depth]}
function draw(){ctx.clearRect(0,0,innerWidth,innerHeight);ctx.fillStyle='#081018';ctx.fillRect(0,0,innerWidth,innerHeight);let pts=cloud.map(project).sort((a,b)=>a[2]-b[2]);for(const p of pts){let r=Math.max(1,Math.min(4,2.2- p[2]*.002));ctx.fillStyle=p[2]<0?'#61d6c5':'#f0b35a';ctx.fillRect(p[0],p[1],r,r)}let rp=project(robot), tip=project([robot[0]+Math.cos(yaw)*.7,robot[1]+Math.sin(yaw)*.7,robot[2]]);ctx.strokeStyle='#ff5b61';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(rp[0],rp[1]);ctx.lineTo(tip[0],tip[1]);ctx.stroke();ctx.fillStyle='#ff5b61';ctx.beginPath();ctx.arc(rp[0],rp[1],5,0,Math.PI*2);ctx.fill()}
async function poll(){try{let r=await fetch('/map?'+Date.now());let d=await r.json();cloud=d.points||[];robot=d.robot||[0,0,0];yaw=d.yaw||0;status.textContent=`${d.count||0} points | frame ${d.frame||'-'} | drag to orbit, wheel to zoom`;draw()}catch(e){status.textContent='waiting for map'}setTimeout(poll,250)}
resize();poll();
</script></body></html>"""


class RealtimePointCloudViewer:
    """Serve a compact world-coordinate point cloud and robot pose over HTTP."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8091, max_points: int = 30_000) -> None:
        self._lock = threading.Lock()
        self._max_points = max(100, max_points)
        self._payload: dict[str, Any] = {"points": [], "count": 0, "frame": None, "robot": [0.0, 0.0, 0.0], "yaw": 0.0}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path.startswith("/map"):
                    with owner._lock:
                        body = json.dumps(owner._payload, separators=(",", ":")).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store")
                else:
                    body = _HTML.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                return

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def update(self, points: np.ndarray, *, frame_id: str, robot: tuple[float, float, float], yaw: float) -> None:
        cloud = np.asarray(points, dtype=np.float32)
        if cloud.ndim != 2 or cloud.shape[1] != 3:
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

    def close(self) -> None:
        self._server.shutdown()