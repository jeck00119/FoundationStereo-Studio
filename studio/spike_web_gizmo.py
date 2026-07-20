"""Standalone spike #2: the Blender/Unreal-style gizmo.

three.js `TransformControls` (translate arrows + scale handles) driving a measure
box over a real point cloud, rendered with WebGL, embedded in a PySide6
QWebEngineView. The gizmo is 100% three.js — none of it is hand-rolled — which is
the whole point: it looks and feels like a game engine because it IS the widget
those tools' web cousins use.

For the spike the point-in-box measurement is computed in JS (cheap over ~700k
points); the real integration would ship the box bounds back to measure.py over
QWebChannel. Nothing here touches the app.

Run:  .venv\\Scripts\\python.exe -m studio.spike_web_gizmo  [optional_cloud.ply]
Needs internet once (three.js from a CDN).
"""
from __future__ import annotations

import functools
import http.server
import os
import socketserver
import sys
import threading

import numpy as np

DEFAULT_PLY = r"C:\Users\andre\Desktop\s2ms_test.ply"
MAX_POINTS = 700_000        # downsample target — plenty to judge feel, snappy in WebGL

_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><style>
  html,body{margin:0;height:100%;background:#0a0c12;overflow:hidden;
    font-family:"Cascadia Mono",Consolas,monospace;color:#e7ecf4}
  #c{position:fixed;inset:0}
  #readout{position:fixed;left:14px;top:12px;background:rgba(10,12,18,.82);
    padding:9px 12px;border-radius:8px;font-size:12px;line-height:1.5;white-space:pre;
    pointer-events:none;border:1px solid rgba(245,217,92,.25)}
  #bar{position:fixed;left:14px;bottom:14px;display:flex;gap:8px}
  button{background:#161b26;color:#cdd6e4;border:1px solid #2a3342;border-radius:7px;
    padding:7px 13px;font:inherit;font-size:12px;cursor:pointer}
  button.on{background:#f5d95c;color:#1a1a1a;border-color:#f5d95c;font-weight:600}
  #hint{position:fixed;right:14px;bottom:14px;font-size:11px;color:#8394ab;text-align:right}
</style></head><body>
<canvas id="c"></canvas>
<div id="readout">loading…</div>
<div id="bar">
  <button id="bT" class="on">Move (W)</button>
  <button id="bR">Rotate (E)</button>
  <button id="bS">Scale (R)</button>
</div>
<div id="hint">drag empty space to orbit · scroll to zoom<br>W move · E rotate · R scale — drag a handle</div>
<script type="importmap">
{ "imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
}}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';

const N=__N__, CX=__CX__, CY=__CY__, CZ=__CZ__, BSIZE=__BSIZE__, PTSIZE=__PTSIZE__;
const CAMDIST=__CAMDIST__;

const renderer = new THREE.WebGLRenderer({canvas:document.getElementById('c'),antialias:true});
renderer.setPixelRatio(devicePixelRatio); renderer.setSize(innerWidth,innerHeight);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0c12);
const camera = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, 0.01, 100000);
camera.up.set(0,-1,0);                                   // world Y is image-down
camera.position.set(CX, CY, CZ - CAMDIST);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(CX,CY,CZ); controls.update();

let positions=null;
const readout=document.getElementById('readout');

// ---- load the cloud (binary, same-origin over the local server) ----
Promise.all([
  fetch('positions.bin').then(r=>r.arrayBuffer()),
  fetch('colors.bin').then(r=>r.arrayBuffer())
]).then(([pb,cb])=>{
  positions = new Float32Array(pb);
  const cols = new Uint8Array(cb);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions,3));
  geo.setAttribute('color', new THREE.BufferAttribute(cols,3,true));   // normalized
  const mat = new THREE.PointsMaterial({size:PTSIZE, vertexColors:true, sizeAttenuation:false});
  scene.add(new THREE.Points(geo, mat));
  measure();
});

// ---- the measure box + gizmo ----
const boxGeo = new THREE.BoxGeometry(1,1,1);
const box = new THREE.Mesh(boxGeo, new THREE.MeshBasicMaterial(
  {color:0xf5d95c, transparent:true, opacity:0.10, depthWrite:false}));
box.position.set(CX,CY,CZ); box.scale.set(BSIZE,BSIZE,BSIZE);
box.add(new THREE.LineSegments(new THREE.EdgesGeometry(boxGeo),
  new THREE.LineBasicMaterial({color:0xf5d95c})));
scene.add(box);

const tc = new TransformControls(camera, renderer.domElement);
tc.setMode('translate'); tc.setSpace('world'); tc.attach(box);
scene.add(tc);
tc.addEventListener('dragging-changed', e=>{ controls.enabled = !e.value; });
tc.addEventListener('objectChange', measure);

// mode buttons + keys (three.js convention: W move, E rotate, R scale)
const bT=document.getElementById('bT'), bR=document.getElementById('bR'), bS=document.getElementById('bS');
function setMode(m){ tc.setMode(m);
  // translate feels right in world axes; rotate/scale in the box's OWN axes, so
  // once you have tilted the box the handles follow it (and scale can't shear it)
  tc.setSpace(m==='translate' ? 'world' : 'local');
  bT.classList.toggle('on', m==='translate');
  bR.classList.toggle('on', m==='rotate');
  bS.classList.toggle('on', m==='scale'); }
bT.onclick=()=>setMode('translate'); bR.onclick=()=>setMode('rotate'); bS.onclick=()=>setMode('scale');
addEventListener('keydown', e=>{ if(e.key==='w')setMode('translate');
  if(e.key==='e')setMode('rotate'); if(e.key==='r')setMode('scale'); });

// ---- measurement in the BOX'S OWN frame (OBB) so rotation is meaningful ----
// Each point is projected onto the box's three world-space axes (the columns of
// its rotation matrix): local = ( (p-c)·X, (p-c)·Y, (p-c)·Z ). "Inside" is then
// |local| <= size/2 per axis, and the height is the span along the box's Z axis —
// which, once you tilt the box to lie along a pin, is the true pin height.
// (trim/voxel/fill still live in measure.py; this is the spike's honest preview.)
function measure(){
  if(!positions) return;
  const e = new THREE.Matrix4().makeRotationFromQuaternion(box.quaternion).elements;
  const Xx=e[0],Xy=e[1],Xz=e[2], Yx=e[4],Yy=e[5],Yz=e[6], Zx=e[8],Zy=e[9],Zz=e[10];
  const cx=box.position.x, cy=box.position.y, cz=box.position.z;
  const hx=box.scale.x/2, hy=box.scale.y/2, hz=box.scale.z/2;
  let n=0, lxmn=1e9,lxmx=-1e9, lymn=1e9,lymx=-1e9, lzmn=1e9,lzmx=-1e9;
  for(let i=0;i<positions.length;i+=3){
    const dx=positions[i]-cx, dy=positions[i+1]-cy, dz=positions[i+2]-cz;
    const lx=dx*Xx+dy*Xy+dz*Xz;  if(lx<-hx||lx>hx) continue;
    const ly=dx*Yx+dy*Yy+dz*Yz;  if(ly<-hy||ly>hy) continue;
    const lz=dx*Zx+dy*Zy+dz*Zz;  if(lz<-hz||lz>hz) continue;
    n++;
    if(lx<lxmn)lxmn=lx; if(lx>lxmx)lxmx=lx;
    if(ly<lymn)lymn=ly; if(ly>lymx)lymx=ly;
    if(lz<lzmn)lzmn=lz; if(lz>lzmx)lzmx=lz;
  }
  const f=v=>v.toFixed(2);
  readout.textContent = n===0 ? '▣  box is empty'
    : `▣  ${n.toLocaleString()} pts   ·  measured in the box's own frame\n`+
      `   height   ${f(lzmx-lzmn)} mm   (along the box's blue Z axis)\n`+
      `   section  ${f(lxmx-lxmn)} × ${f(lymx-lymn)} mm   (X · Y)\n`+
      `   box      ${f(box.scale.x)} × ${f(box.scale.y)} × ${f(box.scale.z)} mm`;
}

addEventListener('resize', ()=>{ camera.aspect=innerWidth/innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth,innerHeight); });
(function loop(){ requestAnimationFrame(loop); controls.update(); renderer.render(scene,camera); })();
</script></body></html>
"""


def _load_cloud(path):
    import open3d as o3d

    pc = o3d.io.read_point_cloud(path)
    pts = np.asarray(pc.points, np.float32)
    cols = ((np.asarray(pc.colors) * 255.0).astype(np.uint8) if pc.has_colors()
            else np.full((len(pts), 3), 200, np.uint8))
    if len(pts) > MAX_POINTS:                       # stride-downsample for the spike
        step = int(np.ceil(len(pts) / MAX_POINTS))
        pts, cols = pts[::step], cols[::step]
    return np.ascontiguousarray(pts), np.ascontiguousarray(cols)


def _serve(webdir):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=webdir)
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd.server_address[1]


def main():
    from PySide6.QtCore import QUrl
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWidgets import QApplication, QMainWindow

    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PLY
    pts, cols = _load_cloud(path)
    med = np.median(pts, axis=0)
    extent = float(np.linalg.norm(pts.max(0) - pts.min(0))) or 1.0

    webdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_spike_web")
    os.makedirs(webdir, exist_ok=True)
    pts.tofile(os.path.join(webdir, "positions.bin"))
    cols.tofile(os.path.join(webdir, "colors.bin"))
    html = _HTML
    for tok, val in (("__N__", len(pts)), ("__CX__", med[0]), ("__CY__", med[1]),
                     ("__CZ__", med[2]), ("__BSIZE__", 6.0), ("__PTSIZE__", 2.0),
                     ("__CAMDIST__", extent * 0.9)):
        html = html.replace(tok, repr(float(val)) if isinstance(val, float) else str(val))
    with open(os.path.join(webdir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    port = _serve(webdir)

    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle(f"three.js gizmo spike — {os.path.basename(path)} · {len(pts):,} pts")
    win.resize(1180, 820)
    view = QWebEngineView()
    view.load(QUrl(f"http://127.0.0.1:{port}/index.html"))
    win.setCentralWidget(view)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
