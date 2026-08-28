import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// Must match the point buffer capacity implied by the server's per-frame
// downsampling (point_stride / conf_threshold) times the video's frame count.
const MAX_POINTS = 1_000_000;

const statusEl = document.getElementById("status");
const videoEl = document.getElementById("video");
const container = document.getElementById("cloud-container");

// ---- three.js scene -------------------------------------------------------
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x05070c);

const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 1000);
camera.position.set(0, -1.5, -3);
camera.up.set(0, -1, 0); // model world is OpenCV-style (y-down); flip "up" so orbit feels natural

const renderer = new THREE.WebGLRenderer({ antialias: true });
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.AxesHelper(0.5));

const positions = new Float32Array(MAX_POINTS * 3);
const colors = new Float32Array(MAX_POINTS * 3);
const geometry = new THREE.BufferGeometry();
geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
geometry.setDrawRange(0, 0);
const material = new THREE.PointsMaterial({ size: 0.015, vertexColors: true });
const pointCloud = new THREE.Points(geometry, material);
scene.add(pointCloud);

// Small gizmo marking the model's current estimated camera pose.
const cameraMarker = new THREE.AxesHelper(0.2);
scene.add(cameraMarker);

let writeCursor = 0;

function resize() {
  const { clientWidth, clientHeight } = container;
  if (clientWidth === 0 || clientHeight === 0) return;
  renderer.setSize(clientWidth, clientHeight);
  camera.aspect = clientWidth / clientHeight;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

function appendPoints(pointsF32, colorsU8) {
  const n = pointsF32.length / 3;
  if (n === 0) return;
  if (writeCursor + n > MAX_POINTS) {
    console.warn("Point buffer full; dropping incoming points");
    return;
  }
  positions.set(pointsF32, writeCursor * 3);
  for (let i = 0; i < n * 3; i++) {
    colors[writeCursor * 3 + i] = colorsU8[i] / 255;
  }
  writeCursor += n;
  geometry.setDrawRange(0, writeCursor);
  geometry.attributes.position.needsUpdate = true;
  geometry.attributes.color.needsUpdate = true;
}

function updateCameraPose(c2wFlat) {
  const m = new THREE.Matrix4();
  m.set(...c2wFlat); // row-major, matches THREE.Matrix4.set()'s argument order
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  m.decompose(position, quaternion, scale);
  cameraMarker.position.copy(position);
  cameraMarker.quaternion.copy(quaternion);
}

// ---- WebSocket wire protocol ------------------------------------------------
// Text (JSON) control frames: {"type": "start"|"busy"|"error"|"end", ...}
// Binary frames, one per reconstructed video frame:
//   uint32 frame_index, float32 timestamp_s, uint32 point_count,
//   16x float32 camera_to_world (row-major), point_count x float32[3] xyz,
//   point_count x uint8[3] rgb.
let frameCount = 0;
let lastStatsTime = performance.now();

function handleFrame(buffer) {
  const dv = new DataView(buffer);
  let offset = 0;
  const frameIndex = dv.getUint32(offset, true);
  offset += 4;
  const timestamp = dv.getFloat32(offset, true);
  offset += 4;
  const numPoints = dv.getUint32(offset, true);
  offset += 4;
  const c2w = new Float32Array(buffer, offset, 16);
  offset += 64;
  const points = new Float32Array(buffer, offset, numPoints * 3);
  offset += numPoints * 3 * 4;
  const colorBytes = new Uint8Array(buffer, offset, numPoints * 3);

  appendPoints(points, colorBytes);
  updateCameraPose(c2w);

  if (videoEl.readyState >= 1) {
    videoEl.currentTime = timestamp;
  }

  frameCount++;
  const now = performance.now();
  if (now - lastStatsTime > 500) {
    const fps = (frameCount / ((now - lastStatsTime) / 1000)).toFixed(1);
    statusEl.textContent = `frame ${frameIndex} · ${writeCursor.toLocaleString()} pts · ${fps} fps`;
    frameCount = 0;
    lastStatsTime = now;
  }
}

function handleControlMessage(msg) {
  if (msg.type === "start") {
    videoEl.src = msg.video_url;
    videoEl.pause();
    statusEl.textContent = "streaming…";
  } else if (msg.type === "busy") {
    statusEl.textContent = msg.message;
  } else if (msg.type === "error") {
    statusEl.textContent = `error: ${msg.message}`;
  } else if (msg.type === "end") {
    statusEl.textContent = `done — ${writeCursor.toLocaleString()} points total`;
  }
}

function connect() {
  const socket = new WebSocket(`ws://${location.host}/ws`);
  socket.binaryType = "arraybuffer";
  socket.onopen = () => {
    statusEl.textContent = "connected — waiting for the model…";
  };
  socket.onclose = () => {
    if (!statusEl.textContent.startsWith("done")) {
      statusEl.textContent = "disconnected";
    }
  };
  socket.onerror = () => {
    statusEl.textContent = "connection error";
  };
  socket.onmessage = (event) => {
    if (typeof event.data === "string") {
      handleControlMessage(JSON.parse(event.data));
    } else {
      handleFrame(event.data);
    }
  };
}

connect();
