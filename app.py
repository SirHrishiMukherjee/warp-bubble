# Simulon Research Institute — Exciton-Driven Warp Bubble UI
# Single-file Flask app that serves an interactive HTML dashboard.
# Controls: camber, AoA, wall thickness, bubble radii, ship speed, and exciton parameters.
# Plots: shape function f(x,y) and |\nabla f| heatmaps; reports energy cost with exciton coherence factor.
#
# Usage:
#   pip install flask numpy
#   python app.py
#   Visit http://127.0.0.1:5000/

from __future__ import annotations
import math
from typing import Tuple

import numpy as np
from flask import Flask, jsonify, request, Response

app = Flask(__name__)

# --------------------------- Core Model --------------------------- #

def naca_camber_line(x: np.ndarray, chord: float, m: float = 0.04, p: float = 0.4) -> np.ndarray:
    """NACA 4-digit camber line (scaled):
    m: maximum camber fraction of chord (0..0.1 typical)
    p: location of max camber (fraction of chord), default 0.4
    x: physical x (meters or arbitrary units), 0..chord
    returns y_c(x) with same units as chord
    """
    # Normalize to x/c in [0,1]
    s = np.clip(x / chord, 0.0, 1.0)
    yc = np.zeros_like(s)
    # piecewise definition
    mask1 = s < p
    mask2 = ~mask1
    if p <= 0 or p >= 1:
        # Degenerate; fall back to simple parabola shape
        return m * chord * (2 * s - s**2)
    yc[mask1] = m / (p**2) * (2 * p * s[mask1] - s[mask1] ** 2)
    yc[mask2] = m / ((1 - p) ** 2) * ((1 - 2 * p) + 2 * p * s[mask2] - s[mask2] ** 2)
    return yc * chord


def shape_function(
    X: np.ndarray,
    Y: np.ndarray,
    Rf: float,
    Rperp: float,
    sigma: float,
    alpha: float,
    theta_deg: float,
    camber_m: float,
    camber_p: float,
) -> np.ndarray:
    """Cambered anisotropic top-hat shape function f(x,y) in 2D.
    X, Y: grids (same shape)
    Rf: longitudinal radius (sets chord length ~ 2*Rf)
    Rperp: lateral radius
    sigma: wall sharpness (>0)
    alpha: camber amplitude multiplier (-1..1)
    theta_deg: angle of attack in degrees
    camber_m, camber_p: NACA camber params
    Returns f in [0,1]
    """
    # Rotate coordinates by AoA theta around origin
    th = math.radians(theta_deg)
    Xr = X * math.cos(th) + Y * math.sin(th)
    Yr = -X * math.sin(th) + Y * math.cos(th)

    # Map Xr to local chord coordinate [x0, x1]
    chord = 2.0 * Rf
    x0 = -Rf
    # Compute camber line C(x) along chord coordinates
    # Shift to [0, chord]
    xv = np.clip(Xr - x0, 0.0, chord)
    Cx = naca_camber_line(xv, chord=chord, m=camber_m, p=camber_p)

    # Camber displacement applied with amplitude alpha
    Xc = Xr - alpha * Cx  # fore-aft asymmetry

    # Elliptic distance-like function
    Phi = np.sqrt((Xc**2) / (Rf**2) + (Yr**2) / (Rperp**2))

    # Smooth top-hat via tanh walls around Phi=1
    f = 0.5 * (np.tanh(sigma * (Phi + 1.0)) - np.tanh(sigma * (Phi - 1.0)))
    return f


def exciton_coherence_factor(omega: float, phi: float, n_x: float) -> float:
    """Return xi in (0.2..1] as a reduction factor on energy cost.
    omega: normalized exciton drive frequency (0..2)
    phi: phase (radians)
    n_x: normalized exciton density (0..1)
    """
    # Gaussian resonance centered at 1 with width 0.25
    g = math.exp(-((omega - 1.0) ** 2) / (2 * 0.25 ** 2))
    # Coherence based on phase alignment (phi=0 best)
    coh = np.clip(n_x, 0.0, 1.0) * (1.0 + math.cos(phi)) / 2.0
    xi = 1.0 - 0.8 * coh * g
    # Clamp for stability
    return float(np.clip(xi, 0.2, 1.0))


def compute_energy(
    f: np.ndarray,
    dx: float,
    dy: float,
    vs: float,
    xi: float,
) -> Tuple[float, float]:
    """Compute mean gradient-squared energy with ship speed vs and factor xi.
    Returns (E_eff, wall_fraction)
    """
    # Gradients
    dfx, dfy = np.gradient(f, dx, dy, edge_order=2)
    grad2 = dfx**2 + dfy**2
    wall_mask = grad2 > (0.02 * grad2.max() + 1e-12)
    wall_fraction = wall_mask.mean()
    E = (vs**2) * float(grad2.mean()) * xi
    return E, float(wall_fraction)


# --------------------------- Web UI --------------------------- #

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Simulon — Exciton-Driven Warp Bubble UI</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    :root { --bg:#0b1120; --panel:#111827; --text:#e5e7eb; --accent:#60a5fa; }
    body { margin:0; font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial; background:var(--bg); color:var(--text); }
    header { padding:16px 20px; background:linear-gradient(90deg,#0b1120,#111827); border-bottom:1px solid #1f2937; }
    h1 { margin:0; font-weight:700; letter-spacing:0.4px; }
    .grid { display:grid; grid-template-columns: 340px 1fr; gap:16px; padding:16px; }
    .card { background:var(--panel); border:1px solid #1f2937; border-radius:16px; padding:16px; box-shadow:0 10px 30px rgba(0,0,0,0.25); }
    .section-title { font-size:14px; color:#9ca3af; margin:10px 0 8px; text-transform:uppercase; letter-spacing:1.2px; }
    .row { display:flex; align-items:center; gap:8px; margin:10px 0; }
    .row label { flex: 0 0 150px; font-size:14px; color:#cbd5e1; }
    input[type=range] { width:100%; }
    .val { width:72px; text-align:right; color:#93c5fd; font-weight:600; }
    .metrics { display:flex; gap:16px; flex-wrap:wrap; }
    .chip { background:#0b1229; border:1px solid #1f2937; padding:8px 12px; border-radius:12px; color:#bfdbfe; font-weight:600; }
    #plots { height: calc(100vh - 120px); display:grid; grid-template-rows: 1fr 1fr; gap:12px; }
    a { color: var(--accent); text-decoration:none; }
  </style>
</head>
<body>
  <header>
    <h1>Simulon Research — Exciton-Driven Warp Bubble Control</h1>
    <div class="metrics" id="metrics">
      <div class="chip">Energy E: <span id="E">—</span></div>
      <div class="chip">Coherence ξ: <span id="XI">—</span></div>
      <div class="chip">Wall fraction: <span id="WF">—</span></div>
    </div>
  </header>
  <div class="grid">
    <div class="card" id="controls">
      <div class="section-title">Bubble geometry</div>
      <div class="row"><label>Rf (longitudinal)</label><input type="range" id="Rf" min="0.6" max="2.0" step="0.02" value="1.2"/><div class="val" id="Rf_v"></div></div>
      <div class="row"><label>R⊥ (lateral)</label><input type="range" id="Rperp" min="0.6" max="2.0" step="0.02" value="1.0"/><div class="val" id="Rperp_v"></div></div>
      <div class="row"><label>σ (wall sharpness)</label><input type="range" id="sigma" min="2" max="30" step="1" value="12"/><div class="val" id="sigma_v"></div></div>
      <div class="row"><label>α (camber amp)</label><input type="range" id="alpha" min="-0.6" max="0.6" step="0.02" value="0.25"/><div class="val" id="alpha_v"></div></div>
      <div class="row"><label>AoA θ (deg)</label><input type="range" id="theta" min="-15" max="15" step="0.5" value="2"/><div class="val" id="theta_v"></div></div>
      <div class="row"><label>Camber m</label><input type="range" id="m" min="0" max="0.12" step="0.005" value="0.04"/><div class="val" id="m_v"></div></div>
      <div class="row"><label>Camber p</label><input type="range" id="p" min="0.1" max="0.9" step="0.02" value="0.4"/><div class="val" id="p_v"></div></div>

      <div class="section-title">Ship + exciton controls</div>
      <div class="row"><label>v_s (c = 1)</label><input type="range" id="vs" min="0" max="2.0" step="0.02" value="0.6"/><div class="val" id="vs_v"></div></div>
      <div class="row"><label>Exciton density n_x</label><input type="range" id="nx" min="0" max="1" step="0.02" value="0.7"/><div class="val" id="nx_v"></div></div>
      <div class="row"><label>ω (norm freq)</label><input type="range" id="omega" min="0" max="2" step="0.02" value="1.0"/><div class="val" id="omega_v"></div></div>
      <div class="row"><label>ϕ (phase, rad)</label><input type="range" id="phi" min="0" max="6.283" step="0.02" value="0.0"/><div class="val" id="phi_v"></div></div>
      <p style="font-size:12px;color:#9ca3af;line-height:1.4">Tip: For lower energy cost, align phase (ϕ≈0), set ω≈1, and increase n_x. Explore camber and AoA to mimic a supercritical airfoil front/long-tail profile.</p>
      <p style="font-size:12px;color:#9ca3af;">Source: <a href="/schema" target="_blank">Model equations</a></p>
    </div>
    <div class="card">
      <div id="plots">
        <div id="fplot"></div>
        <div id="gplot"></div>
      </div>
    </div>
  </div>

<script>
  const ids = ["Rf","Rperp","sigma","alpha","theta","m","p","vs","nx","omega","phi"];
  function val(id){return document.getElementById(id).value;}
  function setv(id,v){document.getElementById(id+"_v").textContent = (+v).toFixed(3)}
  ids.forEach(id=>{ document.getElementById(id).addEventListener('input', update); setv(id, val(id)); });

  let first = true; // to create Plotly layouts once

  async function update(){
    ids.forEach(id=>setv(id, val(id)));
    const params = new URLSearchParams({
      Rf: val('Rf'), Rperp: val('Rperp'), sigma: val('sigma'), alpha: val('alpha'), theta: val('theta'),
      m: val('m'), p: val('p'), vs: val('vs'), nx: val('nx'), omega: val('omega'), phi: val('phi')
    });
    const r = await fetch('/compute?'+params.toString());
    const d = await r.json();
    document.getElementById('E').textContent = d.E.toFixed(6);
    document.getElementById('XI').textContent = d.xi.toFixed(4);
    document.getElementById('WF').textContent = d.wall_fraction.toFixed(3);

    const dataF = [{
      z: d.f.z, x: d.f.x, y: d.f.y, type:'heatmap', hovertemplate:'x=%{x:.2f}<br>y=%{y:.2f}<br>f=%{z:.3f}<extra></extra>'
    }];
    const layoutF = {title:'Shape function f(x,y)', margin:{l:40,r:10,b:35,t:35}, xaxis:{title:'x'}, yaxis:{title:'y', scaleanchor:'x', scaleratio:1}};

    const dataG = [{
      z: d.grad.z, x: d.grad.x, y: d.grad.y, type:'heatmap', hovertemplate:'|∇f|=%{z:.4f}<extra></extra>'
    }];
    const layoutG = {title:'Gradient magnitude |∇f|', margin:{l:40,r:10,b:35,t:35}, xaxis:{title:'x'}, yaxis:{title:'y', scaleanchor:'x', scaleratio:1}};

    if(first){
      Plotly.newPlot('fplot', dataF, layoutF, {displayModeBar:false});
      Plotly.newPlot('gplot', dataG, layoutG, {displayModeBar:false});
      first = false;
    } else {
      Plotly.react('fplot', dataF, layoutF, {displayModeBar:false});
      Plotly.react('gplot', dataG, layoutG, {displayModeBar:false});
    }
  }
  update();
</script>
</body>
</html>
"""


@app.get("/")
def index() -> Response:
    return Response(INDEX_HTML, mimetype="text/html; charset=utf-8")


@app.get("/schema")
def schema() -> Response:
    txt = (
        "<pre style='color:#e5e7eb;background:#111827;padding:16px;border-radius:12px'>"
        "ds^2 = -c^2 dt^2 + (dx - v_s f dt)^2 + dy^2 + dz^2\n\n"
        "f(x,y) = 0.5[ tanh(σ(Φ+1)) - tanh(σ(Φ-1)) ]\n"
        "Φ(x,y) = sqrt( (x-α C(x))^2 / R_f^2 + y^2 / R_⊥^2 )\n\n"
        "NACA camber C(x) with m,p on chord 2R_f\n"
        "Energy E ∝ v_s^2 ⟨|∇f|^2⟩ · ξ(ω,φ)  (with ξ∈(0.2..1])\n"
        "</pre>"
    )
    return Response(txt, mimetype="text/html; charset=utf-8")


@app.get("/compute")
def compute():
    # Parameters with defaults
    Rf = float(request.args.get("Rf", 1.2))
    Rperp = float(request.args.get("Rperp", 1.0))
    sigma = float(request.args.get("sigma", 12.0))
    alpha = float(request.args.get("alpha", 0.25))
    theta = float(request.args.get("theta", 2.0))
    m = float(request.args.get("m", 0.04))
    p = float(request.args.get("p", 0.4))
    vs = float(request.args.get("vs", 0.6))
    nx = float(request.args.get("nx", 0.7))
    omega = float(request.args.get("omega", 1.0))
    phi = float(request.args.get("phi", 0.0))

    # Grid (square aspect to keep axes equal)
    # Domain spans ±1.2 radii
    Nx, Ny = 140, 120
    xlim = 1.25 * Rf
    ylim = 1.25 * Rperp
    x = np.linspace(-xlim, xlim, Nx)
    y = np.linspace(-ylim, ylim, Ny)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    X, Y = np.meshgrid(x, y)

    f = shape_function(X, Y, Rf, Rperp, sigma, alpha, theta, m, p)

    xi = exciton_coherence_factor(omega=omega, phi=phi, n_x=nx)
    E, wall_fraction = compute_energy(f, dx, dy, vs, xi)

    # Gradient magnitude for display
    dfx, dfy = np.gradient(f, dx, dy, edge_order=2)
    grad_mag = np.sqrt(dfx**2 + dfy**2)

    # Downsample for transport (keep 100x100 max)
    def pack_field(arr: np.ndarray):
        # convert to lists for JSON
        return [[float(v) for v in row] for row in arr]

    payload = {
        "E": float(E),
        "xi": float(xi),
        "wall_fraction": float(wall_fraction),
        "f": {"x": [float(v) for v in x], "y": [float(v) for v in y], "z": pack_field(f)},
        "grad": {"x": [float(v) for v in x], "y": [float(v) for v in y], "z": pack_field(grad_mag)},
    }
    return jsonify(payload)
