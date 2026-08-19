/* Validate the JS wPCA port against the paper's replication gates.
 * Case 1 (uni TMI, ref Paulo Afonso dep 6, k=17, stride 8, top 250):
 *   end 47.04, AUC 57.9, 5 hits, hit ranks {5:[10],103:[4],126:[7],166:[0],220:[1]}
 * Case 2 (concat, ref Alemão dep 3 of multi shp, k TMI=2, k U=8, alpha=0.503):
 *   end 63.75, AUC 106.1, 3 hits, hit ranks {10:[0],142:[1],246:[4]}
 */
"use strict";
const fs = require("fs");
const path = require("path");
const W = require("./wpca_core.js");

const DATA = "/home/user/repro/data";
const NL = 3527, NC = 2485, PX = 200.0;
const X0 = 279900.0, Y0 = 9656100.0, NODATA = -99999.0;
const STRIDE = 8, N_TOP = 250, MIN_COVER = 0.5;

function readShp(p) { return W.readShpPolygons(fs.readFileSync(p)); }

function loadCropped() {
  const demo = readShp(path.join(DATA, "Carajas_Brazil_Univariate_TMI/Demo_area_polygon.shp"))[0];
  const xs = demo[0].map(p => p[0]), ys = demo[0].map(p => p[1]);
  const c0 = Math.floor((Math.min(...xs) - X0) / PX), c1 = Math.ceil((Math.max(...xs) - X0) / PX);
  const r0 = Math.floor((Y0 - Math.max(...ys)) / PX), r1 = Math.ceil((Y0 - Math.min(...ys)) / PX);
  const H = r1 - r0, Wd = c1 - c0;
  function rd(p) {
    const buf = fs.readFileSync(p);
    const full = new Float32Array(buf.buffer, buf.byteOffset, NL * NC);
    const a = new Float64Array(H * Wd);
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < Wd; c++) {
        const v = full[(r0 + r) * NC + (c0 + c)];
        a[r * Wd + c] = v === NODATA ? NaN : v;
      }
    }
    return a;
  }
  const tmi = rd(path.join(DATA, "Carajas_Brazil_Univariate_TMI/1097_1125_1129_TMI_merged"));
  const u = rd(path.join(DATA, "Carajas_Brazil_Multivariate_TMI_U/1097_1125_1129_RAD_eU_merged"));
  const ox = X0 + c0 * PX, oy = Y0 - r0 * PX;
  // mask cells whose centers fall outside the demo polygon
  for (let r = 0; r < H; r++) {
    const cy = oy - (r + 0.5) * PX;
    for (let c = 0; c < Wd; c++) {
      const cx = ox + (c + 0.5) * PX;
      if (!W.pointInRings(cx, cy, demo)) { tmi[r * Wd + c] = NaN; u[r * Wd + c] = NaN; }
    }
  }
  return { tmi, u, H, W: Wd, ox, oy };
}

function templateRect(rings, ox, oy, H, Wd) {
  const xs = [], ys = [];
  for (const r of rings) for (const p of r) { xs.push(p[0]); ys.push(p[1]); }
  let cs = Math.floor((Math.min(...xs) - ox) / PX), ce = Math.ceil((Math.max(...xs) - ox) / PX);
  let rs = Math.floor((oy - Math.max(...ys)) / PX), re = Math.ceil((oy - Math.min(...ys)) / PX);
  cs = Math.max(cs, 0); ce = Math.min(ce, Wd); rs = Math.max(rs, 0); re = Math.min(re, H);
  return { rs, re, cs, ce };
}

function rectsFromOrder(order, dep, idx, wh, ww, ox, oy, nTop) {
  const rects = [];
  for (const o of order) {
    if (o === dep) continue;
    if (rects.length >= nTop) break;
    const r = idx[3 * o], c = idx[3 * o + 1];
    const x1 = ox + c * PX, y2 = oy - r * PX;
    rects.push([x1, x1 + ww * PX, y2 - wh * PX, y2]);
  }
  return rects;
}

function wgs84Deps(shpPath) {
  return readShp(shpPath).map(rings => rings.map(ring => ring.map(([x, y]) => W.sad69ToWgs84Utm(x, y))));
}

function fmtRanks(hitByRank) {
  return Object.keys(hitByRank).sort((a, b) => a - b).map(r => `${r}:[${hitByRank[r]}]`).join(" ");
}

function runCase1(g) {
  const t0 = Date.now();
  const depsRaw = readShp(path.join(DATA, "Carajas_Brazil_Univariate_TMI/Prospect_in Carajas_v2.shp"));
  const tpl = templateRect(depsRaw[6 - 1], g.ox, g.oy, g.H, g.W);
  const wm = W.buildWindowMatrix([g.tmi], g.H, g.W, tpl, STRIDE);
  console.log(`case1: n windows = ${wm.nRows - 1}, window = (${wm.wh}, ${wm.ww})  [expect 13031, (28, 46)]`);
  const { order, dep } = W.rankWindows(wm, [17], [1.0]);
  const rects = rectsFromOrder(order, dep, wm.idx, wm.wh, wm.ww, g.ox, g.oy, N_TOP);
  const depsW = wgs84Deps(path.join(DATA, "Carajas_Brazil_Univariate_TMI/Prospect_in Carajas_v2.shp"));
  const test = {};
  depsW.forEach((rings, i) => { if (i !== 6 - 1) test[i] = rings; });
  const m = W.metricsOf(W.footprintRecovery(rects, test, MIN_COVER), N_TOP);
  console.log(`case1 k=17: end ${m.end.toFixed(2)}  AUC ${m.auc.toFixed(1)}  hits ${m.hits}  ranks ${fmtRanks(m.hitRanks)}`);
  console.log(`  gates:    end 47.04  AUC 57.9  hits 5  ranks 5:[10] 103:[4] 126:[7] 166:[0] 220:[1]`);
  console.log(`  (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
  const pass = Math.abs(m.end - 47.04) < 0.02 && Math.abs(m.auc - 57.9) < 0.1 && m.hits === 5
    && fmtRanks(m.hitRanks) === "5:[10] 103:[4] 126:[7] 166:[0] 220:[1]";
  console.log(pass ? "CASE 1 GATES: PASS" : "CASE 1 GATES: FAIL");
  return pass;
}

function runCase2(g) {
  const t0 = Date.now();
  const depsRaw = readShp(path.join(DATA, "Carajas_Brazil_Multivariate_TMI_U/Prospect_in Carajas_multi.shp"));
  const tpl = templateRect(depsRaw[3 - 1], g.ox, g.oy, g.H, g.W);
  const wm = W.buildWindowMatrix([g.tmi, g.u], g.H, g.W, tpl, STRIDE);
  console.log(`case2: n windows = ${wm.nRows - 1}, window = (${wm.wh}, ${wm.ww})`);
  const { order, dep } = W.rankWindows(wm, [2, 8], [0.503, 0.497]);
  const rects = rectsFromOrder(order, dep, wm.idx, wm.wh, wm.ww, g.ox, g.oy, N_TOP);
  const depsW = wgs84Deps(path.join(DATA, "Carajas_Brazil_Multivariate_TMI_U/Prospect_in Carajas_multi.shp"));
  const test = {};
  depsW.forEach((rings, i) => { if (i !== 3 - 1) test[i] = rings; });
  const m = W.metricsOf(W.footprintRecovery(rects, test, MIN_COVER), N_TOP);
  console.log(`case2: end ${m.end.toFixed(2)}  AUC ${m.auc.toFixed(1)}  hits ${m.hits}  ranks ${fmtRanks(m.hitRanks)}`);
  console.log(`  gates: end 63.75  AUC 106.1  hits 3  ranks 10:[0] 142:[1] 246:[4]`);
  console.log(`  (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
  const pass = Math.abs(m.end - 63.747) < 0.02 && Math.abs(m.auc - 106.15) < 0.1 && m.hits === 3
    && fmtRanks(m.hitRanks) === "10:[0] 142:[1] 246:[4]";
  console.log(pass ? "CASE 2 GATES: PASS" : "CASE 2 GATES: FAIL");
  return pass;
}

const g = loadCropped();
console.log(`cropped grid: ${g.H} x ${g.W}, origin (${g.ox}, ${g.oy})`);
const which = process.argv[2] || "both";
let ok = true;
if (which === "1" || which === "both") ok = runCase1(g) && ok;
if (which === "2" || which === "both") ok = runCase2(g) && ok;
process.exit(ok ? 0 : 1);
