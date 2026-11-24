/* Validate the JS wPCA port against the paper's replication gates.
 * Case 1 (uni TMI, ref Paulo Afonso dep 6, k=17, stride 8, top 250):
 *   end 47.04, AUC 57.9, 5 hits, hit ranks {5:[10],103:[4],126:[7],166:[0],220:[1]},
 *   3.093 times a random selection of matched area
 * Case 2 (TMI + U, ref Alemão dep 3, k TMI=2, k U=8, alpha TMI=0.541, each
 * variable's distance divided by its own median before the weighted average;
 * test set = deposits 1-6 of the twelve-deposit shapefile less Alemão):
 *   end 58.81, AUC 122.9, 3 hits, hit ranks {11:[1],17:[0],74:[4]},
 *   6.699 times a random selection of matched area
 * Run from the repo root or from webapp/. DATA defaults to ../data relative to
 * this file and can be overridden with the WPCA_DATA environment variable.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const W = require("./wpca_core.js");

const DATA = process.env.WPCA_DATA || path.join(__dirname, "..", "data");
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

// Ratio to a random selection covering the same area as the top nTop windows
// (paper/run_all_twelve_case2.py): ranking recovery over the exact expected
// random recovery at matched area.
function matchedRatio(m, wm, order, dep, H, Wd, ox, oy, test, nTop) {
  const n = wm.nRows - 1;
  const cnt = new Int32Array(H * Wd);
  const allRects = [];
  for (let w = 0; w < n; w++) {
    const r = wm.idx[3 * w], c = wm.idx[3 * w + 1];
    for (let j = 0; j < wm.wh; j++) for (let i = 0; i < wm.ww; i++) cnt[(r + j) * Wd + c + i]++;
    const x1 = ox + c * PX, y2 = oy - r * PX;
    allRects.push([x1, x1 + wm.ww * PX, y2 - wm.wh * PX, y2]);
  }
  const freq = new Map();
  for (const v of cnt) if (v > 0) freq.set(v, (freq.get(v) || 0) + 1);
  const cover = new Uint8Array(H * Wd);
  let union = 0, k = 0;
  for (const o of order) {
    if (o === dep) continue;
    if (k++ >= nTop) break;
    const r = wm.idx[3 * o], c = wm.idx[3 * o + 1];
    for (let j = 0; j < wm.wh; j++) for (let i = 0; i < wm.ww; i++) {
      const p = (r + j) * Wd + c + i;
      if (!cover[p]) { cover[p] = 1; union++; }
    }
  }
  const { tStar, expected } = W.randomAtMatchedArea(allRects, test, n, freq, union);
  return { ratio: m.end / 100 / expected, expected, tStar, union };
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
  const q = matchedRatio(m, wm, order, dep, g.H, g.W, g.ox, g.oy, test, N_TOP);
  console.log(`case1 k=17: end ${m.end.toFixed(2)}  AUC ${m.auc.toFixed(1)}  hits ${m.hits}  ranks ${fmtRanks(m.hitRanks)}`);
  console.log(`  vs random of matched area: ${q.ratio.toFixed(3)}x  (random ${(100 * q.expected).toFixed(2)}%, t* ${q.tStar.toFixed(2)}, union ${q.union} cells)`);
  console.log(`  gates:    end 47.04  AUC 57.9  hits 5  ranks 5:[10] 103:[4] 126:[7] 166:[0] 220:[1]  ratio 3.093x (random 15.21%)`);
  console.log(`  (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
  const pass = Math.abs(m.end - 47.04) < 0.02 && Math.abs(m.auc - 57.9) < 0.1 && m.hits === 5
    && fmtRanks(m.hitRanks) === "5:[10] 103:[4] 126:[7] 166:[0] 220:[1]"
    && Math.abs(q.ratio - 3.0931) < 0.002;
  console.log(pass ? "CASE 1 GATES: PASS" : "CASE 1 GATES: FAIL");
  return pass;
}

function runCase2(g) {
  const t0 = Date.now();
  // Records 1-5 of the twelve-deposit shapefile are identical to the five of
  // Prospect_in Carajas_multi.shp, so the Alemão window is the same from either.
  const shp12 = path.join(DATA, "Carajas_Brazil_Univariate_TMI/Prospect_in Carajas_v2.shp");
  const depsRaw = readShp(shp12);
  const tpl = templateRect(depsRaw[3 - 1], g.ox, g.oy, g.H, g.W);
  const wm = W.buildWindowMatrix([g.tmi, g.u], g.H, g.W, tpl, STRIDE);
  console.log(`case2: n windows = ${wm.nRows - 1}, window = (${wm.wh}, ${wm.ww})`);
  const { order, dep } = W.rankWindows(wm, [2, 8], [0.541, 0.459]);
  const rects = rectsFromOrder(order, dep, wm.idx, wm.wh, wm.ww, g.ox, g.oy, N_TOP);
  const depsW = wgs84Deps(shp12).slice(0, 6);
  const test = {};
  depsW.forEach((rings, i) => { if (i !== 3 - 1) test[i] = rings; });
  const m = W.metricsOf(W.footprintRecovery(rects, test, MIN_COVER), N_TOP);
  const q = matchedRatio(m, wm, order, dep, g.H, g.W, g.ox, g.oy, test, N_TOP);
  console.log(`case2: end ${m.end.toFixed(2)}  AUC ${m.auc.toFixed(1)}  hits ${m.hits}  ranks ${fmtRanks(m.hitRanks)}`);
  console.log(`  vs random of matched area: ${q.ratio.toFixed(3)}x  (random ${(100 * q.expected).toFixed(2)}%, t* ${q.tStar.toFixed(2)}, union ${q.union} cells)`);
  console.log(`  gates: end 58.81  AUC 122.9  hits 3  ranks 11:[1] 17:[0] 74:[4]  ratio 6.699x (random 8.78%)`);
  console.log(`  (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
  const pass = Math.abs(m.end - 58.809) < 0.02 && Math.abs(m.auc - 122.93) < 0.1 && m.hits === 3
    && fmtRanks(m.hitRanks) === "11:[1] 17:[0] 74:[4]"
    && Math.abs(q.ratio - 6.6985) < 0.002;
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
