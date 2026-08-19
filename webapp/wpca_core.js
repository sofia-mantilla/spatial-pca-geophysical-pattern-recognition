/* wpca_core.js — JavaScript port of the wPCA pipeline, mirroring the repo's
 * paper/numpy_repro_concat.py (pure-numpy replication validated against the
 * published pipeline). Works in Node (validation harness) and the browser.
 *
 * Chain: crop/mask -> stride sliding windows (shared NaN mask across variables,
 * keep only all-finite windows) -> per-block standardized PCA scores (column
 * standardize ddof0, center, eigendecomposition of Zc^T Zc == full-SVD scores)
 * -> z_dep^2 block weights (alpha between variables) -> weighted distance to
 * the appended reference row -> top-N windows -> exact-geometry footprint
 * recovery (Sutherland-Hodgman rect clipping, union via cell decomposition),
 * hit = coverage >= MIN_COVER of a test-deposit footprint.
 */
"use strict";

/* ---------------------------------------------------------------- shapefile */
function readShpPolygons(buf) {
  // buf: ArrayBuffer or Node Buffer of a .shp file; returns [rings...] per record,
  // ring = [[x,y],...]
  const dv = new DataView(buf.buffer ? buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) : buf);
  const recs = [];
  let i = 100;
  while (i < dv.byteLength) {
    const clen = dv.getInt32(i + 4, false); i += 8;
    const start = i; i += clen * 2;
    if (dv.getInt32(start, true) !== 5) continue;
    const nparts = dv.getInt32(start + 36, true);
    const npts = dv.getInt32(start + 40, true);
    const parts = [];
    for (let p = 0; p < nparts; p++) parts.push(dv.getInt32(start + 44 + 4 * p, true));
    parts.push(npts);
    const p0 = start + 44 + 4 * nparts;
    const rings = [];
    for (let r = 0; r < nparts; r++) {
      const ring = [];
      for (let k = parts[r]; k < parts[r + 1]; k++) {
        ring.push([dv.getFloat64(p0 + 16 * k, true), dv.getFloat64(p0 + 16 * k + 8, true)]);
      }
      rings.push(ring);
    }
    recs.push(rings);
  }
  return recs;
}

/* -------------------------------------------- transverse mercator + Helmert */
const GRS67 = [6378160.0, 1 / 298.247167427];
const WGS84 = [6378137.0, 1 / 298.257223563];
const UTM = { k0: 0.9996, lon0: -51.0 * Math.PI / 180, fe: 500000.0, fn: 10000000.0 };
const HELMERT = [-66.87, 4.37, -38.52];

function tmConsts(a, f) {
  const n = f / (2 - f);
  const A = a / (1 + n) * (1 + n ** 2 / 4 + n ** 4 / 64 + n ** 6 / 256);
  const alpha = [
    n / 2 - 2 * n ** 2 / 3 + 5 * n ** 3 / 16 + 41 * n ** 4 / 180 - 127 * n ** 5 / 288 + 7891 * n ** 6 / 37800,
    13 * n ** 2 / 48 - 3 * n ** 3 / 5 + 557 * n ** 4 / 1440 + 281 * n ** 5 / 630 - 1983433 * n ** 6 / 1935360,
    61 * n ** 3 / 240 - 103 * n ** 4 / 140 + 15061 * n ** 5 / 26880 + 167603 * n ** 6 / 181440,
    49561 * n ** 4 / 161280 - 179 * n ** 5 / 168 + 6601661 * n ** 6 / 7257600,
    34729 * n ** 5 / 80640 - 3418889 * n ** 6 / 1995840,
    212378941 * n ** 6 / 319334400,
  ];
  const beta = [
    n / 2 - 2 * n ** 2 / 3 + 37 * n ** 3 / 96 - n ** 4 / 360 - 81 * n ** 5 / 512 + 96199 * n ** 6 / 604800,
    n ** 2 / 48 + n ** 3 / 15 - 437 * n ** 4 / 1440 + 46 * n ** 5 / 105 - 1118711 * n ** 6 / 3870720,
    17 * n ** 3 / 480 - 37 * n ** 4 / 840 - 209 * n ** 5 / 4480 + 5569 * n ** 6 / 90720,
    4397 * n ** 4 / 161280 - 11 * n ** 5 / 504 - 830251 * n ** 6 / 7257600,
    4583 * n ** 5 / 161280 - 108847 * n ** 6 / 3991680,
    20648693 * n ** 6 / 638668800,
  ];
  return { n, A, alpha, beta };
}

function tmInverse(x, y, a, f) {
  const { A, beta } = tmConsts(a, f);
  const e = Math.sqrt(f * (2 - f));
  const xi = (y - UTM.fn) / (UTM.k0 * A);
  const eta = (x - UTM.fe) / (UTM.k0 * A);
  let xiP = xi, etaP = eta;
  for (let j = 1; j <= 6; j++) {
    const b = beta[j - 1];
    xiP -= b * Math.sin(2 * j * xi) * Math.cosh(2 * j * eta);
    etaP -= b * Math.cos(2 * j * xi) * Math.sinh(2 * j * eta);
  }
  const chi = Math.asin(Math.sin(xiP) / Math.cosh(etaP));
  let lat = chi;
  for (let it = 0; it < 8; it++) {
    const s = Math.sinh(e * Math.atanh(e * Math.sin(lat)));
    lat = Math.asin(Math.tanh(Math.asinh(Math.tan(chi)) + Math.asinh(s)));
  }
  const lon = UTM.lon0 + Math.atan2(Math.sinh(etaP), Math.cos(xiP));
  return [lat, lon];
}

function tmForward(lat, lon, a, f) {
  const { A, alpha } = tmConsts(a, f);
  const e = Math.sqrt(f * (2 - f));
  const t = Math.sinh(Math.atanh(Math.sin(lat)) - e * Math.atanh(e * Math.sin(lat)));
  const xiP = Math.atan2(t, Math.cos(lon - UTM.lon0));
  const etaP = Math.atanh(Math.sin(lon - UTM.lon0) / Math.sqrt(1 + t * t));
  let xi = xiP, eta = etaP;
  for (let j = 1; j <= 6; j++) {
    const al = alpha[j - 1];
    xi += al * Math.sin(2 * j * xiP) * Math.cosh(2 * j * etaP);
    eta += al * Math.cos(2 * j * xiP) * Math.sinh(2 * j * etaP);
  }
  return [UTM.fe + UTM.k0 * A * eta, UTM.fn + UTM.k0 * A * xi];
}

function geodeticToGeocentric(lat, lon, a, f) {
  const e2 = f * (2 - f);
  const N = a / Math.sqrt(1 - e2 * Math.sin(lat) ** 2);
  return [N * Math.cos(lat) * Math.cos(lon), N * Math.cos(lat) * Math.sin(lon), N * (1 - e2) * Math.sin(lat)];
}

function geocentricToGeodetic(X, Y, Z, a, f) {
  const e2 = f * (2 - f);
  const lon = Math.atan2(Y, X);
  const p = Math.hypot(X, Y);
  let lat = Math.atan2(Z, p * (1 - e2));
  for (let it = 0; it < 10; it++) {
    const N = a / Math.sqrt(1 - e2 * Math.sin(lat) ** 2);
    lat = Math.atan2(Z + e2 * N * Math.sin(lat), p);
  }
  return [lat, lon];
}

function sad69ToWgs84Utm(x, y) {
  let [lat, lon] = tmInverse(x, y, GRS67[0], GRS67[1]);
  let [X, Y, Z] = geodeticToGeocentric(lat, lon, GRS67[0], GRS67[1]);
  X += HELMERT[0]; Y += HELMERT[1]; Z += HELMERT[2];
  [lat, lon] = geocentricToGeodetic(X, Y, Z, WGS84[0], WGS84[1]);
  return tmForward(lat, lon, WGS84[0], WGS84[1]);
}

/* ---------------------------------------------------------------- geometry */
function polyArea(pts) {
  let s = 0.0;
  for (let i = 0; i < pts.length; i++) {
    const [x1, y1] = pts[i];
    const [x2, y2] = pts[(i + 1) % pts.length];
    s += x1 * y2 - x2 * y1;
  }
  return 0.5 * s;
}

function clipRect(pts, xmin, xmax, ymin, ymax) {
  // Sutherland-Hodgman, mirrors numpy_repro_concat.clip_rect (prev = poly[i-1])
  function clipEdge(poly, inside, intersect) {
    const out = [];
    for (let i = 0; i < poly.length; i++) {
      const cur = poly[i], prev = poly[(i - 1 + poly.length) % poly.length];
      const ci = inside(cur), pi = inside(prev);
      if (ci) {
        if (!pi) out.push(intersect(prev, cur));
        out.push(cur);
      } else if (pi) out.push(intersect(prev, cur));
    }
    return out;
  }
  const ixX = x0 => (p, q) => { const t = (x0 - p[0]) / (q[0] - p[0]); return [x0, p[1] + t * (q[1] - p[1])]; };
  const ixY = y0 => (p, q) => { const t = (y0 - p[1]) / (q[1] - p[1]); return [p[0] + t * (q[0] - p[0]), y0]; };
  let poly = pts.slice();
  const edges = [
    [p => p[0] >= xmin, ixX(xmin)],
    [p => p[0] <= xmax, ixX(xmax)],
    [p => p[1] >= ymin, ixY(ymin)],
    [p => p[1] <= ymax, ixY(ymax)],
  ];
  for (const [inside, ix] of edges) {
    if (!poly.length) return [];
    poly = clipEdge(poly, inside, ix);
  }
  return poly;
}

function closedEq(a, b) { return a[0] === b[0] && a[1] === b[1]; }
function openRing(ring) { return closedEq(ring[0], ring[ring.length - 1]) ? ring.slice(0, -1) : ring; }

function ringsClipArea(rings, xmin, xmax, ymin, ymax) {
  let tot = 0.0;
  for (const ring of rings) {
    const r = openRing(ring);
    const aFull = polyArea(r);
    const c = clipRect(r, xmin, xmax, ymin, ymax);
    if (c.length >= 3) {
      const aClip = Math.abs(polyArea(c));
      tot += aFull < 0 ? aClip : -aClip;
    }
  }
  return Math.max(tot, 0.0);
}

function pointInRings(x, y, rings) {
  let inside = false;
  for (const ring of rings) {
    const r = openRing(ring);
    let j = r.length - 1;
    for (let i = 0; i < r.length; i++) {
      const [xi, yi] = r[i], [xj, yj] = r[j];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
      j = i;
    }
  }
  return inside;
}

/* ------------------------------------------------- symmetric eigensolver
 * JAMA-style tred2 + tql2 (public-domain algorithm). A: Float64Array n*n
 * row-major, SYMMETRIC; overwritten. Returns { d: eigenvalues ascending,
 * V: Float64Array n*n where column j (V[i*n+j]) is the j-th eigenvector }. */
function symEigen(A, n) {
  const V = A;                      // accumulate vectors in place
  const d = new Float64Array(n);
  const e = new Float64Array(n);
  for (let j = 0; j < n; j++) d[j] = V[(n - 1) * n + j];
  // tred2: Householder tridiagonalization
  for (let i = n - 1; i > 0; i--) {
    let scale = 0.0, h = 0.0;
    for (let k = 0; k < i; k++) scale += Math.abs(d[k]);
    if (scale === 0.0) {
      e[i] = d[i - 1];
      for (let j = 0; j < i; j++) { d[j] = V[(i - 1) * n + j]; V[i * n + j] = 0.0; V[j * n + i] = 0.0; }
    } else {
      for (let k = 0; k < i; k++) { d[k] /= scale; h += d[k] * d[k]; }
      let f = d[i - 1];
      let g = Math.sqrt(h);
      if (f > 0) g = -g;
      e[i] = scale * g;
      h -= f * g;
      d[i - 1] = f - g;
      for (let j = 0; j < i; j++) e[j] = 0.0;
      for (let j = 0; j < i; j++) {
        f = d[j];
        V[j * n + i] = f;
        g = e[j] + V[j * n + j] * f;
        for (let k = j + 1; k <= i - 1; k++) {
          g += V[k * n + j] * d[k];
          e[k] += V[k * n + j] * f;
        }
        e[j] = g;
      }
      f = 0.0;
      for (let j = 0; j < i; j++) { e[j] /= h; f += e[j] * d[j]; }
      const hh = f / (h + h);
      for (let j = 0; j < i; j++) e[j] -= hh * d[j];
      for (let j = 0; j < i; j++) {
        f = d[j];
        g = e[j];
        for (let k = j; k <= i - 1; k++) V[k * n + j] -= (f * e[k] + g * d[k]);
        d[j] = V[(i - 1) * n + j];
        V[i * n + j] = 0.0;
      }
    }
    d[i] = h;
  }
  // accumulate transformations
  for (let i = 0; i < n - 1; i++) {
    V[(n - 1) * n + i] = V[i * n + i];
    V[i * n + i] = 1.0;
    const h = d[i + 1];
    if (h !== 0.0) {
      for (let k = 0; k <= i; k++) d[k] = V[k * n + i + 1] / h;
      for (let j = 0; j <= i; j++) {
        let g = 0.0;
        for (let k = 0; k <= i; k++) g += V[k * n + i + 1] * V[k * n + j];
        for (let k = 0; k <= i; k++) V[k * n + j] -= g * d[k];
      }
    }
    for (let k = 0; k <= i; k++) V[k * n + i + 1] = 0.0;
  }
  for (let j = 0; j < n; j++) { d[j] = V[(n - 1) * n + j]; V[(n - 1) * n + j] = 0.0; }
  V[(n - 1) * n + (n - 1)] = 1.0;
  e[0] = 0.0;

  // tql2: implicit QL with shifts
  for (let i = 1; i < n; i++) e[i - 1] = e[i];
  e[n - 1] = 0.0;
  let f = 0.0, tst1 = 0.0;
  const eps = Math.pow(2.0, -52.0);
  for (let l = 0; l < n; l++) {
    tst1 = Math.max(tst1, Math.abs(d[l]) + Math.abs(e[l]));
    let m = l;
    while (m < n) {
      if (Math.abs(e[m]) <= eps * tst1) break;
      m++;
    }
    if (m > l) {
      let iter = 0;
      do {
        iter++;
        if (iter > 100) break;
        let g = d[l];
        let p = (d[l + 1] - g) / (2.0 * e[l]);
        let r = Math.hypot(p, 1.0);
        if (p < 0) r = -r;
        d[l] = e[l] / (p + r);
        d[l + 1] = e[l] * (p + r);
        const dl1 = d[l + 1];
        let h = g - d[l];
        for (let i = l + 2; i < n; i++) d[i] -= h;
        f += h;
        p = d[m];
        let c = 1.0, c2 = c, c3 = c;
        const el1 = e[l + 1];
        let s = 0.0, s2 = 0.0;
        for (let i = m - 1; i >= l; i--) {
          c3 = c2; c2 = c; s2 = s;
          g = c * e[i];
          h = c * p;
          r = Math.hypot(p, e[i]);
          e[i + 1] = s * r;
          s = e[i] / r;
          c = p / r;
          p = c * d[i] - s * g;
          d[i + 1] = h + s * (c * g + s * d[i]);
          for (let k = 0; k < n; k++) {
            h = V[k * n + i + 1];
            V[k * n + i + 1] = s * V[k * n + i] + c * h;
            V[k * n + i] = c * V[k * n + i] - s * h;
          }
        }
        p = -s * s2 * c3 * el1 * e[l] / dl1;
        e[l] = s * p;
        d[l] = c * p;
      } while (Math.abs(e[l]) > eps * tst1);
    }
    d[l] = d[l] + f;
    e[l] = 0.0;
  }
  // sort ascending (selection sort, swap vector columns)
  for (let i = 0; i < n - 1; i++) {
    let k = i; let p = d[i];
    for (let j = i + 1; j < n; j++) if (d[j] < p) { k = j; p = d[j]; }
    if (k !== i) {
      d[k] = d[i]; d[i] = p;
      for (let j = 0; j < n; j++) { p = V[j * n + i]; V[j * n + i] = V[j * n + k]; V[j * n + k] = p; }
    }
  }
  return { d, V };
}

/* --------------------------------------------------------------- PCA scores */
// X: Float64Array nRows x totalCols (row-major). Block = columns [c0, c1).
// Returns Float64Array nRows x k of standardized-PCA scores (== numpy
// block_scores: standardize ddof0, center, SVD, U*S[:, :k]).
function blockScores(X, nRows, totalCols, c0, c1, k) {
  const p = c1 - c0;
  k = Math.min(k, p, nRows);
  // column-major standardized/centered copy
  const cols = new Float64Array(p * nRows);
  for (let j = 0; j < p; j++) {
    let mu = 0.0;
    for (let r = 0; r < nRows; r++) mu += X[r * totalCols + c0 + j];
    mu /= nRows;
    let v = 0.0;
    for (let r = 0; r < nRows; r++) { const t = X[r * totalCols + c0 + j] - mu; v += t * t; }
    let sd = Math.sqrt(v / nRows);            // ddof=0
    if (sd === 0) sd = 1.0;
    const base = j * nRows;
    let m2 = 0.0;
    for (let r = 0; r < nRows; r++) { const z = (X[r * totalCols + c0 + j] - mu) / sd; cols[base + r] = z; m2 += z; }
    m2 /= nRows;                              // re-center exactly (Zc = Z - mean)
    for (let r = 0; r < nRows; r++) cols[base + r] -= m2;
  }
  // C = Zc^T Zc  (p x p)
  const C = new Float64Array(p * p);
  for (let i = 0; i < p; i++) {
    const bi = i * nRows;
    for (let j = i; j < p; j++) {
      const bj = j * nRows;
      let s = 0.0;
      for (let r = 0; r < nRows; r++) s += cols[bi + r] * cols[bj + r];
      C[i * p + j] = s; C[j * p + i] = s;
    }
  }
  const { d, V } = symEigen(C, p);           // ascending
  // top-k columns = last k, order descending eigenvalue
  const scores = new Float64Array(nRows * k);
  for (let comp = 0; comp < k; comp++) {
    const j = p - 1 - comp;                   // eigenvector column (descending)
    for (let c = 0; c < p; c++) {
      const v = V[c * p + j];
      if (v === 0) continue;
      const base = c * nRows;
      for (let r = 0; r < nRows; r++) scores[r * k + comp] += cols[base + r] * v;
    }
  }
  return scores;
}

function blkW(z) {
  const w = new Float64Array(z.length);
  let s = 0.0;
  for (let i = 0; i < z.length; i++) { w[i] = z[i] * z[i]; s += w[i]; }
  if (s > 0) { for (let i = 0; i < w.length; i++) w[i] /= s; }
  else { w.fill(1.0 / Math.max(z.length, 1)); }
  return w;
}

/* ------------------------------------------------------------ window matrix */
// fields: array of Float64Array H*W (NaN = missing), same grid. tpl: {rs,re,cs,ce}.
// Returns { X, nRows, npix, idx: Int32Array n*3 (rowPx, colPx, id), wh, ww }.
// X rows = windows (all-finite under the SHARED mask) + appended reference row.
function buildWindowMatrix(fields, H, W, tpl, strideY, strideX) {
  if (strideX === undefined) strideX = strideY;
  const { rs, re, cs, ce } = tpl;
  const wh = re - rs, ww = ce - cs;
  const nv = fields.length;
  // templates
  const tplRows = [];
  for (const f of fields) {
    const t = new Float64Array(wh * ww);
    for (let r = 0; r < wh; r++) for (let c = 0; c < ww; c++) {
      const v = f[(rs + r) * W + (cs + c)];
      if (Number.isNaN(v)) throw new Error("template contains NaNs");
      t[r * ww + c] = v;
    }
    tplRows.push(t);
  }
  // padded dims (pad with NaN so (pH - wh) % stride == 0)
  const padY = (strideY - ((H - wh) % strideY + strideY) % strideY) % strideY;
  const padX = (strideX - ((W - ww) % strideX + strideX) % strideX) % strideX;
  const pH = H + padY, pW = W + padX;
  const nrows = Math.floor((pH - wh) / strideY) + 1;
  const ncols = Math.floor((pW - ww) / strideX) + 1;
  // shared validity mask: a window is kept iff NO NaN in ANY variable
  const keepList = [];
  for (let wr = 0; wr < nrows; wr++) {
    const r0 = wr * strideY;
    for (let wc = 0; wc < ncols; wc++) {
      const c0 = wc * strideX;
      if (r0 + wh > H || c0 + ww > W) continue;      // padding region is NaN
      let ok = true;
      outer:
      for (const f of fields) {
        for (let r = 0; r < wh; r++) {
          const rowBase = (r0 + r) * W + c0;
          for (let c = 0; c < ww; c++) {
            if (Number.isNaN(f[rowBase + c])) { ok = false; break outer; }
          }
        }
      }
      if (ok) keepList.push([r0, c0, wr * ncols + wc]);
    }
  }
  const n = keepList.length;
  const npix = wh * ww;
  const totalCols = nv * npix;
  const nRows = n + 1;
  const X = new Float64Array(nRows * totalCols);
  for (let i = 0; i < n; i++) {
    const [r0, c0] = keepList[i];
    for (let v = 0; v < nv; v++) {
      const f = fields[v];
      const off = i * totalCols + v * npix;
      for (let r = 0; r < wh; r++) {
        const rowBase = (r0 + r) * W + c0;
        const dst = off + r * ww;
        for (let c = 0; c < ww; c++) X[dst + c] = f[rowBase + c];
      }
    }
  }
  for (let v = 0; v < nv; v++) X.set(tplRows[v], n * totalCols + v * npix);
  const idx = new Int32Array(n * 3);
  for (let i = 0; i < n; i++) { idx[3 * i] = keepList[i][0]; idx[3 * i + 1] = keepList[i][1]; idx[3 * i + 2] = keepList[i][2]; }
  return { X, nRows, npix, totalCols, idx, wh, ww };
}

/* ---------------------------------------------------------------- ranking */
// ks: per-variable k. alphas: per-variable weight (sum 1). Returns order
// (indices into rows, ascending distance; includes dep row somewhere).
function rankWindows(wm, ks, alphas) {
  const { X, nRows, npix, totalCols } = wm;
  const nv = ks.length;
  const dep = nRows - 1;
  const F = [];       // per-variable scores
  const Wt = [];      // per-variable weights (scaled by alpha)
  for (let v = 0; v < nv; v++) {
    const Z = blockScores(X, nRows, totalCols, v * npix, (v + 1) * npix, ks[v]);
    const k = Math.min(ks[v], npix, nRows);
    const zdep = Z.subarray(dep * k, dep * k + k);
    const w = blkW(zdep);
    for (let i = 0; i < k; i++) w[i] *= (nv === 1 ? 1.0 : alphas[v]);
    F.push({ Z, k });
    Wt.push(w);
  }
  const d = new Float64Array(nRows);
  for (let v = 0; v < nv; v++) {
    const { Z, k } = F[v];
    const w = Wt[v];
    const depBase = dep * k;
    for (let r = 0; r < nRows; r++) {
      let s = 0.0;
      const base = r * k;
      for (let j = 0; j < k; j++) {
        const t = Z[base + j] - Z[depBase + j];
        s += w[j] * t * t;
      }
      d[r] += s;
    }
  }
  for (let r = 0; r < nRows; r++) d[r] = Math.sqrt(d[r]);
  const order = Array.from({ length: nRows }, (_, i) => i);
  order.sort((a, b) => (d[a] - d[b]) || (a - b));
  return { order, d, dep };
}

/* -------------------------------------------------------------- validation */
// rects come from ranked windows in map coordinates (meters).
// testDeps: Map/obj of { key: rings } in the SAME map CRS. Mirrors
// footprint_recovery: exact geometry, union via cell decomposition,
// hit at coverage >= minCover, cum mean over test deposits.
function footprintRecovery(rects, testDeps, minCover) {
  const keys = Object.keys(testDeps);
  const depArea = {};
  for (const i of keys) {
    const ring = openRing(testDeps[i][0]);
    depArea[i] = Math.abs(polyArea(ring));
  }
  const coverCells = {};
  for (const i of keys) {
    const ring = openRing(testDeps[i][0]);
    const bx = ring.map(p => p[0]), by = ring.map(p => p[1]);
    const minbx = Math.min(...bx), maxbx = Math.max(...bx);
    const minby = Math.min(...by), maxby = Math.max(...by);
    const cand = [];
    for (let j = 0; j < rects.length; j++) {
      const [a, b, c, dd] = rects[j];
      if (!(b <= minbx || a >= maxbx || dd <= minby || c >= maxby)) cand.push(j);
    }
    const xsSet = new Set([minbx, maxbx]);
    const ysSet = new Set([minby, maxby]);
    for (const j of cand) { xsSet.add(rects[j][0]); xsSet.add(rects[j][1]); ysSet.add(rects[j][2]); ysSet.add(rects[j][3]); }
    const xs = [...xsSet].sort((a, b) => a - b);
    const ys = [...ysSet].sort((a, b) => a - b);
    const cells = [];
    for (let xi = 0; xi < xs.length - 1; xi++) {
      for (let yi = 0; yi < ys.length - 1; yi++) {
        const ar = ringsClipArea(testDeps[i], xs[xi], xs[xi + 1], ys[yi], ys[yi + 1]);
        if (ar > 1e-9) cells.push([xs[xi], xs[xi + 1], ys[yi], ys[yi + 1], ar, false]);
      }
    }
    coverCells[i] = cells;
  }
  const coveredArea = {}, coverage = {}, firstHit = {}, hitByRank = {}, overlapByRank = {};
  const hits = new Set();
  const cumTotal = [], cumMean = [];
  let totalArea = 0.0;
  for (const i of keys) totalArea += depArea[i];
  for (let rank = 1; rank <= rects.length; rank++) {
    const [xa, xb, ya, yb] = rects[rank - 1];
    const newly = [], overlapped = [];
    for (const i of keys) {
      const inter = ringsClipArea(testDeps[i], xa, xb, ya, yb);
      if (inter <= 1e-9) continue;
      overlapped.push(i);
      let add = 0.0;
      for (const cell of coverCells[i]) {
        if (cell[5]) continue;
        if (cell[0] >= xa - 1e-9 && cell[1] <= xb + 1e-9 && cell[2] >= ya - 1e-9 && cell[3] <= yb + 1e-9) {
          cell[5] = true;
          add += cell[4];
        }
      }
      coveredArea[i] = (coveredArea[i] || 0.0) + add;
      coverage[i] = coveredArea[i] / depArea[i];
      if (coverage[i] >= minCover && !hits.has(i)) { hits.add(i); firstHit[i] = rank; newly.push(i); }
    }
    if (newly.length) hitByRank[rank] = newly.slice().sort();
    if (overlapped.length) overlapByRank[rank] = overlapped.slice().sort();
    let tot = 0.0, mean = 0.0;
    for (const i of keys) { tot += coveredArea[i] || 0.0; mean += coverage[i] || 0.0; }
    cumTotal.push(tot / totalArea);
    cumMean.push(mean / keys.length);
  }
  return { cumTotal, cumMean, coverage, depArea, hitByRank, firstHit, overlapByRank };
}

function metricsOf(res, nTop) {
  const mf = res.cumMean;
  let s = 0.0;
  for (const v of mf) s += v;
  return {
    auc: s * (nTop / mf.length),
    end: mf[mf.length - 1] * 100,
    endTotal: res.cumTotal[res.cumTotal.length - 1] * 100,
    hits: Object.keys(res.hitByRank).reduce((n, r) => n + res.hitByRank[r].length, 0),
    hitRanks: res.hitByRank,
  };
}

/* exports (Node + browser global) */
const WPCA = {
  readShpPolygons, sad69ToWgs84Utm,
  polyArea, clipRect, ringsClipArea, pointInRings, openRing,
  symEigen, blockScores, blkW,
  buildWindowMatrix, rankWindows, footprintRecovery, metricsOf,
};
if (typeof module !== "undefined" && module.exports) module.exports = WPCA;
if (typeof window !== "undefined") window.WPCA = WPCA;
