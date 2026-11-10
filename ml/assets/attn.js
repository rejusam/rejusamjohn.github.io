// Attention and transformer numerics for the ML Explorer. Pure functions; the
// trainer follows the shape and rules of nn.js trainer().
//
// Architecture (pre-LN, the order used by most current transformers):
//   x_t = E[id_t] + P[t]                                     token + learned position
//   per block:  x <- x + Wo · MHA(LN1(x))                     attention sublayer
//               x <- x + W2 ReLU(W1 LN2(x) + c1) + c2         only when mlp > 0
//   after the blocks: y = LNf(x); output Wout y + bout
//     'softmax': one logit per vocabulary item at every position (cross-entropy, natural log)
//     'logit':   one logit per position for a binary target (binary cross-entropy)
// With ln: false every LN is the identity and has no parameters.
//
// MHA: for head h, q = Wq_h u, k = Wk_h u, v = Wv_h u (each dk wide, no biases), scores
// q_t · k_s / sqrt(dk), softmax over s; with causal: true the positions s > t are left out
// of the softmax altogether, so their weights are exactly 0. Head outputs are
// concatenated (width heads·dk) and mapped back to d by Wo.
//
// LN uses the population variance (divide by d) and eps = 1e-5:
//   LN(x) = gamma ⊙ (x − mean) / sqrt(var + eps) + beta.
//
// Parameters live in one Float64Array `theta`, in this order (matrices row-major, rows =
// outputs): E (vocab×d), P (len×d); per block g1, b1 (if ln), Wq, Wk, Wv (heads·dk × d),
// Wo (d × heads·dk), then if mlp > 0: g2, b2 (if ln), W1 (mlp×d), c1, W2 (d×mlp), c2;
// then gf, bf (if ln), Wout (out×d), bout. The model exposes each block as a view.
import {mulberry32, randn} from './maths.js';
import {bceLogit, optInit, optStep} from './nn.js';

const EPS = 1e-5;
// E and P are drawn N(0, 1) and scaled by this.
const EMB_SCALE = 0.5;

function normCfg(cfg) {
  const heads = cfg.heads ?? 1;
  const dk = cfg.dk ?? cfg.d / heads;
  if (!Number.isInteger(dk) || dk < 1) throw new Error('dk must be a positive integer');
  return {vocab: cfg.vocab, len: cfg.len, d: cfg.d, heads, dk, causal: cfg.causal ?? false,
    ln: cfg.ln ?? true, mlp: cfg.mlp ?? 0, blocks: cfg.blocks ?? 1,
    out: cfg.out ?? 'softmax', seed: cfg.seed};
}

const outWidth = cfg => (cfg.out === 'logit' ? 1 : cfg.vocab);

// [name, length, kind, fanIn]; kind: 'emb' N(0,1)·EMB_SCALE, 'w' N(0, 1/fanIn), 'one', 'zero'.
function layout(cfg) {
  const {vocab, len, d, heads, dk, ln, mlp, blocks} = cfg;
  const D = heads * dk;
  const L = [['E', vocab * d, 'emb'], ['P', len * d, 'emb']];
  for (let b = 0; b < blocks; b++) {
    if (ln) L.push([`${b}.g1`, d, 'one'], [`${b}.b1`, d, 'zero']);
    L.push([`${b}.Wq`, D * d, 'w', d], [`${b}.Wk`, D * d, 'w', d], [`${b}.Wv`, D * d, 'w', d],
      [`${b}.Wo`, d * D, 'w', D]);
    if (mlp > 0) {
      if (ln) L.push([`${b}.g2`, d, 'one'], [`${b}.b2`, d, 'zero']);
      L.push([`${b}.W1`, mlp * d, 'w', d], [`${b}.c1`, mlp, 'zero'],
        [`${b}.W2`, d * mlp, 'w', mlp], [`${b}.c2`, d, 'zero']);
    }
  }
  if (ln) L.push(['gf', d, 'one'], ['bf', d, 'zero']);
  L.push(['Wout', outWidth(cfg) * d, 'w', d], ['bout', outWidth(cfg), 'zero']);
  return L;
}

const sizeOf = cfg => layout(cfg).reduce((s, e) => s + e[1], 0);

// Named views into buf: {theta, E, P, blocks: [{g1, b1, Wq, ...}], gf, bf, Wout, bout}.
function views(cfg, buf) {
  const o = {theta: buf, blocks: Array.from({length: cfg.blocks}, () => ({}))};
  let p = 0;
  for (const [name, n] of layout(cfg)) {
    const v = buf.subarray(p, p + n);
    p += n;
    const dot = name.indexOf('.');
    if (dot < 0) o[name] = v;
    else o.blocks[+name.slice(0, dot)][name.slice(dot + 1)] = v;
  }
  return o;
}

const build = (cfg, theta) => ({cfg, ...views(cfg, theta)});

// One mulberry32(seed) stream, drawn in the theta order above (LN and biases take no draws).
export function attnInit(cfg) {
  const c = normCfg(cfg);
  const rng = mulberry32(c.seed);
  const theta = new Float64Array(sizeOf(c));
  let p = 0;
  for (const [, n, kind, fanIn] of layout(c)) {
    const s = kind === 'emb' ? EMB_SCALE : 1 / Math.sqrt(fanIn);
    for (let i = 0; i < n; i++) {
      theta[p + i] = kind === 'one' ? 1 : kind === 'zero' ? 0 : s * randn(rng);
    }
    p += n;
  }
  return build(c, theta);
}

export const attnFlatten = model => Array.from(model.theta);
export const attnUnflatten = (model, v) => build(model.cfg, Float64Array.from(v));
export const attnParamCount = model => model.theta.length;

// The scaled score q · k / sqrt(dk) for one query and one key.
export function scaledDot(q, k) {
  let s = 0;
  for (let j = 0; j < q.length; j++) s += q[j] * k[j];
  return s / Math.sqrt(q.length);
}

// ---- layer norm -------------------------------------------------------------------------

// One vector. Population variance (1/d). inv = 1 / sqrt(var + eps).
export function layerNorm(x, gamma, beta, eps = EPS) {
  const d = x.length;
  let mu = 0;
  for (let j = 0; j < d; j++) mu += x[j];
  mu /= d;
  let v = 0;
  for (let j = 0; j < d; j++) v += (x[j] - mu) * (x[j] - mu);
  const inv = 1 / Math.sqrt(v / d + eps);
  return {y: Array.from(x, (xj, j) => gamma[j] * (xj - mu) * inv + beta[j]), mu, inv};
}

// Every row of a T×d buffer.
function lnRows(x, T, d, g, b, y, mu, inv) {
  for (let t = 0; t < T; t++) {
    const o = t * d;
    let m = 0;
    for (let j = 0; j < d; j++) m += x[o + j];
    m /= d;
    let v = 0;
    for (let j = 0; j < d; j++) { const c = x[o + j] - m; v += c * c; }
    const iv = 1 / Math.sqrt(v / d + EPS);
    mu[t] = m; inv[t] = iv;
    for (let j = 0; j < d; j++) y[o + j] = g[j] * (x[o + j] - m) * iv + b[j];
  }
}

// dx += inv · (dxhat − mean(dxhat) − xhat · mean(dxhat · xhat)), dxhat = dy ⊙ gamma.
function lnRowsBack(dy, x, T, d, g, mu, inv, dx, dg, db) {
  for (let t = 0; t < T; t++) {
    const o = t * d, iv = inv[t], m = mu[t];
    let s1 = 0, s2 = 0;
    for (let j = 0; j < d; j++) {
      const xh = (x[o + j] - m) * iv, dxh = dy[o + j] * g[j];
      dg[j] += dy[o + j] * xh;
      db[j] += dy[o + j];
      s1 += dxh;
      s2 += dxh * xh;
    }
    s1 /= d; s2 /= d;
    for (let j = 0; j < d; j++) {
      const xh = (x[o + j] - m) * iv;
      dx[o + j] += iv * (dy[o + j] * g[j] - s1 - xh * s2);
    }
  }
}

// ---- dense layers on T rows ---------------------------------------------------------------

// Y[t] = W X[t] (+ bias). W is nout×nin.
function linear(X, T, nin, W, nout, bias, Y) {
  for (let t = 0; t < T; t++) {
    const xo = t * nin, yo = t * nout;
    for (let o = 0; o < nout; o++) {
      const wo = o * nin;
      let s = bias ? bias[o] : 0;
      for (let i = 0; i < nin; i++) s += W[wo + i] * X[xo + i];
      Y[yo + o] = s;
    }
  }
}

// dW += dY^T X, dB += Σ_t dY, dX += dY W (each optional).
function linearBack(dY, X, T, nin, W, nout, dW, dB, dX) {
  for (let t = 0; t < T; t++) {
    const xo = t * nin, yo = t * nout;
    for (let o = 0; o < nout; o++) {
      const g = dY[yo + o];
      if (g === 0) continue;
      const wo = o * nin;
      if (dB) dB[o] += g;
      for (let i = 0; i < nin; i++) {
        dW[wo + i] += g * X[xo + i];
        if (dX) dX[xo + i] += g * W[wo + i];
      }
    }
  }
}

// ---- forward ------------------------------------------------------------------------------

function forward(model, ids) {
  const {cfg} = model;
  const {d, heads: H, dk, causal, ln, mlp: M} = cfg;
  const T = ids.length, D = H * dk, rs = 1 / Math.sqrt(dk);
  if (T > cfg.len) throw new Error('sequence longer than len');
  let x = new Float64Array(T * d);
  for (let t = 0; t < T; t++) {
    for (let j = 0; j < d; j++) x[t * d + j] = model.E[ids[t] * d + j] + model.P[t * d + j];
  }
  const x0 = x;
  const blocks = [];
  for (const p of model.blocks) {
    const c = {xin: x};
    let u = x;
    if (ln) {
      u = new Float64Array(T * d); c.mu1 = new Float64Array(T); c.inv1 = new Float64Array(T);
      lnRows(x, T, d, p.g1, p.b1, u, c.mu1, c.inv1);
    }
    c.u = u;
    const Q = new Float64Array(T * D), K = new Float64Array(T * D), V = new Float64Array(T * D);
    linear(u, T, d, p.Wq, D, null, Q);
    linear(u, T, d, p.Wk, D, null, K);
    linear(u, T, d, p.Wv, D, null, V);
    const A = new Float64Array(H * T * T), O = new Float64Array(T * D);
    for (let h = 0; h < H; h++) {
      const ho = h * dk;
      for (let t = 0; t < T; t++) {
        const row = (h * T + t) * T, smax = causal ? t : T - 1;
        let m = -Infinity;
        for (let s = 0; s <= smax; s++) {
          let sc = 0;
          for (let j = 0; j < dk; j++) sc += Q[t * D + ho + j] * K[s * D + ho + j];
          sc *= rs;
          A[row + s] = sc;
          if (sc > m) m = sc;
        }
        let z = 0;
        for (let s = 0; s <= smax; s++) { const e = Math.exp(A[row + s] - m); A[row + s] = e; z += e; }
        for (let s = 0; s <= smax; s++) {
          const a = A[row + s] / z;
          A[row + s] = a;
          for (let j = 0; j < dk; j++) O[t * D + ho + j] += a * V[s * D + ho + j];
        }
      }
    }
    Object.assign(c, {Q, K, V, A, O});
    const x1 = new Float64Array(T * d);
    linear(O, T, D, p.Wo, d, null, x1);
    for (let i = 0; i < T * d; i++) x1[i] += x[i];
    c.x1 = x1;
    x = x1;
    if (M > 0) {
      let w = x1;
      if (ln) {
        w = new Float64Array(T * d); c.mu2 = new Float64Array(T); c.inv2 = new Float64Array(T);
        lnRows(x1, T, d, p.g2, p.b2, w, c.mu2, c.inv2);
      }
      c.w = w;
      const hpre = new Float64Array(T * M), hr = new Float64Array(T * M);
      linear(w, T, d, p.W1, M, p.c1, hpre);
      for (let i = 0; i < T * M; i++) hr[i] = hpre[i] > 0 ? hpre[i] : 0;
      const x2 = new Float64Array(T * d);
      linear(hr, T, M, p.W2, d, p.c2, x2);
      for (let i = 0; i < T * d; i++) x2[i] += x1[i];
      Object.assign(c, {hpre, hr});
      x = x2;
    }
    blocks.push(c);
  }
  const cache = {T, ids, x0, blocks, xf: x};
  let y = x;
  if (ln) {
    y = new Float64Array(T * d); cache.muf = new Float64Array(T); cache.invf = new Float64Array(T);
    lnRows(x, T, d, model.gf, model.bf, y, cache.muf, cache.invf);
  }
  cache.y = y;
  const W = outWidth(cfg);
  cache.logits = new Float64Array(T * W);
  linear(y, T, d, model.Wout, W, model.bout, cache.logits);
  return cache;
}

// logits: 'softmax' → T arrays of vocab logits; 'logit' → T numbers.
// A[block][head][t][s]; cache holds every intermediate (Float64Array, T rows).
export function attnForward(model, ids) {
  const c = forward(model, ids);
  const {T} = c, W = outWidth(model.cfg), H = model.cfg.heads;
  const logits = model.cfg.out === 'logit'
    ? Array.from(c.logits)
    : Array.from({length: T}, (_, t) => Array.from(c.logits.subarray(t * W, (t + 1) * W)));
  const A = c.blocks.map(b => Array.from({length: H}, (_, h) =>
    Array.from({length: T}, (_, t) => Array.from(b.A.subarray((h * T + t) * T, (h * T + t + 1) * T)))));
  return {logits, A, cache: c};
}

// ---- loss ---------------------------------------------------------------------------------

const sigmoid = z => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));

// Mean loss over positions with mask 1 (all positions when no mask). With dl, writes
// d loss / d logits · scale into it. No counted positions: loss 0 and no gradient.
function lossFromLogits(cfg, logits, T, targets, mask, dl, scale) {
  const W = outWidth(cfg);
  let n = 0;
  for (let t = 0; t < T; t++) if (!mask || mask[t]) n++;
  if (n === 0) return 0;
  let total = 0;
  for (let t = 0; t < T; t++) {
    if (mask && !mask[t]) continue;
    const o = t * W;
    if (cfg.out === 'logit') {
      const z = logits[o];
      total += bceLogit(targets[t], z);
      if (dl) dl[o] = (sigmoid(z) - targets[t]) * scale / n;
      continue;
    }
    let m = -Infinity;
    for (let k = 0; k < W; k++) if (logits[o + k] > m) m = logits[o + k];
    let s = 0;
    for (let k = 0; k < W; k++) s += Math.exp(logits[o + k] - m);
    const lse = m + Math.log(s);
    total += lse - logits[o + targets[t]];
    if (dl) {
      for (let k = 0; k < W; k++) {
        dl[o + k] = (Math.exp(logits[o + k] - lse) - (k === targets[t] ? 1 : 0)) * scale / n;
      }
    }
  }
  return total / n;
}

// 'softmax': targets are token ids per position; 'logit': 0/1 per position.
export function attnLoss(model, ids, targets, mask) {
  const c = forward(model, ids);
  return lossFromLogits(model.cfg, c.logits, c.T, targets, mask, null, 1);
}

// ---- backward -----------------------------------------------------------------------------

// Adds scale · d loss / d theta into g (views over a gradient buffer); returns the loss.
function backwardInto(model, ids, targets, mask, g, scale) {
  const {cfg} = model;
  const {d, heads: H, dk, ln, mlp: M} = cfg;
  const c = forward(model, ids);
  const {T} = c, D = H * dk, rs = 1 / Math.sqrt(dk), W = outWidth(cfg);
  const dl = new Float64Array(T * W);
  const loss = lossFromLogits(cfg, c.logits, T, targets, mask, dl, scale);

  const dy = new Float64Array(T * d);
  linearBack(dl, c.y, T, d, model.Wout, W, g.Wout, g.bout, dy);
  let dx = dy;
  if (ln) {
    dx = new Float64Array(T * d);
    lnRowsBack(dy, c.xf, T, d, model.gf, c.muf, c.invf, dx, g.gf, g.bf);
  }

  for (let b = cfg.blocks - 1; b >= 0; b--) {
    const p = model.blocks[b], gb = g.blocks[b], k = c.blocks[b];
    // dx is d loss / d (block output); the residual passes it straight through.
    if (M > 0) {
      const dhr = new Float64Array(T * M);
      linearBack(dx, k.hr, T, M, p.W2, d, gb.W2, gb.c2, dhr);
      for (let i = 0; i < T * M; i++) if (!(k.hpre[i] > 0)) dhr[i] = 0;
      const dw = new Float64Array(T * d);
      linearBack(dhr, k.w, T, d, p.W1, M, gb.W1, gb.c1, dw);
      if (ln) lnRowsBack(dw, k.x1, T, d, p.g2, k.mu2, k.inv2, dx, gb.g2, gb.b2);
      else for (let i = 0; i < T * d; i++) dx[i] += dw[i];
    }
    // dx is now d loss / d x1.
    const dO = new Float64Array(T * D);
    linearBack(dx, k.O, T, D, p.Wo, d, gb.Wo, null, dO);
    const dQ = new Float64Array(T * D), dK = new Float64Array(T * D), dV = new Float64Array(T * D);
    const dA = new Float64Array(T);
    for (let h = 0; h < H; h++) {
      const ho = h * dk;
      for (let t = 0; t < T; t++) {
        const row = (h * T + t) * T, smax = cfg.causal ? t : T - 1;
        let dot = 0;
        for (let s = 0; s <= smax; s++) {
          const a = k.A[row + s];
          let da = 0;
          for (let j = 0; j < dk; j++) {
            const go = dO[t * D + ho + j];
            da += go * k.V[s * D + ho + j];
            dV[s * D + ho + j] += a * go;
          }
          dA[s] = da;
          dot += a * da;
        }
        for (let s = 0; s <= smax; s++) {
          const ds = k.A[row + s] * (dA[s] - dot) * rs;
          if (ds === 0) continue;
          for (let j = 0; j < dk; j++) {
            dQ[t * D + ho + j] += ds * k.K[s * D + ho + j];
            dK[s * D + ho + j] += ds * k.Q[t * D + ho + j];
          }
        }
      }
    }
    const du = new Float64Array(T * d);
    linearBack(dQ, k.u, T, d, p.Wq, D, gb.Wq, null, du);
    linearBack(dK, k.u, T, d, p.Wk, D, gb.Wk, null, du);
    linearBack(dV, k.u, T, d, p.Wv, D, gb.Wv, null, du);
    if (ln) lnRowsBack(du, k.xin, T, d, p.g1, k.mu1, k.inv1, dx, gb.g1, gb.b1);
    else for (let i = 0; i < T * d; i++) dx[i] += du[i];
  }
  for (let t = 0; t < T; t++) {
    for (let j = 0; j < d; j++) {
      g.E[ids[t] * d + j] += dx[t * d + j];
      g.P[t * d + j] += dx[t * d + j];
    }
  }
  return loss;
}

// grads has the model's shape: {theta, E, P, blocks: [...], gf, bf, Wout, bout}.
export function attnBackward(model, ids, targets, mask) {
  const grads = views(model.cfg, new Float64Array(model.theta.length));
  const loss = backwardInto(model, ids, targets, mask, grads, 1);
  return {grads, loss};
}

// ---- trainer ------------------------------------------------------------------------------

const DIVERGED = 1e6;
const masksOf = set => set.masks ?? set.mask;

// Mean over sequences of each sequence's masked mean loss.
function setLoss(model, set) {
  const ms = masksOf(set);
  let s = 0;
  for (let i = 0; i < set.seqs.length; i++) s += attnLoss(model, set.seqs[i], set.targets[i], ms?.[i]);
  return s / set.seqs.length;
}

function setGrad(model, set) {
  const g = views(model.cfg, new Float64Array(model.theta.length));
  const ms = masksOf(set), n = set.seqs.length;
  let loss = 0;
  for (let i = 0; i < n; i++) loss += backwardInto(model, set.seqs[i], set.targets[i], ms?.[i], g, 1 / n);
  return {grad: g.theta, loss: loss / n};
}

const isBad = v => !Number.isFinite(v) || Math.abs(v) > DIVERGED;

// Mini-batch training. data: {seqs, targets, masks? (or mask, as echoData gives)}.
// Without a sampler: a seeded Fisher–Yates shuffle each epoch from one mulberry32(seed)
// stream; state.epochLoss[0] is the loss before training, one more entry per epoch.
// With sampler(rng) → {seqs, targets, masks?}: it is called for each batch with the same
// stream instead, and state.evalLoss (at the iterations in state.evalIter) is the loss on
// data.heldOut, before training and every evalEvery steps. The state object is updated
// in place by step(k); on divergence it keeps the last finite parameters; reset() gives a
// fresh state from the same starting model and the same stream.
export function attnTrainer({model, data, batch, kind = 'adam', hyper = {}, seed, sampler,
  evalEvery = 50}) {
  const start = attnUnflatten(model, attnFlatten(model));
  const n = sampler ? 0 : data.seqs.length;
  let rng, order, cursor, opt, params;

  const fresh = () => {
    rng = mulberry32(seed);
    order = null;
    cursor = 0;
    params = attnFlatten(start);
    opt = optInit(kind, params.length);
    const m = attnUnflatten(start, params);
    if (sampler) {
      const first = setLoss(start, data.heldOut);
      return {model: m, iter: 0, batchLoss: [], evalLoss: [first], evalIter: [0],
        diverged: !Number.isFinite(first)};
    }
    const first = setLoss(start, data);
    return {model: m, epoch: 0, iter: 0, batchLoss: [], epochLoss: [first],
      diverged: !Number.isFinite(first)};
  };

  const shuffle = () => {
    order = Array.from({length: n}, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    cursor = 0;
  };

  const nextBatch = () => {
    if (sampler) return sampler(rng);
    if (cursor === 0) shuffle();
    const idx = order.slice(cursor, cursor + batch), ms = masksOf(data);
    return {seqs: idx.map(i => data.seqs[i]), targets: idx.map(i => data.targets[i]),
      masks: ms && idx.map(i => ms[i])};
  };

  const api = {state: fresh()};

  const one = () => {
    const s = api.state;
    const g = setGrad(s.model, nextBatch());
    if (isBad(g.loss)) { s.diverged = true; return; }
    const r = optStep(kind, hyper, opt, params, g.grad);
    if (!r.params.every(Number.isFinite)) { s.diverged = true; return; }
    params = r.params;
    opt = r.state;
    s.model = attnUnflatten(s.model, params);
    s.batchLoss.push(g.loss);
    s.iter++;
    if (sampler) {
      if (s.iter % evalEvery === 0) {
        const el = setLoss(s.model, data.heldOut);
        s.evalLoss.push(el);
        s.evalIter.push(s.iter);
        if (isBad(el)) s.diverged = true;
      }
      return;
    }
    cursor += batch;
    if (cursor >= n) {
      cursor = 0;
      s.epoch++;
      const el = setLoss(s.model, data);
      s.epochLoss.push(el);
      if (isBad(el)) s.diverged = true;
    }
  };

  api.step = (k = 1) => {
    for (let i = 0; i < k && !api.state.diverged; i++) one();
    return api.state;
  };
  api.reset = () => { api.state = fresh(); return api.state; };
  return api;
}
