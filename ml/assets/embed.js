// Word embeddings: skip-gram with a full softmax, trained on the invented world.
// Pure functions; the trainer follows the shape and rules of nn.js trainer().
import {mulberry32, randn} from './maths.js';
import {optInit, optStep} from './nn.js';

// ---- tokens and pairs ----------------------------------------------------------------

// Words split on spaces; the full stop is its own token '.'.
export function wordTokens(sentences) {
  const toks = sentences.map(s =>
    s.trim().split(/\s+/).filter(Boolean).flatMap(w =>
      (w.length > 1 && w.endsWith('.')) ? [w.slice(0, -1), '.'] : [w]));
  const words = [...new Set(toks.flat())].sort();
  const index = new Map(words.map((w, i) => [w, i]));
  return {words, ids: toks.map(t => t.map(w => index.get(w)))};
}

// [centre, context] for every neighbour within `window`, inside one sentence only.
export function skipgramPairs(ids, window = 2) {
  const out = [];
  for (const s of ids) {
    for (let i = 0; i < s.length; i++) {
      for (let j = Math.max(0, i - window); j <= Math.min(s.length - 1, i + window); j++) {
        if (j !== i) out.push([s[i], s[j]]);
      }
    }
  }
  return out;
}

// ---- model ---------------------------------------------------------------------------

export function skipgramInit({V, d = 2, seed, scale = 0.5}) {
  const rng = mulberry32(seed);
  const mat = () => Array.from({length: V}, () => Array.from({length: d}, () => scale * randn(rng)));
  const Win = mat();
  const Wout = mat();
  return {V, d, Win, Wout};
}

// Scores z_j = Wout[j] . Win[c]; returns log-sum-exp (stable) and softmax p.
function scores(model, c) {
  const {V, d, Win, Wout} = model;
  const h = Win[c], z = new Array(V);
  let m = -Infinity;
  for (let j = 0; j < V; j++) {
    let s = 0;
    for (let k = 0; k < d; k++) s += Wout[j][k] * h[k];
    z[j] = s;
    if (s > m) m = s;
  }
  let sum = 0;
  for (let j = 0; j < V; j++) sum += Math.exp(z[j] - m);
  return {z, lse: m + Math.log(sum)};
}

// Mean of -log softmax(Wout . Win[c])[o], natural log.
export function skipgramLoss(model, pairs) {
  let s = 0;
  for (const [c, o] of pairs) {
    const {z, lse} = scores(model, c);
    s += lse - z[o];
  }
  return s / pairs.length;
}

// d loss / d z_j = p_j - [j = o]; then gOut[j] += dz_j h, gIn[c] += sum_j dz_j Wout[j].
export function skipgramGrad(model, pairs) {
  const {V, d, Win, Wout} = model;
  const gIn = Array.from({length: V}, () => new Array(d).fill(0));
  const gOut = Array.from({length: V}, () => new Array(d).fill(0));
  let loss = 0;
  for (const [c, o] of pairs) {
    const {z, lse} = scores(model, c);
    loss += lse - z[o];
    const h = Win[c];
    for (let j = 0; j < V; j++) {
      const dz = Math.exp(z[j] - lse) - (j === o ? 1 : 0);
      for (let k = 0; k < d; k++) {
        gOut[j][k] += dz * h[k];
        gIn[c][k] += dz * Wout[j][k];
      }
    }
  }
  const n = pairs.length;
  for (let j = 0; j < V; j++) {
    for (let k = 0; k < d; k++) { gIn[j][k] /= n; gOut[j][k] /= n; }
  }
  return {gIn, gOut, loss: loss / n};
}

// ---- trainer -------------------------------------------------------------------------

const DIVERGED = 1e6;

const flatten = m => [...m.Win.flat(), ...m.Wout.flat()];
function unflatten(m, v) {
  const {V, d} = m;
  const rows = o => Array.from({length: V}, (_, i) => v.slice(o + i * d, o + (i + 1) * d));
  return {V, d, Win: rows(0), Wout: rows(V * d)};
}

// Mini-batch training with one seeded shuffle stream, reshuffled each epoch. step(k) takes k
// mini-batch steps; the state object is updated in place; reset() gives a fresh one.
export function skipgramTrainer({model, pairs, batch = 64, kind = 'adam', hyper = {}, seed}) {
  const start = unflatten(model, flatten(model));
  const n = pairs.length;
  let rng, order, cursor, opt, params;

  const fresh = () => {
    rng = mulberry32(seed);
    order = null;
    cursor = 0;
    params = flatten(start);
    opt = optInit(kind, params.length);
    const first = skipgramLoss(start, pairs);
    return {model: unflatten(start, params), epoch: 0, iter: 0, batchLoss: [],
      epochLoss: [first], diverged: !Number.isFinite(first)};
  };

  const shuffle = () => {
    order = Array.from({length: n}, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    cursor = 0;
  };

  const api = {state: fresh()};

  const one = () => {
    const s = api.state;
    if (cursor === 0) shuffle();
    const sub = order.slice(cursor, cursor + batch).map(i => pairs[i]);
    const g = skipgramGrad(s.model, sub);
    if (!Number.isFinite(g.loss) || Math.abs(g.loss) > DIVERGED) { s.diverged = true; return; }
    const r = optStep(kind, hyper, opt, params, [...g.gIn.flat(), ...g.gOut.flat()]);
    if (!r.params.every(Number.isFinite)) { s.diverged = true; return; }
    params = r.params;
    opt = r.state;
    s.model = unflatten(s.model, params);
    s.batchLoss.push(g.loss);
    s.iter++;
    cursor += batch;
    if (cursor >= n) {
      cursor = 0;
      s.epoch++;
      const el = skipgramLoss(s.model, pairs);
      s.epochLoss.push(el);
      if (!Number.isFinite(el) || Math.abs(el) > DIVERGED) s.diverged = true;
    }
  };

  api.step = (k = 1) => {
    for (let i = 0; i < k && !api.state.diverged; i++) one();
    return api.state;
  };
  api.reset = () => { api.state = fresh(); return api.state; };
  return api;
}

// ---- similarity ----------------------------------------------------------------------

export function cosine(a, b) {
  let ab = 0, aa = 0, bb = 0;
  for (let i = 0; i < a.length; i++) { ab += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; }
  if (aa === 0 || bb === 0) return 0;
  return ab / Math.sqrt(aa * bb);
}

// Neighbours of `word` by cosine of the input vectors, best first; the word itself is left out.
export function nearest(model, word, words, {exclude = []} = {}) {
  const i = words.indexOf(word);
  const skip = new Set([word, ...exclude]);
  return words
    .map((w, j) => ({word: w, cos: cosine(model.Win[i], model.Win[j])}))
    .filter(r => !skip.has(r.word))
    .sort((a, b) => b.cos - a.cos);
}
