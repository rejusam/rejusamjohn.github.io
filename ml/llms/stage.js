// Lesson L5.6, "Large language models, in miniature": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l5-llms.test.mjs); the DOM is
// only touched inside createStage. Every forward pass is attnForward from attn.js, every
// training step is attnTrainer, and every number in the page's text comes from
// pageValues or storyValues.

import { mulberry32 } from '../assets/maths.js';
import { ALPHABET, WORLD, worldText, checkFact } from '../assets/world.js';
import { sanitise, charTokenise, ngramModel, nextProbs, softmax, sampleIndex } from '../assets/text.js';
import { attnInit, attnForward, attnTrainer, attnParamCount } from '../assets/attn.js';
import { el } from '../assets/plot.js';
import { createTrainLoop, trainButtons, trainStatus } from '../assets/train-loop.js';

// ---- data and settings -------------------------------------------------------------------

export const TRAIN_SEED = 261;
export const HELD_SEED = 262;
export const SAMPLER_SEED = 291;
export const INIT_SEED = 292;
export const SAMPLE_SEED = 293;
export const FACT_SEED = 294;
export const CONT_SEED = 295;
export const N_TRAIN = 2000;
export const N_HELD = 300;
export const LEN = 32;
export const BATCH = 8;
export const ETA = 0.003;
export const RUN = 2500;
export const EVAL_EVERY = 100;
// Held-out scoring starts at character EVAL_FROM + 1, so the n = 5 counting line has a full context
// at every scored position and all four lines score the same characters.
export const EVAL_FROM = 5;
export const NS = [1, 3, 5];
export const TAUS = [0, 0.5, 1, 1.5];
export const MAX_INPUT = 80;
export const DEFAULT_PROMPT = 'zoe likes ';
export const SAMPLE_START = '. ';
export const SAMPLE_LEN = 60;
export const FACT_MAX = 16;
export const CHECKPOINTS = [500, 1000, 1500, 2000, 2500];
export const CFG = { vocab: ALPHABET.length, len: LEN, d: 32, heads: 2, causal: true, ln: true, mlp: 64, blocks: 1, out: 'softmax', seed: INIT_SEED };
const A = ALPHABET.length;

const CACHE = new Map();
const cached = (key, make) => {
  if (!CACHE.has(key)) CACHE.set(key, make());
  return CACHE.get(key);
};
export const trainText = () => cached('train', () => worldText({ n: N_TRAIN, seed: TRAIN_SEED }));
export const heldText = () => cached('held', () => worldText({ n: N_HELD, seed: HELD_SEED }));
const trainIds = () => cached('trainIds', () => charTokenise(trainText()));
const heldIds = () => cached('heldIds', () => charTokenise(heldText()));
export const startModel = () => attnInit(CFG);

// Windows of 32 characters from character EVAL_FROM on, each scoring its 32 next
// characters: every scored position once, with between 1 and 32 characters before it.
export function heldWindows() {
  return cached('windows', () => {
    const h = heldIds();
    const m = Math.floor((h.length - EVAL_FROM) / LEN);
    const seqs = [];
    const targets = [];
    for (let w = 0; w < m; w++) {
      const s = EVAL_FROM - 1 + w * LEN;
      seqs.push(h.slice(s, s + LEN));
      targets.push(h.slice(s + 1, s + LEN + 1));
    }
    return { seqs, targets };
  });
}
export const scoredCount = () => heldWindows().seqs.length * LEN;

// One training batch: BATCH random windows of LEN + 1 characters (input, and the same
// shifted by one as the targets), drawn from the trainer's random stream.
export function sampleBatch(rng) {
  const t = trainIds();
  const seqs = [];
  const targets = [];
  for (let b = 0; b < BATCH; b++) {
    const s = Math.floor(rng() * (t.length - LEN));
    seqs.push(t.slice(s, s + LEN));
    targets.push(t.slice(s + 1, s + LEN + 1));
  }
  return { seqs, targets };
}

// The trainer's own held-out evaluation runs only before the first step (it would take a
// whole frame); the page evaluates every EVAL_EVERY steps in slices instead (makeLive).
export const makeTrainer = (model = startModel()) =>
  attnTrainer({ model, data: { heldOut: heldWindows() }, batch: BATCH, kind: 'adam', hyper: { eta: ETA }, seed: SAMPLER_SEED, sampler: sampleBatch, evalEvery: Infinity });

// ---- held-out bits per character ------------------------------------------------------------

// −log₂ p of each target in windows w0 … w1 − 1, summed (log-sum-exp, natural log, ÷ ln 2).
export function heldBitsPart(model, w0, w1) {
  const W = heldWindows();
  let sum = 0;
  for (let w = w0; w < w1; w++) {
    const { logits } = attnForward(model, W.seqs[w]);
    for (let t = 0; t < LEN; t++) {
      const z = logits[t];
      let m = -Infinity;
      for (const v of z) if (v > m) m = v;
      let s = 0;
      for (const v of z) s += Math.exp(v - m);
      sum += (m + Math.log(s) - z[W.targets[w][t]]) / Math.LN2;
    }
  }
  return sum;
}
export const heldBits = (model) => heldBitsPart(model, 0, heldWindows().seqs.length) / scoredCount();

// The n-gram lines of the previous lesson (add-k, k = 0.01), scored on the same positions.
export function ngramBits(n) {
  return cached(`ng${n}`, () => {
    const m = ngramModel(trainText(), n);
    const text = heldText();
    const h = heldIds();
    let sum = 0;
    const end = EVAL_FROM + scoredCount();
    for (let i = EVAL_FROM; i < end; i++) sum -= Math.log2(nextProbs(m, text.slice(i - n, i))[h[i]]);
    return sum / scoredCount();
  });
}

// ---- predicting and writing -------------------------------------------------------------------

// Probabilities of the next character at temperature τ, from the last LEN ids; null if none.
export function nextChar(model, ids, tau) {
  if (!ids.length) return null;
  return softmax(attnForward(model, ids.slice(-LEN)).logits.at(-1), tau);
}

// One character drawn at τ from rng.
export const continueOne = (model, ids, tau, rng) => ALPHABET[sampleIndex(nextChar(model, ids, tau), rng)];

// A sample written in slices: from `start`, `length` characters at τ = 1, each from the
// last LEN characters, with mulberry32(seed).
export function sampleJob(model, seed, start = SAMPLE_START, length = SAMPLE_LEN) {
  const rng = mulberry32(seed);
  const xs = charTokenise(start);
  const n0 = xs.length;
  return {
    advance(k) {
      for (let i = 0; i < k && xs.length - n0 < length; i++) xs.push(sampleIndex(nextChar(model, xs, 1), rng));
      return xs.length - n0 >= length;
    },
    text: () => xs.slice(n0).map((i) => ALPHABET[i]).join(''),
  };
}
// The sample shown for step `iter` of the run uses seed SAMPLE_SEED + iter.
export function sampleAt(model, iter) {
  const j = sampleJob(model, SAMPLE_SEED + iter);
  j.advance(SAMPLE_LEN);
  return j.text();
}

// Typed text: lower-cased and capped by sanitise. Each character is a chip; the context is
// every known character, up to the last LEN (unknown characters are skipped).
export function promptState(raw) {
  const { text, unknown } = sanitise(raw, MAX_INPUT);
  const cs = Array.from(text);
  const bad = new Set(unknown);
  const known = cs.map((_, i) => i).filter((i) => !bad.has(i));
  const ctx = new Set(known.slice(-LEN));
  const chips = cs.map((ch, i) => ({ ch, unknown: bad.has(i), ctx: ctx.has(i) }));
  const ids = [...ctx].map((i) => ALPHABET.indexOf(cs[i]));
  return { text, chips, unknown: unknown.length, ids, context: ids.length ? ids.map((i) => ALPHABET[i]).join('') : null };
}

// ---- true or false in the invented world ------------------------------------------------------

// The 16 prompts, two per person; each has its true continuation and the false ones.
// Continuations end with the full stop, so they are the exact sentences checkFact rates
// 'true' or 'false' (a "lives" sentence needs the place's own by/in).
export const FACT_PROMPTS = WORLD.people.flatMap((p) => [
  { person: p, kind: 'likes', prompt: `${p} likes `, conts: WORLD.foods.map((f) => ({ cont: `${f}.`, ok: WORLD.likes[p] === f })) },
  { person: p, kind: 'lives', prompt: `${p} lives `, conts: WORLD.places.map((pl) => ({ cont: `${WORLD.prep[pl]} the ${pl}.`, ok: WORLD.livesIn[p] === pl })) },
]);

// The probability, at each τ in taus, that sampling continues `prompt` with exactly `cont`:
// the product of the per-character probabilities, all read from one forward pass.
function contProbs(model, prompt, cont, taus) {
  const xs = charTokenise(prompt + cont);
  const { logits } = attnForward(model, xs.slice(0, -1));
  return taus.map((tau) => {
    let p = 1;
    for (let i = prompt.length; i < xs.length; i++) p *= softmax(logits[i - 1], tau)[xs[i]];
    return p;
  });
}

// For each τ: per person the probability of the true sentence (t), of a false one (f) and of
// anything else (o = 1 − t − f), and their means over the 8 people. Exact, no sampling.
export function factProbs(model, taus) {
  const out = taus.map(() => ({ per: WORLD.people.map(() => ({})), likes: { t: 0, f: 0, o: 0 }, lives: { t: 0, f: 0, o: 0 } }));
  FACT_PROMPTS.forEach((fp, k) => {
    const i = k >> 1;
    const tf = taus.map(() => ({ t: 0, f: 0 }));
    for (const c of fp.conts) {
      contProbs(model, fp.prompt, c.cont, taus).forEach((p, j) => { tf[j][c.ok ? 't' : 'f'] += p; });
    }
    tf.forEach((v, j) => {
      const r = { t: v.t, f: v.f, o: 1 - v.t - v.f };
      out[j].per[i][fp.kind] = r;
      for (const key of ['t', 'f', 'o']) out[j][fp.kind][key] += r[key] / WORLD.people.length;
    });
  });
  return out;
}

// One sentence per prompt, sampled at τ until a full stop or FACT_MAX characters, then rated.
export function writeFacts(model, tau, seed) {
  const rng = mulberry32(seed);
  const dot = ALPHABET.indexOf('.');
  return FACT_PROMPTS.map(({ prompt }) => {
    const xs = charTokenise(prompt);
    const n0 = xs.length;
    while (xs.length - n0 < FACT_MAX && xs.at(-1) !== dot) xs.push(sampleIndex(nextChar(model, xs, tau), rng));
    const cont = xs.slice(n0).map((i) => ALPHABET[i]).join('');
    const sentence = prompt + cont;
    return { prompt, cont, sentence, verdict: checkFact(sentence) };
  });
}
export function factCounts(list) {
  const c = { true: 0, false: 0, other: 0 };
  for (const r of list) c[r.verdict]++;
  return c;
}

// ---- the live run -----------------------------------------------------------------------------

const EVAL_CHUNK = 12;     // held-out windows per unit of work
const SAMPLE_CHUNK = 8;    // sample characters per unit of work

// The trainer plus, at step 0 and every EVAL_EVERY steps, a held-out point and a sample.
// work(m) does m units: a training step, or a slice of the pending point and sample, which
// are for the model as it was at that step. The run is the same whatever the slicing.
export function makeLive(model = startModel()) {
  const tr = makeTrainer(model);
  const M = heldWindows().seqs.length;
  const live = { trainer: tr, curve: [], samples: [], models: new Map(), job: null };
  const begin = () => {
    const s = tr.state;
    live.job = { iter: s.iter, model: s.model, w: 0, sum: 0, sample: sampleJob(s.model, SAMPLE_SEED + s.iter), sampled: false };
    if (CHECKPOINTS.includes(s.iter)) live.models.set(s.iter, s.model);
  };
  const unit = () => {
    if (!live.job && !live.curve.length) begin();
    const j = live.job;
    if (j) {
      if (j.w < M) {
        const w1 = Math.min(M, j.w + EVAL_CHUNK);
        j.sum += heldBitsPart(j.model, j.w, w1);
        j.w = w1;
      } else if (!j.sampled) j.sampled = j.sample.advance(SAMPLE_CHUNK);
      else {
        live.curve.push({ iter: j.iter, bits: j.sum / scoredCount() });
        live.samples.push({ iter: j.iter, text: j.sample.text() });
        live.job = null;
      }
      return;
    }
    tr.step(1);
    if (!tr.state.diverged && tr.state.iter % EVAL_EVERY === 0) begin();
  };
  live.idle = () => !live.job && live.curve.length > 0;
  live.done = () => tr.state.diverged || (tr.state.iter >= RUN && live.idle());
  live.work = (m) => { for (let i = 0; i < m && !live.done(); i++) unit(); };
  live.reset = () => {
    tr.reset();
    live.curve = [];
    live.samples = [];
    live.models = new Map();
    live.job = null;
  };
  return live;
}

// ---- the story: one full run from a given start -------------------------------------------------

export function runFrom(model) {
  const live = makeLive(model);
  while (!live.done()) live.work(1000);
  return live;
}
export const fullRun = () => cached('run', () => runFrom(startModel()));

// Complete words of a sample (the first and last may be cut off, so they are dropped), and
// the share of them that are words of the training text.
export function worldWordShare(text) {
  const vocab = cached('words', () => new Set(trainText().split(/[ .]+/).filter(Boolean)));
  const words = text.split(/[ .]+/).slice(1, -1).filter(Boolean);
  return words.length ? words.filter((w) => vocab.has(w)).length / words.length : 0;
}
// The first step from which every later sample's share is at least `level`.
export function sharesFrom(live, level) {
  const s = live.samples.map((x) => worldWordShare(x.text));
  let i = s.length;
  while (i > 0 && s[i - 1] >= level) i--;
  return live.samples[i].iter;
}

// Mean probabilities of the true and false sentences at τ = 1, at each checkpoint.
export const checkpointFacts = (live) => CHECKPOINTS.map((iter) => ({ iter, ...factProbs(live.models.get(iter), [1])[0] }));

export const count = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
const num2 = (v) => v.toFixed(2);

export function storyValuesFor(model, live = runFrom(model)) {
  const v = {};
  const [n1, n3, n5] = NS.map(ngramBits);
  v['ngram-1'] = num2(n1);
  v['ngram-3'] = num2(n3);
  v['ngram-5'] = num2(n5);
  v['ngram-5-3'] = n5.toFixed(3);
  const c = live.curve;
  v['bits-start'] = num2(c[0].bits);
  v['below-1-at'] = count(c.find((p) => p.bits < n1).iter);
  v['below-3-at'] = count(c.find((p) => p.bits < n3).iter);
  v['bits-end'] = c.at(-1).bits.toFixed(3);
  v['words-start'] = `${Math.round(worldWordShare(live.samples[0].text) * 100)}%`;
  v['words-90-from'] = count(sharesFrom(live, 0.9));
  v['words-all-from'] = count(sharesFrom(live, 1));
  const cps = checkpointFacts(live);
  for (const cp of cps) {
    v[`food-${cp.iter}`] = num2(cp.likes.t);
    v[`place-${cp.iter}`] = num2(cp.lives.t);
  }
  // Shape before facts: at the first checkpoint, a well-formed food sentence (true or false).
  v['form-first'] = num2(cps[0].likes.t + cps[0].likes.f);
  v['food-false-end'] = num2(cps.at(-1).likes.f);
  v['place-false-end'] = num2(cps.at(-1).lives.f);
  v['dip-at'] = count(c.find((p) => p.bits < n5).iter);
  const split = placeSplit(live.trainer.state.model);
  v['prep-end'] = num2(split.prep);
  v['pair-end'] = num2(split.pair);
  const byTau = factProbs(live.trainer.state.model, TAUS);
  TAUS.forEach((_, i) => {
    if (i === 0) return;   // τ = 0 is given as counts out of 8 (greedy-food, greedy-place)
    const r = byTau[i];
    v[`tau-${i}-ft`] = num2(r.likes.t);
    v[`tau-${i}-ff`] = num2(r.likes.f);
    v[`tau-${i}-fo`] = num2(Math.max(0, r.likes.o));
    v[`tau-${i}-pt`] = num2(r.lives.t);
    v[`tau-${i}-pf`] = num2(r.lives.f);
    v[`tau-${i}-po`] = num2(Math.max(0, r.lives.o));
  });
  // At τ = 0 each prompt has one sentence: how many of the 8 come out true.
  v['greedy-food'] = String(byTau[0].per.filter((r) => r.likes.t === 1).length);
  v['greedy-place'] = String(byTau[0].per.filter((r) => r.lives.t === 1).length);
  return v;
}

// After "<name> lives ": the mean probability of the person's own "by " or "in ", and the
// mean share the true place gets of the two places that share that word (τ = 1).
export function placeSplit(model) {
  let prep = 0;
  let pair = 0;
  for (const p of WORLD.people) {
    const own = WORLD.livesIn[p];
    const word = WORLD.prep[own];
    const prompt = `${p} lives `;
    prep += contProbs(model, prompt, `${word} `, [1])[0] / WORLD.people.length;
    const sib = WORLD.places.find((x) => x !== own && WORLD.prep[x] === word);
    const t = contProbs(model, prompt, `${word} the ${own}.`, [1])[0];
    const s = contProbs(model, prompt, `${word} the ${sib}.`, [1])[0];
    pair += t / (t + s) / WORLD.people.length;
  }
  return { prep, pair };
}
export const storyValues = () => cached('story', () => storyValuesFor(startModel(), fullRun()));

// ---- the parameter count, part by part ---------------------------------------------------------

export function paramParts(model) {
  const { d, mlp, heads, dk, vocab, len } = model.cfg;
  const D = heads * dk;
  return [
    { name: 'token embeddings E', n: vocab * d },
    { name: 'position embeddings P', n: len * d },
    { name: 'attention (2 heads)', n: 4 * D * d },
    { name: 'per-position network', n: mlp * d + mlp + d * mlp + d },
    { name: 'layer norms (3)', n: 3 * 2 * d },
    { name: 'output layer', n: vocab * d + vocab },
  ];
}

// ---- values written into the page -------------------------------------------------------------

export function pageValues() {
  const m5 = cached('m5', () => ngramModel(trainText(), 5));
  return {
    'n-params': count(attnParamCount(startModel())),
    'n-train': count(N_TRAIN),
    'n-held': count(N_HELD),
    'train-chars': count(trainText().length),
    'scored': count(scoredCount()),
    'run': count(RUN),
    'predictions': count(RUN * BATCH * LEN),
    'uniform-bits': Math.log2(A).toFixed(2),
    'p5-a': nextProbs(m5, 'ikes ')[ALPHABET.indexOf('a')].toFixed(2),
  };
}

// ---- read-outs --------------------------------------------------------------------------------------

export const shown = (s) => s.replace(/ /g, '␣');
const TAU_NAME = (t) => (t === 0 ? '0 (greedy)' : String(t));
export function top3(p) {
  return p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]).slice(0, 3)
    .map(([v, i]) => `${shown(ALPHABET[i])} ${v.toFixed(2)}`).join(' · ');
}
const modelLine = (iter) => (iter === 0 ? 'untrained: press Train in step 2' : `model after step ${count(iter)} of ${count(RUN)}`);

export const stepOneInfo = () => ({ p5: pageValues()['p5-a'] });

export function liveInfo(live, running) {
  const s = live.trainer.state;
  return { iter: s.iter, running, done: live.done(), diverged: s.diverged, last: live.curve.at(-1) || null };
}

export function promptInfo(model, raw, tau, iter) {
  const ps = promptState(raw);
  return { ps, tau, iter, p: nextChar(model, ps.ids, tau) };
}

export function readout(step, s) {
  if (step === 0) return ['each position predicts the character after it', `n = 5 counting, after ␣ikes␣: p(a) = ${s.p5}`, 'this model sees up to 32 characters back'];
  if (step === 1) {
    const pt = s.last ? `held-out: ${s.last.bits.toFixed(2)} bits/char (step ${count(s.last.iter)})` : 'held-out: not measured yet';
    return [`step ${count(s.iter)} of ${count(RUN)} · ${trainStatus(s)}`, pt, `n-grams: n = 1, 3, 5 dashed`];
  }
  if (step === 2) {
    if (!s.p) return [modelLine(s.iter), 'no known characters: nothing to predict', `τ = ${TAU_NAME(s.tau)}`];
    return [modelLine(s.iter), `τ = ${TAU_NAME(s.tau)} · context: ${s.ps.ids.length} characters`, top3(s.p)];
  }
  if (step === 3) {
    const r = (k) => `true ${s.fp[k].t.toFixed(2)} · false ${s.fp[k].f.toFixed(2)} · else ${Math.max(0, s.fp[k].o).toFixed(2)}`;
    return [`${modelLine(s.iter)} · τ = ${s.tau}`, `likes: ${r('likes')}`, `lives: ${r('lives')}`];
  }
  if (step === 4) return [`${pageValues()['n-params']} weights and biases`, `${pageValues()['train-chars']} characters of training text`, `${pageValues().predictions} predictions in the run`];
  if (step === 5) return ['1 pretraining: this lesson', '2 instruction tuning · 3 preference tuning', 'each step is more training of the same model'];
  return [];
}

// ---------------------------------------------------------------- the stage

const STRONG = '#f1f5f9';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const VIOLET = '#a78bfa';
const BG = '#0f172a';
const LINE = '#334155';
const SURFACE = '#1e293b';
const HALO = { stroke: BG, 'stroke-width': 3, 'paint-order': 'stroke', 'stroke-linejoin': 'round' };
const P = (v) => Number(v.toFixed(2));
const RY = [292, 314, 336];
const FONT = "'JetBrains Mono', monospace";
const STEP1 = 'tom likes apples.';
const SHOW_SAMPLES = 4;

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  let step = 0;
  let tauIx = 2;
  let typed = DEFAULT_PROMPT;
  let live = null;
  let factSeed = FACT_SEED;
  let facts = null;            // { key, list, totals }
  const contRng = mulberry32(CONT_SEED);
  const FP = new Map();

  svg.replaceChildren();
  svg.setAttribute('viewBox', '0 0 600 350');
  svg.setAttribute('role', 'img');
  const show = (node, on) => { node.style.display = on ? '' : 'none'; };
  const label = (parent, x, y, text, anchor = 'start', fill = MUTED, size = 12) => {
    const t = el('text', { class: 'stage-text', x, y, 'text-anchor': anchor, 'font-size': size, text, ...HALO }, parent);
    t.style.fill = fill;
    return t;
  };
  const getLive = () => {
    if (!live) live = makeLive();
    return live;
  };
  const model = () => (live ? live.trainer.state.model : cached('start', startModel));
  const iterNow = () => (live ? live.trainer.state.iter : 0);

  // ---- step 1: one piece of text, every position predicting the character after it.
  const oneG = el('g', {}, svg);
  {
    const CW = 32;
    const X0 = 74;
    const input = STEP1.slice(0, -1);
    const target = STEP1.slice(1);
    label(oneG, X0 - 10, 56, 'input', 'end', STRONG, 13);
    label(oneG, X0 - 10, 106, 'next', 'end', STRONG, 13);
    for (let i = 0; i < input.length; i++) {
      const x = X0 + i * CW;
      const hot = i === 9;
      el('rect', { x: x + 1, y: 36, width: CW - 2, height: 28, rx: 3, fill: SURFACE, stroke: hot ? CYAN : LINE, 'stroke-width': hot ? 2.5 : 1 }, oneG);
      label(oneG, x + CW / 2, 56, shown(input[i]), 'middle', STRONG, 14);
      el('line', { x1: x + CW / 2, x2: x + CW / 2, y1: 66, y2: 84, stroke: hot ? CYAN : LINE, 'stroke-width': hot ? 2 : 1 }, oneG);
      el('rect', { x: x + 1, y: 86, width: CW - 2, height: 28, rx: 3, fill: 'none', stroke: hot ? CYAN : LINE, 'stroke-width': hot ? 2.5 : 1, 'stroke-dasharray': '4 3' }, oneG);
      label(oneG, x + CW / 2, 106, shown(target[i]), 'middle', hot ? CYAN : STRONG, 14);
    }
    // What each model can use for the prediction after "tom likes ": brackets under the input.
    const brk = (i0, i1, y, text, colour, dash) => {
      const x0 = X0 + i0 * CW + 3;
      const x1 = X0 + (i1 + 1) * CW - 3;
      el('path', { d: `M${x0},${y - 8}L${x0},${y}L${x1},${y}L${x1},${y - 8}`, fill: 'none', stroke: colour, 'stroke-width': 2, 'stroke-dasharray': dash }, oneG);
      label(oneG, x0, y + 18, text, 'start', colour, 12);
    };
    brk(5, 9, 150, 'counting with n = 5 uses only these', VIOLET, '5 3');
    brk(0, 9, 196, 'this model can use all of them (up to 32)', CYAN, null);
    label(oneG, X0, 248, 'one prediction per column: 16 here, 32 per window', 'start', MUTED, 12);
  }

  // ---- step 2: held-out bits per character, with the n-gram lines.
  const curG = el('g', {}, svg);
  const AX0 = 64;
  const AX1 = 540;
  const AY0 = 250;
  const AY1 = 40;
  const YMAX = 2.5;
  const ax = (i) => AX0 + (i / RUN) * (AX1 - AX0);
  const ay = (b) => AY0 - (Math.min(b, YMAX) / YMAX) * (AY0 - AY1);
  label(curG, 16, 22, 'held-out bits per character', 'start', STRONG);
  for (const b of [0, 0.5, 1, 1.5, 2, 2.5]) {
    el('line', { x1: AX0, x2: AX1, y1: P(ay(b)), y2: P(ay(b)), stroke: LINE, 'stroke-dasharray': b ? '2 4' : null }, curG);
    label(curG, AX0 - 8, P(ay(b)) + 4, b.toFixed(1), 'end', MUTED, 10);
  }
  for (const i of [0, 500, 1000, 1500, 2000, 2500]) label(curG, P(ax(i)), AY0 + 16, count(i), 'middle', MUTED, 10);
  label(curG, AX1, AY0 + 30, 'training step', 'end', MUTED, 11);
  const DASH = { 1: '10 4', 3: '4 3', 5: '1 3' };
  const refs = NS.map((n) => ({
    n,
    line: el('line', { x1: AX0, x2: AX1, stroke: VIOLET, 'stroke-width': 1.75, 'stroke-dasharray': DASH[n] }, curG),
    lab: label(curG, AX1 + 6, 0, `n = ${n}`, 'start', VIOLET, 11),
  }));
  const curPath = el('path', { d: '', fill: 'none', stroke: CYAN, 'stroke-width': 2, 'stroke-linejoin': 'round' }, curG);
  const curDot = el('circle', { r: 4, fill: CYAN, stroke: BG, 'stroke-width': 1.5 }, curG);
  const offTop = label(curG, AX0 + 6, AY1 - 6, '', 'start', MUTED, 11);

  // ---- step 3: next-character probabilities, bars over the 28 symbols.
  const barG = el('g', {}, svg);
  const CX0 = 40;
  const CW = 19.5;
  const cx = (i) => CX0 + i * CW;
  const BY0 = 222;
  const BH = 170;
  const barTitle = label(barG, 16, 24, '', 'start', STRONG);
  label(barG, 584, 24, 'dashed: τ = 1, the model itself', 'end', MUTED, 11);
  el('line', { x1: CX0 - 4, x2: cx(A), y1: BY0, y2: BY0, stroke: LINE }, barG);
  for (const v of [0.5, 1]) {
    el('line', { x1: CX0 - 4, x2: cx(A), y1: BY0 - v * BH, y2: BY0 - v * BH, stroke: LINE, 'stroke-dasharray': '2 4' }, barG);
    label(barG, CX0 - 8, BY0 - v * BH + 4, String(v), 'end', MUTED, 10);
  }
  const bars = ALPHABET.map((ch, i) => {
    const ref = el('rect', { x: P(cx(i) + 2), width: CW - 4, fill: 'none', stroke: VIOLET, 'stroke-width': 1.25, 'stroke-dasharray': '3 2' }, barG);
    const rect = el('rect', { x: P(cx(i) + 2), width: CW - 4, fill: CYAN, 'fill-opacity': 0.45, stroke: CYAN, 'stroke-width': 1 }, barG);
    const val = label(barG, P(cx(i) + CW / 2), 0, '', 'middle', STRONG, 9);
    label(barG, P(cx(i) + CW / 2), BY0 + 18, shown(ch), 'middle', STRONG, 13);
    return { ref, rect, val };
  });

  // ---- step 4: true, false and other, as two stacked bars (one per sentence pattern).
  const factG = el('g', {}, svg);
  const FX0 = 120;
  const FW = 440;
  const factTitle = label(factG, 16, 24, '', 'start', STRONG);
  label(factG, 16, 44, 'exact probabilities, mean over the 8 people', 'start', MUTED, 11);
  const KINDS = [['likes', '"<name> likes …"'], ['lives', '"<name> lives …"']];
  const segs = KINDS.map(([kind, name], r) => {
    const y = 84 + r * 84;
    label(factG, FX0, y - 10, name, 'start', STRONG, 12);
    label(factG, FX0 - 10, y + 22, kind, 'end', MUTED, 12);
    const t = el('rect', { y, height: 32, fill: CYAN, 'fill-opacity': 0.6, stroke: CYAN }, factG);
    const f = el('rect', { y, height: 32, fill: VIOLET, 'fill-opacity': 0.35, stroke: VIOLET, 'stroke-dasharray': '5 3' }, factG);
    const o = el('rect', { y, height: 32, fill: 'none', stroke: MUTED, 'stroke-dasharray': '1 3' }, factG);
    const lt = label(factG, 0, y + 52, '', 'start', CYAN, 12);
    const lf = label(factG, 0, y + 52, '', 'middle', VIOLET, 12);
    return { kind, y, t, f, o, lt, lf };
  });
  {
    const y = 262;
    el('rect', { x: FX0, y: y - 11, width: 14, height: 14, fill: CYAN, 'fill-opacity': 0.6, stroke: CYAN }, factG);
    label(factG, FX0 + 20, y, 'true', 'start', STRONG, 11);
    el('rect', { x: FX0 + 80, y: y - 11, width: 14, height: 14, fill: VIOLET, 'fill-opacity': 0.35, stroke: VIOLET, 'stroke-dasharray': '5 3' }, factG);
    label(factG, FX0 + 100, y, 'false', 'start', STRONG, 11);
    el('rect', { x: FX0 + 160, y: y - 11, width: 14, height: 14, fill: 'none', stroke: MUTED, 'stroke-dasharray': '1 3' }, factG);
    label(factG, FX0 + 180, y, 'anything else', 'start', STRONG, 11);
  }

  // ---- step 5: where the weights are.
  const sizeG = el('g', {}, svg);
  label(sizeG, 16, 24, `the ${pageValues()['n-params']} weights and biases, part by part`, 'start', STRONG);
  {
    const parts = paramParts(model());
    const mx = Math.max(...parts.map((p) => p.n));
    parts.forEach((p, i) => {
      const y = 50 + i * 36;
      label(sizeG, 220, y + 18, p.name, 'end', STRONG, 12);
      const w = (p.n / mx) * 280;
      el('rect', { x: 230, y, width: P(w), height: 24, fill: CYAN, 'fill-opacity': 0.45, stroke: CYAN }, sizeG);
      label(sizeG, P(236 + w), y + 17, count(p.n), 'start', STRONG, 12);
    });
  }

  // ---- step 6: after pretraining.
  const tuneG = el('g', {}, svg);
  {
    const boxes = [['pretraining', 'predict the next token', 'on a very large', 'amount of text'], ['instruction tuning', 'more training on', 'requests paired with', 'good replies'], ['preference tuning', 'more training towards', 'replies people', 'preferred']];
    boxes.forEach(([h, a, b, c], i) => {
      const x = 16 + i * 196;
      el('rect', { x, y: 70, width: 176, height: 120, rx: 6, fill: SURFACE, stroke: i === 0 ? CYAN : LINE, 'stroke-width': i === 0 ? 2 : 1 }, tuneG);
      label(tuneG, x + 12, 100, `${i + 1}  ${h}`, 'start', STRONG, 13);
      label(tuneG, x + 12, 132, a, 'start', MUTED, 11);
      label(tuneG, x + 12, 150, b, 'start', MUTED, 11);
      label(tuneG, x + 12, 168, c, 'start', MUTED, 11);
      if (i < 2) {
        el('path', { d: `M${x + 180},130L${x + 192},130`, stroke: MUTED, 'stroke-width': 1.5 }, tuneG);
        el('path', { d: `M${x + 188},126L${x + 194},130L${x + 188},134Z`, fill: MUTED }, tuneG);
      }
    });
    label(tuneG, 28, 216, 'this lesson builds step 1, in miniature', 'start', CYAN, 12);
  }

  const rowsT = RY.map((y, k) => label(svg, 16, y, '', 'start', k === 0 ? STRONG : MUTED, 13));

  // ---- the panel under the picture: samples, the prompt box, written sentences. Text only.
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const small = (tag, text = '') => {
    const n = make(tag, { textContent: text });
    Object.assign(n.style, { margin: '0 0 6px', color: MUTED, fontFamily: FONT, fontSize: '12px' });
    return n;
  };
  const panel = make('div');
  panel.className = 'token-panel';
  Object.assign(panel.style, { marginTop: '8px', paddingTop: '8px', borderTop: `1px solid ${LINE}` });
  const sampTitle = small('p');
  const sampBox = make('div');
  sampBox.setAttribute('aria-live', 'off');
  const inRow = make('label');
  Object.assign(inRow.style, { display: 'grid', gap: '4px', margin: '0 0 8px', color: MUTED, fontFamily: FONT, fontSize: '12px' });
  const input = make('input', { type: 'text', autocomplete: 'off', spellcheck: false, value: typed });
  input.maxLength = MAX_INPUT;
  input.setAttribute('autocapitalize', 'off');
  Object.assign(input.style, { width: '100%', boxSizing: 'border-box', padding: '6px 8px', background: BG, color: STRONG, border: `1px solid ${MUTED}`, borderRadius: '4px', fontFamily: FONT, fontSize: '14px' });
  inRow.append('your text so far (up to 80 characters, never stored or sent)', input);
  const pTitle = small('p');
  const chipsBox = make('div');
  chipsBox.setAttribute('aria-hidden', 'true');
  Object.assign(chipsBox.style, { display: 'flex', flexWrap: 'wrap', alignItems: 'flex-start', marginBottom: '6px' });
  const chipsSaid = make('p');
  chipsSaid.className = 'vh';
  const factTotals = small('p');
  const factBox = make('div');
  factBox.setAttribute('aria-live', 'off');
  Object.assign(factBox.style, { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(210px, 1fr))', columnGap: '12px' });
  panel.append(sampTitle, sampBox, inRow, pTitle, chipsBox, chipsSaid, factTotals, factBox);
  svg.after(panel);

  const line = (tag, text, strong) => {
    const row = make('p');
    Object.assign(row.style, { margin: '0 0 4px', padding: '2px 6px', fontFamily: FONT, fontSize: '13px', lineHeight: '1.5', overflowWrap: 'anywhere' });
    const t = make('span', { textContent: `${tag}  ` });
    Object.assign(t.style, { color: MUTED, whiteSpace: 'pre' });
    const s = make('span', { textContent: text });
    s.style.color = strong ? STRONG : '#e2e8f0';
    row.append(t, s);
    return row;
  };
  const chip = (c) => {
    const s = make('span');
    s.className = c.unknown ? 'token-chip unknown' : 'token-chip';
    s.textContent = c.unknown ? '?' : shown(c.ch);
    if (c.ctx) Object.assign(s.style, { borderColor: CYAN, color: STRONG });
    return s;
  };
  const MARK = { true: '✓ true ', false: '✗ false', other: '– other' };

  // ---- controls.
  const tauGroup = make('div');
  const tauLab = make('label');
  const tauIn = make('input', { type: 'range', min: '0', max: String(TAUS.length - 1), step: '1', value: String(tauIx) });
  const tauOut = make('output');
  tauOut.setAttribute('aria-hidden', 'true');
  tauLab.append('temperature τ', ' ', tauIn, ' ', tauOut);
  tauGroup.append(tauLab);
  const contGroup = make('div');
  const contBtn = make('button', { type: 'button', textContent: 'Continue' });
  contGroup.append(contBtn);
  const factGroup = make('div');
  const factBtn = make('button', { type: 'button', textContent: 'Write 16 sentences' });
  factGroup.append(factBtn);
  const trainGroup = make('div');
  trainGroup.setAttribute('role', 'group');
  trainGroup.setAttribute('aria-label', 'train the language model');
  controls.append(trainGroup, tauGroup, contGroup, factGroup);

  const tau = () => TAUS[tauIx];
  const factsNow = () => {
    const key = `${iterNow()}:${tauIx}`;
    if (!FP.has(key)) {
      if (FP.size > 40) FP.clear();
      FP.set(key, factProbs(model(), [tau()])[0]);
    }
    return FP.get(key);
  };

  function info() {
    if (step === 0) return stepOneInfo();
    if (step === 1) return liveInfo(getLive(), loop.activity());
    if (step === 2) return promptInfo(model(), typed, tau(), iterNow());
    if (step === 3) return { iter: iterNow(), tau: tau(), fp: factsNow() };
    return null;
  }

  function drawBars(p, ref) {
    bars.forEach((b, i) => {
      const h = p ? p[i] * BH : 0;
      const hr = ref ? ref[i] * BH : 0;
      b.rect.setAttribute('y', P(BY0 - h));
      b.rect.setAttribute('height', P(h));
      b.ref.setAttribute('y', P(BY0 - hr));
      b.ref.setAttribute('height', P(hr));
      b.val.textContent = '';
      b.val.setAttribute('y', P(BY0 - Math.max(h, hr) - 5));
    });
    if (!p) return;
    const done = new Set();
    p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).forEach(([v, i]) => {
      if (v < 0.1 || done.has(i - 1) || done.has(i + 1)) return;
      bars[i].val.textContent = v.toFixed(2).replace(/^0\./, '.');
      done.add(i);
    });
  }

  function render() {
    show(oneG, step === 0);
    show(curG, step === 1);
    show(barG, step === 2);
    show(factG, step === 3);
    show(sizeG, step === 4);
    show(tuneG, step === 5);
    tauOut.textContent = TAU_NAME(tau());
    tauIn.value = String(tauIx);
    tauIn.setAttribute('aria-valuetext', `temperature ${TAU_NAME(tau())}`);
    const s = info();

    if (step === 1) {
      const lv = getLive();
      refs.forEach((r) => {
        const y = P(ay(ngramBits(r.n)));
        r.line.setAttribute('y1', y);
        r.line.setAttribute('y2', y);
        // n = 3 and n = 5 sit close together: one label above its line, one below.
        r.lab.setAttribute('y', P(y + (r.n === 3 ? -4 : r.n === 5 ? 12 : 4)));
      });
      const pts = lv.curve.map((c) => [ax(c.iter), ay(c.bits)]);
      curPath.setAttribute('d', pts.map(([x, y], i) => `${i ? 'L' : 'M'}${P(x)},${P(y)}`).join(''));
      show(curDot, pts.length > 0);
      if (pts.length) { curDot.setAttribute('cx', P(pts.at(-1)[0])); curDot.setAttribute('cy', P(pts.at(-1)[1])); }
      const high = lv.curve.filter((c) => c.bits > YMAX);
      offTop.textContent = high.length ? `step 0: ${high[0].bits.toFixed(2)} bits, above the top` : '';
    }
    if (step === 2) {
      drawBars(s.p, s.p ? nextChar(model(), s.ps.ids, 1) : null);
      barTitle.textContent = s.p ? `next character after your text, at τ = ${TAU_NAME(s.tau)}` : 'next character: type a letter first';
    }
    if (step === 3) {
      factTitle.textContent = `the sentence after the prompt, at τ = ${TAU_NAME(s.tau)}`;
      for (const g of segs) {
        const r = s.fp[g.kind];
        const wt = r.t * FW;
        const wf = r.f * FW;
        const wo = Math.max(0, 1 - r.t - r.f) * FW;
        g.t.setAttribute('x', FX0); g.t.setAttribute('width', P(wt));
        g.f.setAttribute('x', P(FX0 + wt)); g.f.setAttribute('width', P(wf));
        g.o.setAttribute('x', P(FX0 + wt + wf)); g.o.setAttribute('width', P(wo));
        g.lt.textContent = `true ${r.t.toFixed(2)}`;
        g.lt.setAttribute('x', FX0);
        g.lf.textContent = `false ${r.f.toFixed(2)}`;
        g.lf.setAttribute('x', P(Math.min(Math.max(FX0 + wt + wf / 2, FX0 + 150), FX0 + FW - 40)));
      }
    }
    renderPanel(s);
    const texts = readout(step, s);
    rowsT.forEach((t, k) => { t.textContent = texts[k] || ''; });
    svg.setAttribute('aria-label', ariaFor(texts));
  }

  function renderPanel(s) {
    show(panel, step >= 1 && step <= 3);
    show(sampTitle, step === 1);
    show(sampBox, step === 1);
    show(inRow, step === 2);
    show(pTitle, step === 2);
    show(chipsBox, step === 2);
    show(chipsSaid, step === 2);
    show(factTotals, step === 3);
    factBox.style.display = step === 3 ? 'grid' : 'none';
    if (step === 1) {
      const lv = getLive();
      sampTitle.textContent = `samples written during training (τ = 1, each from its own seed), newest first`;
      const last = lv.samples.slice(-SHOW_SAMPLES).reverse();
      sampBox.replaceChildren(...(last.length ? last.map((x, i) => line(`step ${count(x.iter).padStart(5)}`, x.text, i === 0)) : [line('', 'press Train: a sample appears at step 0 and every 100 steps', false)]));
    }
    if (step === 2) {
      pTitle.textContent = s.ps.context === null ? 'no characters from the 28 yet' : `outlined: the context (the last ${s.ps.ids.length} known characters); ? marks a character outside the ${A}`;
      chipsBox.replaceChildren(...s.ps.chips.map(chip));
      chipsSaid.textContent = `Context: ${s.ps.context === null ? 'none' : s.ps.context.replace(/ /g, 'space ')}.`;
    }
    if (step === 3) {
      const key = `${iterNow()}:${tauIx}`;
      if (!facts || facts.key !== key) {
        factTotals.textContent = `Write 16 sentences: one for each prompt, sampled at τ = ${TAU_NAME(tau())}`;
        factBox.replaceChildren();
      } else {
        const c = facts.totals;
        factTotals.textContent = `so far at this τ and step: ${c.true + c.false + c.other} sentences · ${c.true} true · ${c.false} false · ${c.other} other`;
        factBox.replaceChildren(...facts.list.map((r) => line(MARK[r.verdict], r.sentence, r.verdict !== 'true')));
      }
    }
  }

  function ariaFor(texts) {
    const rest = texts.filter(Boolean).join('. ');
    if (step === 0) return `The text tom likes apples, one character per column, with the character that follows each one underneath. Brackets show that the counting model with n = 5 uses only the last 5 characters before the prediction after likes, while this model can use all of them. ${rest}.`;
    if (step === 1) return `A line of held-out bits per character against training step, with dashed lines for the n-gram models with n = 1, 3 and 5. ${rest}.`;
    if (step === 2) return `Bars of the probability of each of the ${A} possible next characters after your text, with dashed outlines at temperature 1. ${rest}.`;
    if (step === 3) return `Two stacked bars, for likes and lives sentences: the probability that the sampled sentence is true, false, or anything else. ${rest}.`;
    if (step === 4) return `Bars of the number of weights and biases in each part of the model. ${rest}.`;
    return `Three boxes in a row: pretraining, then instruction tuning, then preference tuning. ${rest}.`;
  }

  const spoken = () => `${readout(step, info()).filter(Boolean).join('. ')}.`;

  // ---- the live run (step 2).
  const loop = createTrainLoop({
    step: (n) => getLive().work(n),
    render,
    done: () => getLive().done(),
    onStop: () => {
      buttons.sync();
      render();
      if (getLive().done()) announce(`Finished. ${spoken()}`);
    },
  });
  const buttons = trainButtons(trainGroup, loop, {
    onReset: () => getLive().reset(),
    reduced,
    work: (m) => getLive().work(m),
  });
  trainGroup.addEventListener('click', () => { if (step === 1) { render(); announce(spoken()); } });

  tauIn.addEventListener('input', () => {
    tauIx = Number(tauIn.value) || 0;
    render();
    announce(spoken());
  });
  contBtn.addEventListener('click', () => {
    const ps = promptState(typed);
    if (!ps.ids.length) { announce('Type a letter first: there is nothing to continue.'); return; }
    if (Array.from(ps.text).length >= MAX_INPUT) { announce(`The text is at its ${MAX_INPUT}-character limit.`); return; }
    typed = ps.text + continueOne(model(), ps.ids, tau(), contRng);
    input.value = typed;
    render();
    announce(spoken());
  });
  factBtn.addEventListener('click', () => {
    const key = `${iterNow()}:${tauIx}`;
    const list = writeFacts(model(), tau(), factSeed);
    factSeed += 1;
    const c = factCounts(list);
    const prev = facts && facts.key === key ? facts.totals : { true: 0, false: 0, other: 0 };
    facts = { key, list, totals: { true: prev.true + c.true, false: prev.false + c.false, other: prev.other + c.other } };
    render();
    announce(`${c.true} true, ${c.false} false, ${c.other} other.`);
  });
  input.addEventListener('input', () => {
    // maxlength is the first cap; sanitise caps again (in characters) and lower-cases.
    typed = input.value.slice(0, MAX_INPUT * 2);
    render();
  });
  input.addEventListener('change', () => announce(spoken()));

  const filled = pageValues();
  for (const span of doc.querySelectorAll('main.lesson [data-value]')) {
    const val = filled[span.dataset.value];
    if (val !== undefined) span.textContent = val;
  }

  render();
  if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(render, () => {});

  return {
    count: 6,
    show(i) {
      buttons.stop();
      step = i;
      const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };
      trainGroup.style.display = i === 1 ? 'flex' : 'none';
      controls.classList.toggle('has-train', i === 1);
      showGroup(tauGroup, i === 2 || i === 3);
      showGroup(contGroup, i === 2);
      showGroup(factGroup, i === 3);
      render();
      announce(spoken());
      return Promise.resolve();
    },
  };
}
