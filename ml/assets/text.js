// Text numerics for the Level 5 lessons: sanitising typed text, character tokens,
// byte-pair merging, character n-gram models, softmax with temperature, sampling.
// Pure functions on strings and plain arrays. No DOM.
import {mulberry32} from './maths.js';
import {ALPHABET} from './world.js';

const ID = new Map(ALPHABET.map((c, i) => [c, i]));
const chars = s => Array.from(s);   // code points, so an emoji is one character

// Lower-case, cap the length, keep characters as typed. `unknown` lists the indices
// (in characters) of anything outside ALPHABET.
export function sanitise(s, max = 80) {
  const cs = chars(String(s).toLowerCase()).slice(0, max);
  const unknown = [];
  cs.forEach((c, i) => { if (!ID.has(c)) unknown.push(i); });
  return {text: cs.join(''), unknown};
}

// Ids in ALPHABET, -1 for a character outside it.
export const charTokenise = s => chars(s).map(c => (ID.has(c) ? ID.get(c) : -1));

// ---- Byte-pair merging --------------------------------------------------------------

const isBreak = t => t === ' ' || t === '.';

function applyMerge(tokens, a, b) {
  const out = [];
  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i] === a && tokens[i + 1] === b) { out.push(a + b); i++; }
    else out.push(tokens[i]);
  }
  return out;
}

// Greedy merges. Spaces and full stops stay single tokens, so a pair never spans two
// words. Each round the most frequent adjacent pair wins; a tie goes to the pair that
// occurs first in the text. Counts are adjacent occurrences (overlaps counted). Stops
// early when no pair occurs twice. lengths[r] is the token count after r merges.
export function bpeTrain(text, nMerges) {
  let tokens = chars(text);
  const present = new Set(tokens);
  const vocab = ALPHABET.filter(c => present.has(c));
  for (const c of tokens) if (!ID.has(c) && !vocab.includes(c)) vocab.push(c);
  const merges = [], lengths = [tokens.length];
  for (let r = 0; r < nMerges; r++) {
    const stat = new Map();   // 'a\u0000b' -> {a, b, count, first}
    for (let i = 0; i + 1 < tokens.length; i++) {
      const a = tokens[i], b = tokens[i + 1];
      if (isBreak(a) || isBreak(b)) continue;
      const key = a + '\u0000' + b;
      const s = stat.get(key);
      if (s) s.count++; else stat.set(key, {a, b, count: 1, first: i});
    }
    let best = null;
    for (const s of stat.values()) {
      if (!best || s.count > best.count || (s.count === best.count && s.first < best.first)) best = s;
    }
    if (!best || best.count < 2) break;
    merges.push([best.a, best.b, best.count]);
    vocab.push(best.a + best.b);
    tokens = applyMerge(tokens, best.a, best.b);
    lengths.push(tokens.length);
  }
  return {merges, vocab, lengths};
}

export function bpeEncode(s, merges) {
  let tokens = chars(s);
  for (const [a, b] of merges) tokens = applyMerge(tokens, a, b);
  return tokens;
}

export const bpeDecode = tokens => tokens.join('');

// ---- Character n-gram model ---------------------------------------------------------

// n = context length in characters (1 to 5). Counts the next character after each
// context, over ALPHABET. Characters outside ALPHABET in `text` are dropped first.
export function ngramModel(text, n, {k = 0.01} = {}) {
  const ids = charTokenise(text).filter(i => i >= 0);
  const counts = new Map();
  for (let i = n; i < ids.length; i++) {
    const ctx = ids.slice(i - n, i).join(',');
    let row = counts.get(ctx);
    if (!row) { row = new Array(ALPHABET.length).fill(0); counts.set(ctx, row); }
    row[ids[i]]++;
  }
  return {n, k, counts};
}

// Probabilities of the next character, using the last n known characters of `context`
// (unknown characters are skipped). A context never seen in training (or shorter than n
// known characters) has all-zero counts and gives the uniform distribution.
export function nextProbs(model, context) {
  const A = ALPHABET.length;
  const known = charTokenise(context).filter(i => i >= 0);
  const row = known.length >= model.n ? model.counts.get(known.slice(known.length - model.n).join(',')) : undefined;
  if (!row) return new Array(A).fill(1 / A);
  const total = row.reduce((s, c) => s + c, 0) + model.k * A;
  return row.map(c => (c + model.k) / total);
}

// Mean -log2 p(next character) over positions n .. len-1 of `text`.
export function crossEntropyBits(model, text) {
  const ids = charTokenise(text).filter(i => i >= 0), n = model.n;
  let sum = 0, count = 0;
  for (let i = n; i < ids.length; i++) {
    const p = nextProbs(model, ids.slice(i - n, i).map(j => ALPHABET[j]).join(''));
    sum -= Math.log2(p[ids[i]]);
    count++;
  }
  return count ? sum / count : 0;
}

// Share of those positions whose context never occurred in training.
export function unseenContextShare(model, text) {
  const ids = charTokenise(text).filter(i => i >= 0), n = model.n;
  let unseen = 0, count = 0;
  for (let i = n; i < ids.length; i++) {
    if (!model.counts.has(ids.slice(i - n, i).join(','))) unseen++;
    count++;
  }
  return count ? unseen / count : 0;
}

// ---- Softmax, sampling, generation --------------------------------------------------

// Max-subtracted. For tau < 1e-3 it is the argmax, with exact ties split evenly.
export function softmax(z, tau = 1) {
  const m = Math.max(...z);
  if (tau < 1e-3) {
    const hits = z.filter(v => v === m).length;
    return z.map(v => (v === m ? 1 / hits : 0));
  }
  const e = z.map(v => Math.exp((v - m) / tau));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / s);
}

// Inverse CDF. If rounding leaves u above the total, the last index with p > 0.
export function sampleIndex(p, rng) {
  const u = rng();
  let c = 0, last = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] > 0) last = i;
    c += p[i];
    if (u < c) return i;
  }
  return last;
}

// start plus `length` generated characters. Each step: nextProbs, reweight p^(1/tau)
// (normalised; tau near 0 is the argmax), sample.
export function generateNgram(model, {start, length, tau, seed}) {
  const rng = mulberry32(seed);
  let out = start;
  for (let i = 0; i < length; i++) {
    const p = nextProbs(model, out);
    const q = softmax(p.map(v => Math.log(v)), tau);
    out += ALPHABET[sampleIndex(q, rng)];
  }
  return out;
}
