// Level 6 agent tools and formatting. The people, foods and prices are invented:
// they are rules of the page, not claims about anyone. Pure functions, no DOM.
// Each tool's run(input, state) depends only on its input and on state.notes / state.sent;
// the caller owns state, and the notepad and send tools create the arrays if they are missing.
import {WORLD} from './world.js';

export const PRICES = {apples: 0.8, pears: 1.1, plums: 1.25, figs: 2.4, limes: 0.6, grapes: 3.5};

const MINUS = '−';
const MAX_EXPR = 60;

// calc: recursive descent, never eval.
//   expr := term (('+'|'-') term)*   term := factor (('*'|'/'|'×'|'÷') factor)*
//   factor := '-' factor | number | '(' expr ')'
function tokenise(s) {
  const out = [];
  let i = 0;
  while (i < s.length) {
    const c = s[i];
    if (c === ' ') { i++; continue; }
    const m = /^\d+(\.\d+)?/.exec(s.slice(i));
    if (m) { out.push({n: Number(m[0])}); i += m[0].length; continue; }
    if ('+-*/×÷()'.includes(c)) { out.push({op: c === '×' ? '*' : c === '÷' ? '/' : c}); i++; continue; }
    return null;
  }
  return out;
}

export function calc(expr) {
  const bad = {ok: false, error: `cannot read "${expr}"`};
  if (typeof expr !== 'string' || expr.length === 0 || expr.length > MAX_EXPR) return bad;
  const toks = tokenise(expr);
  if (!toks || toks.length === 0) return bad;
  let pos = 0;
  const fail = {};
  const peek = () => toks[pos];
  const isOp = (t, ...ops) => t && t.op !== undefined && ops.includes(t.op);

  function parseExpr() {
    let v = parseTerm();
    while (isOp(peek(), '+', '-')) {
      const op = toks[pos++].op;
      const r = parseTerm();
      v = op === '+' ? v + r : v - r;
    }
    return v;
  }
  function parseTerm() {
    let v = parseFactor();
    while (isOp(peek(), '*', '/')) {
      const op = toks[pos++].op;
      const r = parseFactor();
      if (op === '/' && r === 0) throw new Error('division by zero');
      v = op === '*' ? v * r : v / r;
    }
    return v;
  }
  function parseFactor() {
    const t = toks[pos++];
    if (!t) throw fail;
    if (t.n !== undefined) return t.n;
    if (t.op === '-') return -parseFactor();
    if (t.op === '(') {
      const v = parseExpr();
      if (!isOp(toks[pos++], ')')) throw fail;
      return v;
    }
    throw fail;
  }

  try {
    const value = parseExpr();
    if (pos !== toks.length) return bad;
    return {ok: true, value};
  } catch (e) {
    if (e === fail) return bad;
    if (e instanceof Error && e.message === 'division by zero') return {ok: false, error: e.message};
    throw e;
  }
}

// Rounding rule (a Python port reproduces it exactly): r = floor(abs(x) * 100 + 0.5) / 100,
// i.e. half away from zero on the double x * 100. Then print r with two decimals and drop
// trailing zeros. A negative that rounds to zero prints "0", never a minus sign.
const roundCents = x => Math.floor(Math.abs(x) * 100 + 0.5) / 100;

export function fmtNumber(x) {
  const r = roundCents(x);
  const s = r.toFixed(2).replace(/\.?0+$/, '');
  return x < 0 && r !== 0 ? MINUS + s : s;
}

export function fmtMoney(x) {
  const r = roundCents(x);
  return (x < 0 && r !== 0 ? MINUS : '') + '$' + r.toFixed(2);
}

export function countWords(text) {
  const t = String(text).trim();
  return t === '' ? 0 : t.split(/\s+/).length;
}

export function lookupText(key) {
  if (WORLD.people.includes(key)) {
    const pl = WORLD.livesIn[key];
    return `${key} likes ${WORLD.likes[key]} and lives ${WORLD.prep[pl]} the ${pl}.`;
  }
  if (WORLD.foods.includes(key)) return `${key} cost ${fmtMoney(PRICES[key])} each.`;
  return `not found: ${key}`;
}

export const TOOLS = {
  calculator: {
    name: 'calculator', kind: 'reads',
    about: 'Works out an arithmetic sum such as 3*0.8+2.',
    run(input) {
      const r = calc(input.expr);
      return r.ok ? {ok: true, text: fmtNumber(r.value), value: r.value} : {ok: false, text: r.error};
    },
  },
  lookup: {
    name: 'lookup', kind: 'reads',
    about: 'Looks up one person or one food in the invented world.',
    run(input) {
      const text = lookupText(input.key);
      return {ok: !text.startsWith('not found'), text};
    },
  },
  notepad: {
    name: 'notepad', kind: 'reads',
    about: 'Writes a line to the notes, or reads all the notes back.',
    run(input, state) {
      if (!state.notes) state.notes = [];
      if (input.action === 'write') {
        state.notes.push(input.text);
        return {ok: true, text: 'noted'};
      }
      if (input.action === 'read') {
        const value = state.notes.slice();
        return {ok: true, text: value.length ? value.join('\n') : 'notes are empty', value};
      }
      return {ok: false, text: `cannot read action "${input.action}"`};
    },
  },
  send: {
    name: 'send', kind: 'changes things',
    about: 'Sends a message to one person. Here it only records the message.',
    run(input, state) {
      if (!state.sent) state.sent = [];
      state.sent.push({to: input.to, text: input.text});
      return {ok: true, text: `sent to ${input.to}`};
    },
  },
};
