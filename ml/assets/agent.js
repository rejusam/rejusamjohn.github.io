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

// ---- The stand-in model, the loop and the tasks ----
// A trace is a list of messages: {role, text, step, rule?, tool?, input?, unseen?}.
// The stand-in reads each task's structured goal, never its English text, and the
// visible messages and notes. It applies the first rule in RULES that fits.

export const TASKS = {
  'mia-food': {id: 'mia-food', text: 'What food does mia like?', goal: {kind: 'food', person: 'mia'}},
  sum: {id: 'sum', text: 'What is 3 × 1.25 + 2 × 0.8?', goal: {kind: 'sum', terms: [[3, 1.25], [2, 0.8]]}},
  'eli-shop': {id: 'eli-shop', text: 'eli buys 4 of the food eli likes. What does that cost?',
    goal: {kind: 'shop', people: ['eli'], qty: 4}},
  'pair-shop': {id: 'pair-shop', text: 'ana and sam each buy 3 of the food they like. What do they pay in total?',
    goal: {kind: 'shop', people: ['ana', 'sam'], qty: 3}},
  'kai-food': {id: 'kai-food', text: 'What food does kai like?', goal: {kind: 'food', person: 'kai'}},
  'tell-zoe': {id: 'tell-zoe', text: 'Tell zoe what pears cost.', goal: {kind: 'send', to: 'zoe', food: 'pears'}},
};

export const RULES = [
  {id: 'R1', text: 'If the last result says "not found": answer that it was not found, or call the same lookup again, or guess (the first food in the food table or the first price in the price list) and go on to the next rule, as set.'},
  {id: 'R2', text: 'If the notepad is on and the last result is a fact not yet in the notes: write it in the notes.'},
  {id: 'R3', text: 'If the task needs a fact that is not in the messages or the notes: call lookup for it. With lookup off, guess the first food in the food table or the first price in the price list, then go on to the next rule.'},
  {id: 'R4', text: 'If the task needs a sum and no calculator result for it is in the messages: call the calculator. With the calculator off, or sums set to guess, round each number to one significant figure, work out the sum exactly, then go on.'},
  {id: 'R5', text: 'If the task asks to send a message and none has been sent: with send off, answer that it cannot send; with ask before changes on, ask a person first; otherwise call send.'},
  {id: 'R6', text: 'Otherwise: answer from the facts, and the sum if there is one.'},
];

export const DEFAULT_CONFIG = {
  tools: {calculator: true, lookup: true, notepad: false, send: true},
  stepLimit: 8, budget: Infinity, repeatCheck: true, askBeforeChanges: true,
  onNotFound: 'say so', sums: 'calculator',
};

function withDefaults(config) {
  const c = config || {};
  return {...DEFAULT_CONFIG, ...c, tools: {...DEFAULT_CONFIG.tools, ...(c.tools || {})}};
}

// Round to one significant figure, half away from zero. Small positive numbers only.
export function roundOneFigure(x) {
  if (x === 0) return 0;
  const e = Math.floor(Math.log10(Math.abs(x)));
  if (e < 0) {
    const s = 10 ** (-e);
    return Math.floor(Math.abs(x) * s + 0.5) / s;
  }
  const s = 10 ** e;
  return Math.floor(Math.abs(x) / s + 0.5) * s;
}

export function callText(tool, input) {
  const parts = Object.keys(input).map(k => '"' + input[k] + '"');
  return tool + '(' + parts.join(', ') + ')';
}

const LIKES_RE = /^([a-z]+) likes ([a-z]+) /;
const PRICE_RE = /^([a-z]+) cost \$(\d+\.\d\d) each\.$/;

// The facts the stand-in can read: lookup results in the visible messages, and the notes.
function readFacts(visible, notes) {
  const likes = {};
  const price = {};
  const texts = [];
  for (const m of visible) if (m.role === 'result' && m.tool === 'lookup') texts.push(m.text);
  for (const n of notes) texts.push(n);
  for (const t of texts) {
    let m = LIKES_RE.exec(t);
    if (m) likes[m[1]] = m[2];
    m = PRICE_RE.exec(t);
    if (m) price[m[1]] = Number(m[2]);
  }
  return {likes, price};
}

function lastResult(visible) {
  for (let i = visible.length - 1; i >= 0; i--) if (visible[i].role === 'result') return visible[i];
  return null;
}

function sumTerms(goal, facts) {
  if (goal.kind === 'sum') return goal.terms;
  if (goal.kind === 'shop') return goal.people.map(p => [goal.qty, facts.price[facts.likes[p]]]);
  return null;
}

const termsExpr = terms => terms.map(t => fmtNumber(t[0]) + '*' + fmtNumber(t[1])).join('+');

export function standIn(view, config) {
  const cfg = withDefaults(config);
  const tools = cfg.tools;
  const goal = view.task.goal;
  const visible = view.visible;
  const facts = readFacts(visible, view.notes);
  const plan = [];
  const rules = [];
  const use = id => { if (!rules.includes(id)) rules.push(id); };
  const finish = (action, rule) => { use(rule); return {plan: plan.join(' '), action, rule, rules}; };
  const last = lastResult(visible);
  const firstFood = WORLD.foods[0];
  const firstPrice = PRICES[Object.keys(PRICES)[0]];

  // R1: the last result says "not found".
  if (last && last.tool === 'lookup' && last.text.startsWith('not found: ')) {
    const key = last.input.key;
    if (cfg.onNotFound === 'say so') {
      plan.push(`Lookup found nothing for ${key}; say so.`);
      return finish({type: 'answer', text: `I could not find ${key}.`}, 'R1');
    }
    if (cfg.onNotFound === 'try again') {
      plan.push(`Lookup found nothing for ${key}; try the same lookup again.`);
      return finish({type: 'call', tool: 'lookup', input: {key}}, 'R1');
    }
    // guess: fill the gap the same way R3 does with lookup off, then go on.
    use('R1');
    if (WORLD.foods.includes(key)) {
      facts.price[key] = firstPrice;
      plan.push(`Guess (first price in the list): ${key} cost ${fmtMoney(firstPrice)}.`);
    } else {
      facts.likes[key] = firstFood;
      plan.push(`Guess (first food in the table): ${key} likes ${firstFood}.`);
    }
  }

  // R2: write the last lookup fact in the notes.
  if (tools.notepad && last && last.tool === 'lookup' && !last.text.startsWith('not found: ') &&
      !view.notes.includes(last.text)) {
    plan.push(`Keep the fact about ${last.input.key}; use notepad.`);
    return finish({type: 'call', tool: 'notepad', input: {action: 'write', text: last.text}}, 'R2');
  }

  // R3: the facts the task needs, in order.
  const needed = [];
  if (goal.kind === 'food') needed.push(['likes', goal.person]);
  if (goal.kind === 'send') needed.push(['price', goal.food]);
  if (goal.kind === 'shop') for (const p of goal.people) needed.push(['likes', p], ['price', null, p]);
  for (const n of needed) {
    const kind = n[0];
    const key = kind === 'price' && n[1] === null ? facts.likes[n[2]] : n[1];
    const have = kind === 'likes' ? key in facts.likes : key in facts.price;
    if (have) continue;
    if (tools.lookup) {
      plan.push(kind === 'likes' ? `Need the food ${key} likes; use lookup.` : `Need the price of ${key}; use lookup.`);
      return finish({type: 'call', tool: 'lookup', input: {key}}, 'R3');
    }
    use('R3');
    if (kind === 'likes') {
      facts.likes[key] = firstFood;
      plan.push(`Guess (first food in the table): ${key} likes ${firstFood}.`);
    } else {
      facts.price[key] = firstPrice;
      plan.push(`Guess (first price in the list): ${key} cost ${fmtMoney(firstPrice)}.`);
    }
  }

  // R4: the sum.
  let total = null;
  const terms = sumTerms(goal, facts);
  if (terms) {
    const expr = termsExpr(terms);
    for (const m of visible) {
      if (m.role === 'result' && m.tool === 'calculator' && m.input.expr === expr) total = Number(m.text);
    }
    if (total === null) {
      if (tools.calculator && cfg.sums === 'calculator') {
        plan.push(`Need ${expr}; use calculator.`);
        return finish({type: 'call', tool: 'calculator', input: {expr}}, 'R4');
      }
      use('R4');
      const rounded = terms.map(t => [roundOneFigure(t[0]), roundOneFigure(t[1])]);
      total = 0;
      for (const t of rounded) total += t[0] * t[1];
      plan.push(`Guess (each number rounded to one significant figure): ${termsExpr(rounded)} = ${fmtNumber(total)}.`);
    }
  }

  // R5: send.
  if (goal.kind === 'send') {
    let sent = false;
    for (const m of visible) if (m.role === 'result' && m.tool === 'send') sent = true;
    const text = `${goal.food} cost ${fmtMoney(facts.price[goal.food])} each.`;
    if (!sent) {
      if (!tools.send) {
        plan.push('Send is off; say so.');
        return finish({type: 'answer', text: `I cannot send messages, so ${goal.to} was not told.`}, 'R5');
      }
      const input = {to: goal.to, text};
      if (cfg.askBeforeChanges) {
        plan.push('The message is ready; ask before using send.');
        return finish({type: 'ask', tool: 'send', input}, 'R5');
      }
      plan.push('The message is ready; use send.');
      return finish({type: 'call', tool: 'send', input}, 'R5');
    }
    plan.push('All done; answer.');
    return finish({type: 'answer', text: `Told ${goal.to}: ${text}`}, 'R6');
  }

  // R6: answer.
  plan.push('All facts are in; answer.');
  let answer;
  if (goal.kind === 'food') answer = `${goal.person} likes ${facts.likes[goal.person]}.`;
  else if (goal.kind === 'sum') answer = `The sum is ${fmtNumber(total)}.`;
  else answer = `That costs ${fmtMoney(total)}.`;
  return finish({type: 'answer', text: answer}, 'R6');
}

// The task message is always kept. Then the newest messages, going back, while the
// running word total of the kept non-task messages stays within the budget. The first
// message that does not fit, and everything older, is not seen.
export function contextWindow(messages, budget) {
  const seen = messages.map(m => m.role === 'task');
  let total = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === 'task') continue;
    total += countWords(messages[i].text);
    if (total > budget) break;
    seen[i] = true;
  }
  return seen;
}

const sameCall = (a, b) => a !== null && a.tool === b.tool && callText(a.tool, a.input) === callText(b.tool, b.input);

// Each model call is one step. Tools, approval and waiting are not steps.
// Plan and action rows carry rule (the rule that chose the action) and rules (every rule
// applied in that model call, in order, guesses first). Each plan row carries seen: the
// indices of the messages that model call could see (its context window).
// After the run, messages outside the context window of the whole trace before the stop row
// carry unseen: true (the key is only present when true).
// opts.approve answers the first ask only; a later ask in the same run stops the run with
// 'waiting for approval'.
export function runAgent(taskId, config, opts) {
  const cfg = withDefaults(config);
  const o = opts || {};
  const tools = o.tools || TOOLS;
  const task = TASKS[taskId];
  const state = {notes: [], sent: []};
  const messages = [{role: 'task', text: task.text, step: 0}];
  let steps = 0;
  let prevCall = null;
  let answer = null;
  let stop = null;
  let asked = false;
  while (stop === null) {
    if (steps >= cfg.stepLimit) { stop = 'step limit'; break; }
    steps += 1;
    const seen = contextWindow(messages, cfg.budget);
    const visible = messages.filter((m, i) => seen[i]);
    const seenIdx = [];
    for (let i = 0; i < seen.length; i++) if (seen[i]) seenIdx.push(i);
    const out = standIn({task, visible, notes: state.notes.slice()}, cfg);
    messages.push({role: 'plan', text: out.plan, step: steps, rule: out.rule, rules: out.rules.slice(), seen: seenIdx});
    const a = out.action;
    if (a.type === 'answer') {
      messages.push({role: 'answer', text: a.text, step: steps, rule: out.rule, rules: out.rules.slice()});
      answer = a.text;
      stop = 'done';
      break;
    }
    let run = a;
    if (a.type === 'ask') {
      messages.push({role: 'approval', text: `Waiting for a yes before ${callText(a.tool, a.input)}`,
        step: steps, rule: out.rule, rules: out.rules.slice(), tool: a.tool, input: a.input});
      const firstAsk = !asked;
      asked = true;
      if (o.approve === undefined || !firstAsk) { stop = 'waiting for approval'; break; }
      if (o.approve === false) {
        messages.push({role: 'approval', text: 'No.', step: steps});
        stop = 'declined';
        break;
      }
      messages.push({role: 'approval', text: 'Yes.', step: steps});
    }
    const call = {role: 'call', text: callText(run.tool, run.input), step: steps, rule: out.rule,
      rules: out.rules.slice(), tool: run.tool, input: run.input};
    messages.push(call);
    if (cfg.repeatCheck && sameCall(prevCall, call)) { stop = 'repeated call'; break; }
    prevCall = call;
    const r = tools[run.tool].run(run.input, state);
    messages.push({role: 'result', text: r.text, step: steps, tool: run.tool, input: run.input});
    const notFound = run.tool === 'lookup' && r.text.startsWith('not found: ');
    if (!r.ok && !notFound) { stop = 'tool error'; break; }
  }
  const seen = contextWindow(messages, cfg.budget);
  for (let i = 0; i < messages.length; i++) if (!seen[i]) messages[i].unseen = true;
  messages.push({role: 'stop', text: stop, step: steps});
  return {messages, stop, answer, steps, state};
}

// One model call, no tools: only the guess rules can fill the gaps.
export function oneAnswer(taskId) {
  const task = TASKS[taskId];
  const cfg = withDefaults({tools: {calculator: false, lookup: false, notepad: false, send: false},
    sums: 'guess', onNotFound: 'guess'});
  const messages = [{role: 'task', text: task.text, step: 0}];
  const out = standIn({task, visible: messages.slice(), notes: []}, cfg);
  messages.push({role: 'plan', text: out.plan, step: 1, rule: out.rule, rules: out.rules.slice(), seen: [0]});
  messages.push({role: 'answer', text: out.action.text, step: 1, rule: out.rule, rules: out.rules.slice()});
  return {messages, answer: out.action.text};
}

export const chainSuccess = (p, k) => p ** k;

// Largest k with p**k >= target (0 < target <= 1). p = 1 gives Infinity.
export function stepsForTarget(p, target) {
  if (p >= 1) return Infinity;
  if (p <= 0) return 0;
  let k = Math.max(0, Math.floor(Math.log(target) / Math.log(p)));
  while (p ** (k + 1) >= target) k += 1;
  while (k > 0 && p ** k < target) k -= 1;
  return k;
}
