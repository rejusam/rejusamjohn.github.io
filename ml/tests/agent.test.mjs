import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {WORLD} from '../assets/world.js';
import {PRICES, calc, fmtNumber, fmtMoney, countWords, lookupText, TOOLS} from '../assets/agent.js';

test('calc follows arithmetic', () => {
  const table = [['2+3*4', 14], ['(2+3)*4', 20], ['-3+5', 2], ['4*1.25', 5],
    ['3*0.8+3*3.5', 3 * 0.8 + 3 * 3.5], ['10/4', 2.5], ['2 × 3', 6], ['9 ÷ 3', 3],
    ['2*(3+(4-1))', 12]];
  for (const [e, v] of table) assert.deepEqual(calc(e), {ok: true, value: v}, e);
});

test('calc refuses what it cannot read', () => {
  assert.deepEqual(calc('1/0'), {ok: false, error: 'division by zero'});
  for (const e of ['2+', 'abc', '2**3', '', '1+'.repeat(30) + '1']) {
    const r = calc(e);
    assert.equal(r.ok, false, e);
    assert.ok(r.error.startsWith('cannot read'), e);
  }
});

test('calc never uses eval', () => {
  const src = readFileSync(new URL('../assets/agent.js', import.meta.url), 'utf8');
  assert.ok(!src.includes('eval('));
  assert.ok(!src.includes('Function('));
});

test('fmtNumber and fmtMoney never show float tails', () => {
  assert.equal(fmtNumber(3 * 0.8), '2.4');
  assert.equal(fmtNumber(5), '5');
  assert.equal(fmtNumber(-0.75), '−0.75');
  assert.equal(fmtNumber(-0.001), '0');
  assert.equal(fmtMoney(3 * 0.8), '$2.40');
  assert.equal(fmtMoney(12.9), '$12.90');
  assert.equal(fmtMoney(5), '$5.00');
});

test('countWords counts whitespace-separated pieces', () => {
  assert.equal(countWords('mia likes figs.'), 3);
  assert.equal(countWords('  a  b '), 2);
  assert.equal(countWords(''), 0);
});

test('lookup reads the invented world', () => {
  assert.equal(lookupText('mia'), 'mia likes figs and lives in the town.');
  assert.equal(lookupText('plums'), 'plums cost $1.25 each.');
  assert.equal(lookupText('kai'), 'not found: kai');
  for (const p of WORLD.people) {
    const pl = WORLD.livesIn[p];
    assert.equal(lookupText(p), `${p} likes ${WORLD.likes[p]} and lives ${WORLD.prep[pl]} the ${pl}.`);
  }
  for (const f of WORLD.foods) assert.equal(lookupText(f), `${f} cost ${fmtMoney(PRICES[f])} each.`);
  assert.deepEqual(Object.keys(PRICES), WORLD.foods);
});

test('tools: calculator and lookup run on input', () => {
  assert.deepEqual(TOOLS.calculator.run({expr: '3*0.8'}, {}), {ok: true, text: '2.4', value: 3 * 0.8});
  const bad = TOOLS.calculator.run({expr: '1/0'}, {});
  assert.equal(bad.ok, false);
  assert.equal(bad.text, 'division by zero');
  assert.deepEqual(TOOLS.lookup.run({key: 'mia'}, {}), {ok: true, text: 'mia likes figs and lives in the town.'});
  assert.equal(TOOLS.lookup.run({key: 'kai'}, {}).ok, false);
});

test('notepad keeps what is written', () => {
  const state = {notes: [], sent: []};
  assert.equal(TOOLS.notepad.run({action: 'read'}, state).text, 'notes are empty');
  TOOLS.notepad.run({action: 'write', text: 'first'}, state);
  TOOLS.notepad.run({action: 'write', text: 'second'}, state);
  assert.deepEqual(state.notes, ['first', 'second']);
  const r = TOOLS.notepad.run({action: 'read'}, state);
  assert.deepEqual(r, {ok: true, text: 'first\nsecond', value: ['first', 'second']});
  // state arrays are created when missing
  const bare = {};
  TOOLS.notepad.run({action: 'write', text: 'x'}, bare);
  assert.deepEqual(bare.notes, ['x']);
  assert.equal(TOOLS.notepad.run({action: 'nope'}, bare).ok, false);
});

test('send only records', () => {
  const state = {notes: [], sent: []};
  const r = TOOLS.send.run({to: 'zoe', text: 'hi'}, state);
  assert.equal(r.ok, true);
  assert.deepEqual(state.sent, [{to: 'zoe', text: 'hi'}]);
  assert.deepEqual(state.notes, []);
  assert.equal(TOOLS.send.kind, 'changes things');
  for (const k of ['calculator', 'lookup', 'notepad']) assert.equal(TOOLS[k].kind, 'reads');
  for (const [k, t] of Object.entries(TOOLS)) {
    assert.equal(t.name, k);
    assert.equal(typeof t.about, 'string');
  }
});

// ---- Task 2: stand-in model, loop and tasks ----
import {TASKS, RULES, standIn, DEFAULT_CONFIG, runAgent, oneAnswer, contextWindow,
  chainSuccess, stepsForTarget} from '../assets/agent.js';

const cfg = over => ({...DEFAULT_CONFIG, ...over, tools: {...DEFAULT_CONFIG.tools, ...(over && over.tools)}});
const calls = r => r.messages.filter(m => m.role === 'call').map(m => `${m.tool} ${Object.values(m.input).join(' ')}`);

test('tasks and rules are fixed', () => {
  assert.deepEqual(Object.keys(TASKS), ['mia-food', 'sum', 'eli-shop', 'pair-shop', 'kai-food', 'tell-zoe']);
  assert.equal(TASKS['mia-food'].text, 'What food does mia like?');
  assert.equal(TASKS.sum.text, 'What is 3 × 1.25 + 2 × 0.8?');
  assert.equal(TASKS['eli-shop'].text, 'eli buys 4 of the food eli likes. What does that cost?');
  assert.equal(TASKS['pair-shop'].text, 'ana and sam each buy 3 of the food they like. What do they pay in total?');
  assert.equal(TASKS['kai-food'].text, 'What food does kai like?');
  assert.equal(TASKS['tell-zoe'].text, 'Tell zoe what pears cost.');
  for (const [id, t] of Object.entries(TASKS)) { assert.equal(t.id, id); assert.equal(typeof t.goal, 'object'); }
  assert.deepEqual(RULES.map(r => r.id), ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']);
  const banned = /\b(thinks?|knows?|understands?|wants?|decides?|reasons?)\b/i;
  for (const r of RULES) assert.ok(!banned.test(r.text), r.text);
});

test('the stand-in never reads the English task text', () => {
  const t = TASKS['eli-shop'];
  const a = standIn({task: t, visible: [], notes: []}, cfg());
  const b = standIn({task: {...t, text: 'something else entirely'}, visible: [], notes: []}, cfg());
  assert.deepEqual(a, b);
});

test('runAgent is deterministic', () => {
  for (const id of Object.keys(TASKS)) {
    assert.deepEqual(runAgent(id, cfg()), runAgent(id, cfg()), id);
    assert.deepEqual(runAgent(id, cfg({budget: 12, tools: {notepad: true}}), {approve: true}),
      runAgent(id, cfg({budget: 12, tools: {notepad: true}}), {approve: true}), id);
  }
});

test('eli-shop with defaults', () => {
  const r = runAgent('eli-shop', cfg());
  assert.equal(r.stop, 'done');
  assert.equal(r.steps, 4);
  assert.deepEqual(calls(r), ['lookup eli', 'lookup plums', 'calculator 4*1.25']);
  assert.ok(r.answer.includes('$5.00'));
  assert.equal(r.answer, 'That costs $5.00.');
  // plan row and action row share a step
  for (const m of r.messages.filter(m => m.role === 'plan')) {
    assert.ok(r.messages.some(x => x !== m && x.step === m.step && (x.role === 'call' || x.role === 'answer')));
  }
});

test('mia-food is one lookup', () => {
  const r = runAgent('mia-food', cfg());
  assert.equal(r.stop, 'done');
  assert.equal(r.steps, 2);
  assert.equal(r.answer, 'mia likes figs.');
});

test('sum without a calculator guesses', () => {
  const truth = calc('3*1.25+2*0.8').value;
  const guess = 3 * 1 + 2 * 0.8; // each number to one significant figure
  const err = truth - guess;
  for (const over of [{sums: 'guess'}, {tools: {calculator: false}}]) {
    const r = runAgent('sum', cfg(over));
    assert.equal(r.stop, 'done');
    assert.equal(r.steps, 1);
    assert.equal(r.answer, `The sum is ${fmtNumber(guess)}.`);
    assert.ok(!calls(r).some(c => c.startsWith('calculator')));
  }
  assert.equal(fmtNumber(guess), '4.6');
  assert.equal(fmtNumber(truth), '5.35');
  assert.equal(fmtNumber(err), '0.75');
  const r = runAgent('sum', cfg());
  assert.equal(r.answer, 'The sum is 5.35.');
  assert.equal(r.steps, 2);
});

test('one answer guesses', () => {
  const e = oneAnswer('eli-shop');
  assert.equal(e.messages.filter(m => m.role === 'plan').length, 1);
  assert.ok(e.messages.every(m => m.role !== 'call'));
  assert.equal(e.answer, `That costs ${fmtMoney(4 * PRICES[WORLD.foods[0]])}.`);
  assert.equal(e.answer, 'That costs $3.20.');
  const plan = e.messages.find(m => m.role === 'plan').text;
  assert.ok(plan.includes('apples') && plan.includes('$0.80'));
  const m = oneAnswer('mia-food');
  assert.equal(m.answer, 'mia likes apples.');
  assert.notEqual(WORLD.likes.mia, 'apples');
});

test('step limit stops without an answer', () => {
  const a = runAgent('eli-shop', cfg({stepLimit: 3}));
  assert.equal(a.stop, 'step limit');
  assert.equal(a.answer, null);
  assert.equal(a.steps, 3);
  const b = runAgent('eli-shop', cfg({stepLimit: 4}));
  assert.equal(b.stop, 'done');
});

test('kai-food by onNotFound', () => {
  const s = runAgent('kai-food', cfg({onNotFound: 'say so'}));
  assert.equal(s.stop, 'done');
  assert.equal(s.steps, 2);
  assert.equal(s.answer, 'I could not find kai.');
  const t = runAgent('kai-food', cfg({onNotFound: 'try again'}));
  assert.equal(t.stop, 'repeated call');
  assert.equal(t.steps, 2);
  assert.deepEqual(calls(t), ['lookup kai', 'lookup kai']);
  // the repeated call is not run: only one result row
  assert.equal(t.messages.filter(m => m.role === 'result').length, 1);
  const u = runAgent('kai-food', cfg({onNotFound: 'try again', repeatCheck: false, stepLimit: 6}));
  assert.equal(u.stop, 'step limit');
  assert.equal(u.steps, 6);
  assert.equal(u.answer, null);
  const g = runAgent('kai-food', cfg({onNotFound: 'guess'}));
  assert.equal(g.stop, 'done');
  assert.equal(g.steps, 2);
  assert.equal(g.answer, `kai likes ${WORLD.foods[0]}.`);
});

test('pair-shop needs memory', () => {
  const a = runAgent('pair-shop', cfg());
  assert.equal(a.stop, 'done');
  assert.equal(a.steps, 6);
  assert.equal(a.answer, `That costs ${fmtMoney(calc('3*0.8+3*3.5').value)}.`);
  assert.equal(a.answer, 'That costs $12.90.');
  assert.deepEqual(calls(a), ['lookup ana', 'lookup apples', 'lookup sam', 'lookup grapes', 'calculator 3*0.8+3*3.5']);
  const b = runAgent('pair-shop', cfg({budget: 12, stepLimit: 10}));
  assert.equal(b.stop, 'step limit');
  assert.equal(b.answer, null);
  const lk = calls(b).filter(c => c.startsWith('lookup'));
  assert.ok(lk.length > new Set(lk).size, 'some lookup is called twice');
  assert.ok(b.messages.some(m => m.unseen));
  const c = runAgent('pair-shop', cfg({budget: 12, stepLimit: 10, tools: {notepad: true}}));
  assert.equal(c.stop, 'done');
  assert.equal(c.steps, 10);
  assert.equal(c.answer, 'That costs $12.90.');
  assert.equal(c.state.notes.length, 4);
  assert.equal(runAgent('pair-shop', cfg({budget: 12, stepLimit: 9, tools: {notepad: true}})).stop, 'step limit');
});

test('context window fits the budget and keeps the task', () => {
  const trace = runAgent('pair-shop', cfg()).messages;
  for (const budget of [0, 5, 12, 40, Infinity]) {
    const seen = contextWindow(trace, budget);
    assert.equal(seen.length, trace.length);
    assert.equal(seen[0], true);
    const words = trace.reduce((s, m, i) => s + (i > 0 && seen[i] ? countWords(m.text) : 0), 0);
    assert.ok(words <= budget, String(budget));
    const first = seen.indexOf(true, 1);
    if (first !== -1) for (let i = first; i < seen.length; i++) assert.equal(seen[i], true, `suffix ${budget}`);
    // the next older message would not fit
    if (first > 1) assert.ok(words + countWords(trace[first - 1].text) > budget);
    // nothing kept: the newest message alone is over the budget
    if (first === -1) assert.ok(countWords(trace[trace.length - 1].text) > budget);
  }
  assert.ok(contextWindow(trace, Infinity).every(Boolean));
});

test('approval does not use a step', () => {
  const w = runAgent('tell-zoe', cfg());
  assert.equal(w.stop, 'waiting for approval');
  const askAt = w.messages.findIndex(m => m.role === 'approval');
  const plansBefore = w.messages.slice(0, askAt).filter(m => m.role === 'plan').length;
  assert.equal(w.steps, plansBefore); // every model call so far, including the one that wrote the ask
  assert.equal(w.steps, 2);
  assert.equal(w.messages.filter(m => m.role === 'plan').length, w.steps);
  assert.deepEqual(w.state.sent, []);
  const y = runAgent('tell-zoe', cfg(), {approve: true});
  assert.equal(y.stop, 'done');
  assert.equal(y.steps, 3);
  assert.equal(y.state.sent.length, 1);
  assert.equal(y.state.sent[0].to, 'zoe');
  assert.ok(y.state.sent[0].text.includes('$1.10'));
  assert.equal(y.messages.filter(m => m.role === 'plan').length, y.steps);
  const n = runAgent('tell-zoe', cfg(), {approve: false});
  assert.equal(n.stop, 'declined');
  assert.deepEqual(n.state.sent, []);
  assert.equal(n.steps, 2);
  const off = runAgent('tell-zoe', cfg({askBeforeChanges: false}));
  assert.equal(off.stop, 'done');
  assert.equal(off.steps, 3);
  assert.ok(off.messages.every(m => m.role !== 'approval'));
  assert.equal(off.state.sent.length, 1);
  // boundary: approval at the last allowed step; send runs, the next model call would be step 3
  const edge = runAgent('tell-zoe', cfg({stepLimit: 2}), {approve: true});
  assert.equal(edge.stop, 'step limit');
  assert.equal(edge.steps, 2);
  assert.equal(edge.state.sent.length, 1);
  const noSend = runAgent('tell-zoe', cfg({tools: {send: false}}));
  assert.equal(noSend.stop, 'done');
  assert.equal(noSend.answer, 'I cannot send messages, so zoe was not told.');
});

test('a tool error stops the run', () => {
  const tools = {...TOOLS, calculator: {...TOOLS.calculator, run: () => ({ok: false, text: 'cannot read "2+"'})}};
  const r = runAgent('eli-shop', cfg(), {tools});
  assert.equal(r.stop, 'tool error');
  assert.equal(r.answer, null);
  assert.equal(r.steps, 3);
  // "not found" from lookup is not a tool error
  assert.notEqual(runAgent('kai-food', cfg()).stop, 'tool error');
});

test('tools off are never called', () => {
  const names = ['calculator', 'lookup', 'notepad', 'send'];
  for (const id of Object.keys(TASKS)) {
    for (let mask = 0; mask < 16; mask++) {
      const tools = {};
      names.forEach((n, i) => { tools[n] = Boolean(mask & (1 << i)); });
      for (const approve of [undefined, true, false]) {
        const r = runAgent(id, cfg({tools}), approve === undefined ? {} : {approve});
        for (const m of r.messages) if (m.role === 'call') assert.ok(tools[m.tool], `${id} ${mask} ${m.tool}`);
        assert.ok(['done', 'step limit', 'repeated call', 'waiting for approval', 'declined'].includes(r.stop));
      }
    }
  }
});

test('chainSuccess and stepsForTarget', () => {
  assert.equal(chainSuccess(0.95, 10), 0.95 ** 10);
  assert.ok(Math.abs(chainSuccess(0.95, 10) - 0.5987) < 1e-4);
  assert.equal(stepsForTarget(0.95, 0.5), 13);
  assert.equal(stepsForTarget(0.95, 0.5), Math.floor(Math.log(0.5) / Math.log(0.95)));
  assert.ok(0.95 ** 13 >= 0.5 && 0.95 ** 14 < 0.5);
  assert.equal(stepsForTarget(1, 0.5), Infinity);
  for (const p of [0.5, 0.8, 0.9, 0.99]) for (const t of [0.1, 0.5, 0.9]) {
    const k = stepsForTarget(p, t);
    assert.ok(p ** k >= t && p ** (k + 1) < t, `${p} ${t}`);
  }
});

test('every rule a step applies is recorded', () => {
  const g = runAgent('kai-food', cfg({onNotFound: 'guess'}));
  const plan = g.messages.filter(m => m.role === 'plan').pop();
  assert.deepEqual(plan.rules, ['R1', 'R6']);
  assert.equal(plan.rule, 'R6');
  assert.deepEqual(g.messages.find(m => m.role === 'answer').rules, ['R1', 'R6']);
  const e = runAgent('eli-shop', cfg({tools: {lookup: false}}));
  assert.equal(e.stop, 'done');
  assert.equal(e.steps, 2);
  assert.equal(e.answer, 'That costs $3.20.');
  const p1 = e.messages.find(m => m.role === 'plan');
  assert.deepEqual(p1.rules, ['R3', 'R4']);
  assert.deepEqual(e.messages.find(m => m.role === 'call').rules, ['R3', 'R4']);
  for (const id of Object.keys(TASKS)) {
    const r = runAgent(id, cfg(), {approve: true});
    for (const m of r.messages) if (['plan', 'call', 'answer'].includes(m.role) && m.rule) {
      assert.ok(Array.isArray(m.rules) && m.rules[m.rules.length - 1] === m.rule, `${id} ${m.role}`);
    }
  }
  assert.equal(oneAnswer('pair-shop').answer, 'That costs $4.80.');
});

test('approve answers the first ask only', () => {
  const y = runAgent('tell-zoe', cfg({budget: 5}), {approve: true});
  assert.equal(y.stop, 'waiting for approval');
  assert.equal(y.state.sent.length, 1);
  assert.equal(y.messages.filter(m => m.role === 'approval' && m.text === 'Yes.').length, 1);
  const n = runAgent('tell-zoe', cfg({budget: 5}), {approve: false});
  assert.equal(n.stop, 'declined');
  assert.deepEqual(n.state.sent, []);
});

test('seen and unseen match the context window', () => {
  const runs = [['pair-shop', cfg({budget: 12, stepLimit: 10})],
    ['pair-shop', cfg({budget: 12, stepLimit: 10, tools: {notepad: true}})],
    ['tell-zoe', cfg({budget: 5})], ['eli-shop', cfg()], ['kai-food', cfg({budget: 3, onNotFound: 'try again'})]];
  for (const [id, c] of runs) {
    const r = runAgent(id, c, {approve: true});
    const pre = r.messages.slice(0, -1);
    const win = contextWindow(pre, c.budget);
    pre.forEach((m, i) => assert.equal(Boolean(m.unseen), !win[i], `${id} unseen ${i}`));
    assert.ok(!('unseen' in r.messages[r.messages.length - 1]));
    r.messages.forEach((m, i) => {
      if (m.role !== 'plan') return;
      const before = r.messages.slice(0, i);
      const w = contextWindow(before, c.budget);
      const idx = [];
      w.forEach((v, j) => { if (v) idx.push(j); });
      assert.deepEqual(m.seen, idx, `${id} seen at ${i}`);
    });
  }
});

test('plan lines start with a capital', () => {
  for (const id of Object.keys(TASKS)) for (const nf of ['say so', 'try again', 'guess']) {
    for (let mask = 0; mask < 16; mask++) {
      const tools = {calculator: !!(mask & 1), lookup: !!(mask & 2), notepad: !!(mask & 4), send: !!(mask & 8)};
      for (const m of runAgent(id, cfg({tools, onNotFound: nf}), {approve: true}).messages) {
        if (m.role === 'plan') assert.ok(/^[A-Z]/.test(m.text), m.text);
      }
    }
  }
});

import {scheduleRows, _makeScheduler, statusText, rulesFor, ROLE_LABELS} from '../assets/trace-view.js';

test('scheduleRows reveals one row per tick', () => {
  assert.deepEqual(scheduleRows(5, {reduced: false}), [0, 1, 2, 3, 4]);
  assert.deepEqual(scheduleRows(0, {}), []);
});

test('reduced motion reveals all at once', () => {
  assert.deepEqual(scheduleRows(5, {reduced: true}), [4]);
});

function fakeTimers() {
  let id = 0;
  const q = new Map();
  return {
    setTimeout: (f) => { q.set(++id, f); return id; },
    clearTimeout: (i) => { q.delete(i); },
    // Fire the oldest pending timer; also lets a test fire a cleared one late.
    fireAll: () => { for (const [k, f] of [...q]) { q.delete(k); f(); } },
    pending: () => q.size,
  };
}

test('rerender cancels a running animation', () => {
  const t = fakeTimers();
  const s = _makeScheduler(t);
  const seen = [];
  s.start(6, {}, (i) => seen.push(['a', i]));
  t.fireAll();                       // row 1 of run a
  s.start(3, {}, (i) => seen.push(['b', i]));
  for (let k = 0; k < 10; k++) t.fireAll();
  assert.deepEqual(seen.filter((x) => x[0] === 'a'), [['a', 0], ['a', 1]]);
  assert.deepEqual(seen.filter((x) => x[0] === 'b'), [['b', 0], ['b', 1], ['b', 2]]);
  assert.equal(t.pending(), 0);
});

test('a late timer from a cancelled run does nothing', () => {
  let stored;
  const s = _makeScheduler({setTimeout: (f) => { stored = f; return 1; }, clearTimeout: () => {}});
  const seen = [];
  s.start(4, {}, (i) => seen.push(i));
  const late = stored;
  s.cancel();
  late();
  assert.deepEqual(seen, [0]);
});

test('trace view text helpers', () => {
  assert.equal(statusText({stop: 'done', steps: 4}), 'Stopped: done · 4 steps');
  assert.equal(statusText({stop: 'done', steps: 1}), 'Stopped: done · 1 step');
  assert.equal(ROLE_LABELS.approval, 'Waiting for your yes');
  assert.deepEqual(rulesFor({rules: ['R1', 'R6'], rule: 'R6'}).map((r) => r.id), ['R1', 'R6']);
  assert.equal(rulesFor({rule: 'R3'})[0].id, 'R3');
  assert.ok(rulesFor({rule: 'R3'})[0].text.length > 0);
  assert.deepEqual(rulesFor({role: 'result'}), []);
});

import {whyParagraphs} from '../assets/trace-view.js';

test('why box: unseen note shows with and without rules', () => {
  const note = /no longer sees it/;
  const noRule = whyParagraphs({role: 'result', text: 'x'}, true);
  assert.equal(noRule.length, 2);
  assert.match(noRule[1], note);
  const withRules = whyParagraphs({role: 'call', rules: ['R1', 'R6']}, true);
  assert.equal(withRules.length, 3);
  assert.match(withRules[2], note);
  assert.equal(whyParagraphs({role: 'result'}, false).length, 1);
});

// A minimal stand-in for the few DOM calls createTraceView makes, so selection can be tested in Node.
function fakeDocument() {
  const doc = {};
  const make = (tag) => {
    const n = {
      tagName: tag, ownerDocument: doc, children: [], attrs: {}, listeners: {}, className: '', textContent: '', disabled: false,
      classList: {set: new Set(), add(c) { this.set.add(c); }, toggle(c, on) { if (on) this.set.add(c); else this.set.delete(c); }},
      get firstChild() { return this.children[0]; },
      append(...xs) { for (const x of xs) this.children.push(x); },
      replaceChildren() { this.children = []; },
      setAttribute(k, v) { this.attrs[k] = String(v); },
      removeAttribute(k) { delete this.attrs[k]; },
      addEventListener(type, f) { (this.listeners[type] ||= []).push(f); },
      click() { for (const f of this.listeners.click || []) f(); },
      focus() {},
    };
    return n;
  };
  doc.createElement = make;
  doc.createTextNode = (t) => ({textContent: t});
  return doc;
}

import {createTraceView} from '../assets/trace-view.js';

test('onSelect hears every selection, including the view\'s own', () => {
  const doc = fakeDocument();
  const host = doc.createElement('div');
  const tools = doc.createElement('div');
  const heard = [];
  const tv = createTraceView(host, {tools, onSelect: (i) => heard.push(i)});
  const r = runAgent('mia-food');
  tv.show(r, {animate: false});
  assert.deepEqual(heard, [r.messages.length - 1]);   // show selects the newest row
  tv.select(1);
  assert.equal(heard.at(-1), 1);
  // Next message, after a fresh show: re-render, then step on one row at a time
  const [run, next] = tools.children[0].children;
  heard.length = 0;
  tv.show(r, {animate: false});
  run.click();                                          // animated Run: first row now, rest by timer
  tv.stop();
  const before = heard.length;
  next.click();
  assert.equal(heard.length, before + 1);
  assert.equal(heard.at(-1), tv.selected);
  assert.equal(tv.selected, 1);
});
