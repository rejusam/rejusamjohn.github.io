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
