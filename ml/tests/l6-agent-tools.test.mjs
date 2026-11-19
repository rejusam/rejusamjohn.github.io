import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { WORLD } from '../assets/world.js';
import { PRICES, RULES, TASKS, TOOLS, runAgent, calc, fmtMoney, fmtNumber, roundOneFigure, callText, lookupText } from '../assets/agent.js';
import * as S from '../agent-tools/stage.js';

const html = readFileSync(new URL('../agent-tools/index.html', import.meta.url), 'utf8');
const src = readFileSync(new URL('../agent-tools/stage.js', import.meta.url), 'utf8');

const sumGoal = TASKS.sum.goal;
const sumExpr = sumGoal.terms.map(([a, b]) => `${a}*${b}`).join('+');
const sumTrue = calc(sumExpr).value;
const eliGoal = TASKS['eli-shop'].goal;
const eliFood = WORLD.likes.eli;
const eliTrue = calc(`${eliGoal.qty}*${PRICES[eliFood]}`).value;
const ALL = (on) => ({ calculator: on, lookup: on, notepad: on });

test('the lesson runs match runAgent, and the sum task gives 5.35', () => {
  const r = runAgent('sum');
  assert.equal(r.steps, 2);
  assert.equal(r.stop, 'done');
  assert.equal(r.answer, `The sum is ${fmtNumber(sumTrue)}.`);
  assert.equal(r.answer, 'The sum is 5.35.'); // the one literal
  const call = r.messages.find((m) => m.role === 'call');
  assert.equal(call.text, callText('calculator', { expr: sumExpr }));
  assert.equal(r.messages.find((m) => m.role === 'result').text, fmtNumber(sumTrue));
  // The stage's traces are runAgent's, step by step.
  const runs = S.stepRuns();
  assert.deepEqual(runs[0], runAgent('sum'));
  assert.deepEqual(runs[1], runAgent('eli-shop'));
  assert.deepEqual(runs[2], runAgent('sum', { tools: { calculator: false } }));
  assert.deepEqual(runs[3], runAgent('eli-shop', { tools: { lookup: false } }));
  assert.deepEqual(runs[4], runAgent('eli-shop', { tools: { calculator: true, lookup: true, notepad: false } }));
  // Calculator off and sums set to guess give the same trace.
  assert.deepEqual(runAgent('sum', { tools: { calculator: false } }), runAgent('sum', { sums: 'guess' }));
});

test('guess error is exact', () => {
  // The rounding rule: each number to one significant figure, then the sum worked out exactly.
  const g = S.guessSum(sumGoal.terms);
  assert.deepEqual(g.rounded, sumGoal.terms.map(([a, b]) => [roundOneFigure(a), roundOneFigure(b)]));
  assert.deepEqual(g.rounded, [[3, 1], [2, 0.8]]);
  assert.equal(g.expr, '3*1+2*0.8');
  assert.equal(g.value, calc(g.expr).value);
  assert.equal(fmtNumber(g.value), '4.6');
  assert.equal(fmtNumber(sumTrue), '5.35');
  assert.equal(fmtNumber(sumTrue - g.value), '0.75');
  // The run with the calculator off writes exactly that guess, in one step.
  const r = runAgent('sum', { tools: { calculator: false } });
  assert.equal(r.answer, `The sum is ${fmtNumber(g.value)}.`);
  assert.equal(r.steps, 1);
  assert.equal(r.messages.filter((m) => m.role === 'call').length, 0);
  assert.match(r.messages[1].text, new RegExp(`${g.expr.replace(/[*+.]/g, '\\$&')} = ${fmtNumber(g.value)}`));
  assert.deepEqual(r.messages[1].rules, ['R4', 'R6']);
  assert.equal(S.errorOf('sum', r.answer), sumTrue - g.value);
  // Lookup off on eli-shop: apples at the first price; calculator still exact.
  const e = runAgent('eli-shop', { tools: { lookup: false } });
  const first = WORLD.foods[0];
  assert.equal(first, 'apples');
  const firstPrice = PRICES[Object.keys(PRICES)[0]];
  assert.equal(e.answer, `That costs ${fmtMoney(calc(`${eliGoal.qty}*${firstPrice}`).value)}.`);
  assert.equal(e.steps, 2);
  assert.deepEqual(e.messages.filter((m) => m.role === 'call').map((m) => m.text), [callText('calculator', { expr: `${eliGoal.qty}*${firstPrice}` })]);
  assert.match(e.messages[1].text, new RegExp(`eli likes ${first}\\.`));
  assert.deepEqual(e.messages[1].rules, ['R3', 'R4']);
  assert.equal(fmtMoney(S.errorOf('eli-shop', e.answer)), fmtMoney(eliTrue - eliGoal.qty * firstPrice));
  assert.equal(fmtMoney(S.errorOf('eli-shop', e.answer)), '$1.80');
});

test('every checkbox combination gives the trace runAgent gives', () => {
  assert.equal(S.COMBOS.length, 8);
  const seen = new Set();
  for (const t of S.COMBOS) {
    seen.add(JSON.stringify(t));
    const cfg = { tools: { calculator: t.calculator, lookup: t.lookup, notepad: t.notepad } };
    const want = runAgent('eli-shop', cfg);
    assert.deepEqual(S.runFor(t), want);
    // The result line, recomputed here from the run and the true cost.
    const got = Number(want.answer.match(/\$(\d+\.\d\d)/)[1]);
    const verdict = got === Number(eliTrue.toFixed(2)) ? 'right' : `wrong by ${fmtMoney(Math.abs(eliTrue - got))}`;
    assert.equal(S.verdict('eli-shop', want.answer), verdict);
    assert.equal(S.resultLine(want), `${want.answer} ${verdict === 'right' ? '✓' : '✗'} ${verdict} · ${want.steps} ${want.steps === 1 ? 'step' : 'steps'}`);
  }
  assert.equal(seen.size, 8);
  // The values per combination (calculator, lookup, notepad).
  const table = S.COMBOS.map((t) => [+t.calculator, +t.lookup, +t.notepad, S.runFor(t).steps, S.verdict('eli-shop', S.runFor(t).answer)].join(' '));
  assert.deepEqual(table.sort(), [
    '0 0 0 1 wrong by $1.80', '0 0 1 1 wrong by $1.80',
    '0 1 0 3 wrong by $1.00', '0 1 1 5 wrong by $1.00',
    '1 0 0 2 wrong by $1.80', '1 0 1 2 wrong by $1.80',
    '1 1 0 4 right', '1 1 1 6 right',
  ]);
  // Only lookup and calculator both on are right.
  for (const t of S.COMBOS) assert.equal(S.verdict('eli-shop', S.runFor(t).answer) === 'right', t.calculator && t.lookup);
  // Notepad: with lookup on it adds one note per looked-up fact and keeps the answer; with lookup off it changes nothing.
  for (const calcOn of [true, false]) {
    const off = S.runFor({ calculator: calcOn, lookup: true, notepad: false });
    const on = S.runFor({ calculator: calcOn, lookup: true, notepad: true });
    assert.equal(on.answer, off.answer);
    const lookups = off.messages.filter((m) => m.role === 'call' && m.tool === 'lookup').length;
    assert.equal(on.steps - off.steps, lookups);
    assert.equal(on.state.notes.length, lookups);
    const noL = (n) => S.runFor({ calculator: calcOn, lookup: false, notepad: n });
    assert.deepEqual(noL(true), noL(false));
  }
});

test('with all tools on the answer is right', () => {
  const r = S.runFor(ALL(true));
  assert.equal(r.answer, `That costs ${fmtMoney(eliTrue)}.`);
  assert.equal(S.verdict('eli-shop', r.answer), 'right');
  assert.equal(r.stop, 'done');
  assert.equal(S.verdict('sum', runAgent('sum').answer), 'right');
  assert.equal(S.verdict('sum', runAgent('sum', { tools: { calculator: false } }).answer), 'wrong by 0.75');
  // and with all off it is wrong
  assert.notEqual(S.verdict('eli-shop', S.runFor(ALL(false)).answer), 'right');
});

test('who made each row, and the call up close', () => {
  const r = runAgent('eli-shop');
  assert.deepEqual(r.messages.map(S.whoMade),
    [null, 'model', 'model', 'tool', 'model', 'model', 'tool', 'model', 'model', 'tool', 'model', 'model', 'loop']);
  assert.equal(S.whoText(r.messages[0]), 'the task: given to the model');
  assert.equal(S.whoText(r.messages[2]), 'a call: text the model wrote');
  assert.equal(S.whoText(r.messages[3]), 'a result: what lookup returned');
  assert.equal(S.whoText(r.messages[12]), 'the loop stopped the run');
  // callOfStep: the call and result of the selected row's step.
  const c = S.callOfStep(r, 5);
  assert.equal(c.call.text, `lookup("${eliFood}")`);
  assert.equal(c.result.text, lookupText(eliFood));
  assert.equal(S.inputText(c.call.input), `key = "${eliFood}"`);
  assert.equal(S.callOfStep(r, 11), null); // the answer step calls no tool
  assert.equal(S.callOfStep(r, 0), null);
});

test('page values are computed and match the page', () => {
  const v = S.pageValues();
  const sum = runAgent('sum');
  const g = S.guessSum(sumGoal.terms);
  assert.equal(v['sum-task'], TASKS.sum.text);
  assert.equal(v['sum-call'], callText('calculator', { expr: sumExpr }));
  assert.equal(v['sum-result'], fmtNumber(sumTrue));
  assert.equal(v['sum-agent'], sum.answer);
  assert.equal(v['sum-steps'], String(sum.steps));
  assert.equal(v['guess-expr'], g.expr);
  assert.equal(v['guess-value'], fmtNumber(g.value));
  assert.equal(v['guess-answer'], `The sum is ${fmtNumber(g.value)}.`);
  assert.equal(v['sum-error'], fmtNumber(sumTrue - g.value));
  assert.equal(v['rounded-list'], '3 stays 3, 1.25 becomes 1, 2 stays 2 and 0.8 stays 0.8');
  assert.equal(v['guess-rules'], 'R4 and R6');
  assert.equal(v['guess-steps'], String(runAgent('sum', { tools: { calculator: false } }).steps));
  assert.equal(v['about-calculator'], TOOLS.calculator.about);
  assert.equal(v['about-lookup'], TOOLS.lookup.about);
  assert.equal(v['about-notepad'], TOOLS.notepad.about);
  const eli = runAgent('eli-shop');
  assert.equal(v['eli-call-1'], 'lookup("eli")');
  assert.equal(v['eli-input-1'], 'key = "eli"');
  assert.equal(v['eli-result-1'], lookupText('eli'));
  assert.equal(v['eli-calls'], String(eli.messages.filter((m) => m.role === 'call').length));
  assert.equal(v['eli-steps'], String(eli.steps));
  assert.equal(v['eli-qty'], String(eliGoal.qty));
  assert.equal(v['eli-food'], eliFood);
  assert.equal(v['eli-price'], fmtMoney(PRICES[eliFood]));
  assert.equal(v['true-eli'], fmtMoney(eliTrue));
  assert.equal(v['guess-food'], WORLD.foods[0]);
  assert.equal(v['guess-price'], fmtMoney(PRICES[Object.keys(PRICES)[0]]));
  const nl = runAgent('eli-shop', { tools: { lookup: false } });
  assert.equal(v['nolookup-call'], nl.messages.find((m) => m.role === 'call').text);
  assert.equal(v['nolookup-answer'], nl.answer.match(/\$\d+\.\d\d/)[0]);
  assert.equal(v['nolookup-error'], '$1.80');
  assert.equal(v['nolookup-steps'], String(nl.steps));
  assert.equal(v['nolookup-rules'], 'R3 and R4');
  assert.equal(v['combos'], '2 × 2 × 2 = 8');
  const np = S.runFor(ALL(true));
  assert.equal(v['notepad-steps'], String(np.steps));
  assert.equal(v['notepad-extra'], String(np.steps - eli.steps));
  assert.equal(v['notepad-rule'], 'R2');
  assert.equal(v['nocalc-answer'], fmtMoney(calc(`${roundOneFigure(eliGoal.qty)}*${roundOneFigure(PRICES[eliFood])}`).value));
  assert.equal(v['nocalc-expr'], `${roundOneFigure(eliGoal.qty)}*${roundOneFigure(PRICES[eliFood])}`);
  assert.equal(v['price-list'], Object.keys(PRICES).map((f) => `${f} ${fmtMoney(PRICES[f])}`).join(', '));
  assert.equal(v['likes-list'], WORLD.people.map((p) => `${p} ${WORLD.likes[p]}`).join(', '));
  // Every span in the page holds exactly the computed value, and every value is used.
  const spans = [...html.matchAll(/<span data-value="([^"]+)">([^<]*)<\/span>/g)];
  assert.ok(spans.length >= 30);
  const unesc = (s) => s.replace(/&quot;/g, '"').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>');
  for (const [, key, text] of spans) {
    assert.ok(key in v, `unknown value ${key}`);
    assert.equal(unesc(text), v[key], key);
  }
  for (const key of Object.keys(v)) assert.ok(html.includes(`data-value="${key}"`), `unused value ${key}`);
});

test('lesson page follows the contract', () => {
  assert.equal((html.match(/<li class="step"/g) || []).length, 5);
  assert.match(html, /<title>Tools · ML Explorer<\/title>/);
  assert.match(html, /<h1>Tools<\/h1>/);
  assert.match(html, /Level 6 · Agents/);
  assert.match(html, /data-lesson="agent-tools"/);
  assert.match(html, /<link rel="canonical" href="https:\/\/rejusamjohn\.pages\.dev\/ml\/agent-tools\/">/);
  assert.match(html, /<meta property="og:image" content="https:\/\/rejusamjohn\.pages\.dev\/ml\/og\.png">/);
  assert.match(html, /<a class="next-lesson" href="\/ml\/agent-loop\/">Next lesson: The loop and when to stop →<\/a>/);
  assert.match(html, /href="\/ml\/agents\/"/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.match(html, /<script type="module" src="https:\/\/static\.cloudflareinsights\.com\/beacon\.min\.js" data-cf-beacon='\{"token": "e46c62f910fe4def882feaed064efced"\}'><\/script>/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)(?![^>]*application\/ld\+json)[^>]*>/);
  assert.doesNotMatch(html, /\son[a-z]+=/i);
  assert.equal((html.match(/<form class="q"/g) || []).length, 2);
  assert.match(html, /Why give a model a calculator\?/);
  assert.match(html, /Who runs the tool: the model or the loop\?/);
  const step1 = html.slice(html.indexOf('id="step-1"'), html.indexOf('id="step-2"'));
  assert.match(step1, /fixed rules/);
  assert.match(step1, /A real model writes its own calls and can get them wrong/);
  // the rounding rule is stated in step 3, and R4's text matches RULES
  const step3 = html.slice(html.indexOf('id="step-3"'), html.indexOf('id="step-4"'));
  assert.match(step3, /round each number to one significant figure/);
  assert.ok(RULES.find((r) => r.id === 'R4').text.includes('round each number to one significant figure, work out the sum exactly'));
  // send is not one of this page's tools
  assert.match(html, /send, comes later/);
  // further reading, checked
  assert.match(html, /Schick et al\., "Toolformer: Language Models Can Teach Themselves to Use Tools", arXiv 2302\.04761 \(2023\)/);
  assert.match(html, /class="katex"/);
});

test('wording rules', () => {
  const body = html.slice(html.indexOf('<main'), html.indexOf('<ol class="reading">'));
  assert.doesNotMatch(body, /\b(thinks?|knows?|understands?|wants?|decides|autonomous|reasons|creative|conscious)\b|intelligen/i);
  assert.doesNotMatch(src, /\b(thinks?|knows?|understands?|wants?|decides|autonomous|intelligent)\b/i);
  assert.doesNotMatch(html + src, /GPT|Gemini|Llama|LangChain|Mistral|Copilot|Hugging/i);
  assert.doesNotMatch(html, /delve|in today's world|unlock|harness the power|it's important to note|dive in/i);
  assert.doesNotMatch(src, /innerHTML|\beval\(|new Function/);
});
