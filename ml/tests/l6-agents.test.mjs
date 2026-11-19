import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { WORLD } from '../assets/world.js';
import { PRICES, RULES, TASKS, runAgent, oneAnswer, calc, fmtMoney, callText } from '../assets/agent.js';
import * as S from '../agents/stage.js';

const html = readFileSync(new URL('../agents/index.html', import.meta.url), 'utf8');
const src = readFileSync(new URL('../agents/stage.js', import.meta.url), 'utf8');

// The true cost of eli-shop, worked out from the world and the price list with the calculator.
const eliGoal = TASKS['eli-shop'].goal;
const eliTrue = calc(`${eliGoal.qty}*${PRICES[WORLD.likes.eli]}`).value;

test('one answer is wrong on both tasks and the agent is right', () => {
  // mia-food: the one answer names a food mia does not like; the agent names hers.
  const oneMia = oneAnswer('mia-food').answer;
  assert.notEqual(oneMia, `mia likes ${WORLD.likes.mia}.`);
  assert.equal(runAgent('mia-food').answer, `mia likes ${WORLD.likes.mia}.`);
  assert.equal(S.trueAnswer('mia-food'), `mia likes ${WORLD.likes.mia}.`);
  // The one answer comes from the guess rule: the first food in the table.
  assert.equal(oneMia, `mia likes ${WORLD.foods[0]}.`);
  // eli-shop: one answer guesses; the agent's cost matches the calculator on the real price.
  const oneEli = oneAnswer('eli-shop').answer;
  const agentEli = runAgent('eli-shop').answer;
  assert.notEqual(oneEli, `That costs ${fmtMoney(eliTrue)}.`);
  assert.equal(agentEli, `That costs ${fmtMoney(eliTrue)}.`);
  assert.equal(agentEli, 'That costs $5.00.'); // the one literal
  assert.equal(S.trueAnswer('eli-shop'), agentEli);
  // The one answer's guess: first food, first price, rounded to one figure.
  const firstPrice = PRICES[Object.keys(PRICES)[0]];
  assert.equal(oneEli, `That costs ${fmtMoney(eliGoal.qty * firstPrice)}.`);
  for (const id of ['mia-food', 'eli-shop']) {
    assert.equal(S.isRight(id, oneAnswer(id).answer), false);
    assert.equal(S.isRight(id, runAgent(id).answer), true);
  }
});

test('eli-shop takes 4 steps', () => {
  const r = runAgent('eli-shop');
  assert.equal(r.steps, 4);
  assert.equal(r.stop, 'done');
  const calls = r.messages.filter((m) => m.role === 'call');
  assert.equal(calls.length, 3);
  // every step but the last calls one tool
  assert.deepEqual(calls.map((m) => m.step), [1, 2, 3]);
  assert.deepEqual(calls.map((m) => m.text),
    ['lookup("eli")', `lookup("${WORLD.likes.eli}")`, `calculator("${eliGoal.qty}*${PRICES[WORLD.likes.eli]}")`]);
  // every model call is one step: one plan row per step
  assert.equal(r.messages.filter((m) => m.role === 'plan').length, r.steps);
  assert.equal(r.messages.length, 13);
  // the calculator's result row is the number the answer uses
  const res = r.messages.filter((m) => m.role === 'result');
  assert.equal(Number(res[2].text), eliTrue);
  // mia-food: 2 steps, 1 tool call
  const m = runAgent('mia-food');
  assert.equal(m.steps, 2);
  assert.equal(m.messages.filter((x) => x.role === 'call').length, 1);
  assert.equal(m.messages.length, 7);
  // the same task and rules give the same trace every time
  assert.deepEqual(runAgent('eli-shop'), runAgent('eli-shop'));
  assert.deepEqual(oneAnswer('mia-food'), oneAnswer('mia-food'));
});

test('rules list matches RULES', () => {
  const block = html.match(/<ul class="rules-list">([^]*?)<\/ul>/);
  assert.ok(block, 'rules list present');
  const items = [...block[1].matchAll(/<li><strong>(R\d)<\/strong> ([^<]*)<\/li>/g)];
  const unescape = (s) => s.replace(/&quot;/g, '"').replace(/&amp;/g, '&');
  assert.deepEqual(items.map((m) => [m[1], unescape(m[2])]), RULES.map((r) => [r.id, r.text]));
  assert.deepEqual(S.RULE_NAMES.map((r) => r.id), RULES.map((r) => r.id));
});

test('one-answer trace is two rows: the task and the answer', () => {
  const t = S.oneAnswerTrace('mia-food');
  assert.deepEqual(t.messages.map((m) => m.role), ['task', 'answer']);
  assert.equal(t.messages[1].text, oneAnswer('mia-food').answer);
  assert.deepEqual(t.messages[1].rules, ['R3', 'R6']);
  assert.equal(t.steps, 1);
  assert.equal(t.stop, 'done');
  assert.deepEqual(oneAnswer('eli-shop').messages[2].rules, ['R3', 'R4', 'R6']);
});

test('loop boxes and step counter follow the rows', () => {
  const r = runAgent('eli-shop');
  assert.deepEqual(r.messages.map(S.loopBox),
    ['read', 'choose', 'act', 'see', 'choose', 'act', 'see', 'choose', 'act', 'see', 'choose', 'choose', 'stop']);
  assert.equal(S.counterText(r.messages[0], r), 'the task: before step 1');
  assert.equal(S.counterText(r.messages[4], r), 'step 2 of 4 (model call 2)');
  assert.equal(S.counterText(r.messages[12], r), 'stopped after 4 steps');
  const one = runAgent('mia-food');
  assert.equal(S.counterText(one.messages[1], one), 'step 1 of 2 (model call 1)');
  assert.equal(S.rulesUsed(r).join(' '), 'R3 R4 R6');
  assert.deepEqual(S.ruleHighlight(r.messages[1]), ['R3']);
  assert.deepEqual(S.ruleHighlight(r.messages[3]), []);
});

test('page values are computed and match the page', () => {
  const v = S.pageValues();
  const oneMia = oneAnswer('mia-food');
  assert.equal(v['one-mia'], oneMia.answer);
  assert.equal(v['true-mia'], WORLD.likes.mia);
  assert.equal(v['guess-food'], WORLD.foods[0]);
  assert.equal(v['one-mia-rules'], 'R3 and R6');
  assert.equal(v['mia-steps'], String(runAgent('mia-food').steps));
  assert.equal(v['mia-agent'], runAgent('mia-food').answer);
  assert.equal(v['mia-call'], 'lookup("mia")');
  const eli = runAgent('eli-shop');
  assert.equal(v['eli-steps'], String(eli.steps));
  assert.equal(v['eli-calls'], String(eli.messages.filter((m) => m.role === 'call').length));
  assert.equal(v['eli-steps-minus'], `${eli.steps} − 1 = ${eli.steps - 1}`);
  assert.equal(v['eli-calc'], callText('calculator', {expr: `${eliGoal.qty}*${PRICES[WORLD.likes.eli]}`}));
  assert.equal(v['eli-qty'], String(eliGoal.qty));
  assert.equal(v['eli-food'], WORLD.likes.eli);
  assert.equal(v['eli-price'], fmtMoney(PRICES[WORLD.likes.eli]));
  assert.equal(v['agent-eli'], fmtMoney(eliTrue));
  assert.equal(v['true-eli'], fmtMoney(eliTrue));
  const oneEli = Number(oneAnswer('eli-shop').answer.match(/\$(\d+\.\d\d)/)[1]);
  assert.equal(v['one-eli'], fmtMoney(oneEli));
  assert.equal(v['eli-error'], fmtMoney(eliTrue - oneEli));
  assert.equal(v['guess-price'], fmtMoney(PRICES[Object.keys(PRICES)[0]]));
  assert.equal(v['one-eli-rules'], 'R3, R4 and R6');
  assert.equal(v['rules-used'], 'R3, R4 and R6');
  assert.equal(v['rules-unused'], 'R1, R2 and R5');
  assert.equal(v['food-table'], 'apples, pears, plums, figs, limes and grapes');
  assert.equal(v['price-list'], Object.keys(PRICES).map((f) => `${f} ${fmtMoney(PRICES[f])}`).join(', '));
  assert.equal(v['likes-list'], WORLD.people.map((p) => `${p} ${WORLD.likes[p]}`).join(', '));
  // Every span in the page holds exactly the computed value.
  const spans = [...html.matchAll(/<span data-value="([^"]+)">([^<]*)<\/span>/g)];
  assert.ok(spans.length >= 20);
  const unesc = (s) => s.replace(/&quot;/g, '"').replace(/&amp;/g, '&');
  for (const [, key, text] of spans) {
    assert.ok(key in v, `unknown value ${key}`);
    assert.equal(unesc(text), v[key], key);
  }
  for (const key of Object.keys(v)) assert.ok(html.includes(`data-value="${key}"`), `unused value ${key}`);
});

test('lesson page follows the contract', () => {
  assert.equal((html.match(/<li class="step"/g) || []).length, 5);
  assert.match(html, /<title>What is an agent\? · ML Explorer<\/title>/);
  assert.match(html, /<h1>What is an agent\?<\/h1>/);
  assert.match(html, /Level 6 · Agents/);
  assert.match(html, /<link rel="canonical" href="https:\/\/rejusamjohn\.pages\.dev\/ml\/agents\/">/);
  assert.match(html, /<meta property="og:image" content="https:\/\/rejusamjohn\.pages\.dev\/ml\/og\.png">/);
  assert.match(html, /<a class="next-lesson" href="\/ml\/agent-tools\/">Next lesson: Tools →<\/a>/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.match(html, /<script type="module" src="https:\/\/static\.cloudflareinsights\.com\/beacon\.min\.js" data-cf-beacon='\{"token": "e46c62f910fe4def882feaed064efced"\}'><\/script>/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)(?![^>]*application\/ld\+json)[^>]*>/);
  assert.doesNotMatch(html, /\son[a-z]+=/i);
  assert.equal((html.match(/<form class="q"/g) || []).length, 2);
  assert.match(html, /What makes an agent different from a single model answer\?/);
  assert.match(html, /Where did the <span data-value="agent-eli">/);
  // honesty sentence in step 1
  const step1 = html.slice(html.indexOf('id="step-1"'), html.indexOf('id="step-2"'));
  assert.match(step1, /fixed rules/);
  assert.match(step1, /a real system uses a language model instead, and it can write wrong text/);
  assert.match(step1, /Run/);
  assert.match(step1, /Next message/);
  // earlier levels by name
  assert.match(html, /href="\/ml\/language-models\/"/);
  assert.match(html, /href="\/ml\/llms\/"/);
  // further reading, checked
  assert.match(html, /Russell &amp; Norvig, <cite>Artificial Intelligence: A Modern Approach<\/cite> \(4th ed\.\), Chapter 2, "Intelligent Agents"/);
  assert.match(html, /Yao et al\., "ReAct: Synergizing Reasoning and Acting in Language Models", arXiv 2210\.03629 \(2022\)/);
  // maths rendered
  assert.match(html, /data-tex="[^"]*\\text\{policy\}/);
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
