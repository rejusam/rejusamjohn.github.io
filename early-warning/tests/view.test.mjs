import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {compareRisk, curlSnippet, findStudent, formatRisk, pythonSnippet} from '../view.js';
import {rowToStudent, requestBody} from '../whatif.js';

const fixture = JSON.parse(readFileSync(new URL('./fixture-w8.json', import.meta.url)));
const URL_ = 'https://ssew-proxy.rejusamjohn.workers.dev/v1/score';

test('findStudent finds a row by id, from a number or typed text', () => {
  const id = fixture.rows[3][0];
  assert.deepEqual(findStudent(fixture, id), {row: fixture.rows[3], position: 3});
  assert.deepEqual(findStudent(fixture, ` ${id} `), {row: fixture.rows[3], position: 3});
});

test('findStudent gives null for an id not in the cohort, or not a number', () => {
  assert.equal(findStudent(fixture, 1), null);
  assert.equal(findStudent(fixture, 'abc'), null);
  assert.equal(findStudent(fixture, ''), null);
});

test('compareRisk', () => {
  assert.equal(compareRisk(0.2383, 0.2383), 'same');
  assert.equal(compareRisk(0.2383, 0.31), 'different');
  assert.equal(compareRisk(0.2383, 0.2384), 'different');
});

test('formatRisk shows two decimals', () => {
  assert.equal(formatRisk(0.2383), '0.24');
  assert.equal(formatRisk(0.0972), '0.10');
  assert.equal(formatRisk(1), '1.00');
});

test('curlSnippet posts the exact JSON body', () => {
  const body = requestBody(8, rowToStudent(fixture.columns, fixture.rows[0]));
  const text = curlSnippet(URL_, body);
  assert.match(text, /^curl -s -X POST https:\/\/ssew-proxy\.rejusamjohn\.workers\.dev\/v1\/score/);
  assert.ok(text.includes("-H 'Content-Type: application/json'"));
  assert.ok(text.includes(`-d '${JSON.stringify(body)}'`));
});

test('pythonSnippet uses requests and writes null as None', () => {
  const student = rowToStudent(fixture.columns, fixture.rows.find(r => r[11] === null));
  const text = pythonSnippet(URL_, requestBody(8, student));
  assert.ok(text.includes('import requests'));
  assert.ok(text.includes('requests.post(url, json=body, timeout=30)'));
  assert.ok(text.includes('"mean_score": None'));
  assert.ok(!text.includes('null'));
  assert.ok(text.includes(`url = "${URL_}"`));
});
