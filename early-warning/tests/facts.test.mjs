import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

// Every number in the page's text carries data-src="facts#<key>" and must equal the value in
// data/facts.json, which the export script reads from the repo file named in its "source".
const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const facts = JSON.parse(readFileSync(new URL('../data/facts.json', import.meta.url), 'utf8'));
const tagged = [...html.matchAll(/<(\w+)[^>]*\bdata-src="([^"]*)"[^>]*>([^<]*)<\/\1>/g)];

test('the page has numbers tagged with a facts key', () => {
  assert.ok(tagged.length >= 10);
});

test('every data-src names a facts key', () => {
  const all = [...html.matchAll(/data-src="([^"]*)"/g)].map(m => m[1]);
  assert.equal(all.length, tagged.length, 'every data-src element holds plain text');
  for (const src of all) {
    assert.match(src, /^facts#\w+$/);
    assert.ok(src.slice(6) in facts, `${src} is not in data/facts.json`);
  }
});

test('every tagged number equals its value in facts.json', () => {
  for (const [, , src, text] of tagged) {
    const value = facts[src.slice(6)].value;
    const allowed = [String(value), typeof value === 'number' ? value.toLocaleString('en-NZ') : null];
    assert.ok(allowed.includes(text), `${src}: page says ${text}, facts say ${value}`);
  }
});

test('every fact names its source', () => {
  for (const [key, fact] of Object.entries(facts)) assert.ok(fact.source, key);
});
