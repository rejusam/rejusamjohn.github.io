import test from 'node:test';
import assert from 'node:assert/strict';
import { fold, lessonIndex, searchLessons, opensSearch } from '../assets/search.js';
import { PATH, TOTAL_LIVE } from '../assets/path.js';

test('fold lowers case, drops accents and punctuation', () => {
  assert.equal(fold('  Naïve   Bayes! '), 'naive bayes');
  assert.equal(fold('k-nearest neighbours'), 'k nearest neighbours');
  assert.equal(fold('What is a model?'), 'what is a model');
});

test('the index holds every live lesson once, in path order', () => {
  const idx = lessonIndex();
  assert.equal(idx.length, TOTAL_LIVE);
  assert.equal(new Set(idx.map((i) => i.slug)).size, idx.length);
  const order = PATH.filter((l) => !l.soon).flatMap((l) => l.lessons.map((x) => x.slug));
  assert.deepEqual(idx.map((i) => i.slug), order);
});

test('an empty query lists every lesson', () => {
  assert.equal(searchLessons('').length, TOTAL_LIVE);
  assert.equal(searchLessons('   ').length, TOTAL_LIVE);
});

test('a title word finds its lesson first', () => {
  assert.equal(searchLessons('attention')[0].slug, 'attention');
  assert.equal(searchLessons('naive')[0].slug, 'naive-bayes');
  assert.equal(searchLessons('Naïve')[0].slug, 'naive-bayes');
  assert.equal(searchLessons('grad')[0].slug, 'gradient-descent');
});

test('every query word must match', () => {
  const r = searchLessons('random forests');
  assert.equal(r[0].slug, 'random-forests');
  assert.ok(r.every((x) => /random/i.test(x.title)));
  assert.deepEqual(searchLessons('zzzz'), []);
});

test('a level code or level name finds that level', () => {
  const l3 = PATH.find((l) => l.level === 'L3').lessons.map((l) => l.slug);
  assert.deepEqual(searchLessons('l3').map((r) => r.slug), l3);
});

test('title matches rank above level-name matches', () => {
  const index = [
    { slug: 'a', title: 'Something else', level: 'L1', levelName: 'Trees', order: 0 },
    { slug: 'b', title: 'Decision trees', level: 'L2', levelName: 'Other', order: 1 },
  ];
  assert.deepEqual(searchLessons('trees', index).map((r) => r.slug), ['b', 'a']);
});

test('opensSearch: slash outside fields, Ctrl/Cmd+K anywhere', () => {
  const body = { tagName: 'BODY' };
  const input = { tagName: 'INPUT' };
  assert.equal(opensSearch({ key: '/', target: body }), true);
  assert.equal(opensSearch({ key: '/', target: input }), false);
  assert.equal(opensSearch({ key: '/', target: { tagName: 'DIV', isContentEditable: true } }), false);
  assert.equal(opensSearch({ key: 'k', metaKey: true, target: input }), true);
  assert.equal(opensSearch({ key: 'K', ctrlKey: true, target: body }), true);
  assert.equal(opensSearch({ key: 'k', target: body }), false);
  assert.equal(opensSearch({ key: '/', ctrlKey: true, target: body }), false);
});
