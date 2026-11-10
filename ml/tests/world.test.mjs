import {test} from 'node:test';
import assert from 'node:assert/strict';
import {WORLD, worldSentences, worldText, ALPHABET, checkFact} from '../assets/world.js';

test('facts table is complete', () => {
  for (const p of WORLD.people) {
    assert.ok(WORLD.foods.includes(WORLD.likes[p]), p);
    assert.ok(WORLD.places.includes(WORLD.livesIn[p]), p);
  }
  assert.equal(Object.keys(WORLD.likes).length, WORLD.people.length);
  assert.equal(Object.keys(WORLD.livesIn).length, WORLD.people.length);
  for (const f of WORLD.foods) assert.ok(WORLD.people.some(p => WORLD.likes[p] === f), f);
  for (const pl of WORLD.places) assert.ok(WORLD.people.some(p => WORLD.livesIn[p] === pl), pl);
  assert.deepEqual(WORLD.prep, {sea: 'by', hills: 'in', lake: 'by', town: 'in'});
  assert.equal(WORLD.likes.mia, 'figs');
  assert.equal(WORLD.livesIn.mia, 'town');
});

test('alphabet is space, full stop, then a to z', () => {
  assert.equal(ALPHABET.length, 28);
  assert.equal(ALPHABET.join(''), ' .abcdefghijklmnopqrstuvwxyz');
});

test('world text is reproducible and grammatical', () => {
  const a = worldSentences({n: 300, seed: 7});
  assert.deepEqual(a, worldSentences({n: 300, seed: 7}));
  assert.notDeepEqual(a, worldSentences({n: 300, seed: 8}));
  assert.equal(a.length, 300);
  const P = WORLD.people.join('|'), F = WORLD.foods.join('|'), L = WORLD.places.join('|');
  const re = new RegExp(`^(?:(?:${P}) likes (?:${F})\\.|(?:${P}) lives (?:by|in) the (?:${L})\\.|(?:${P}) and (?:${P}) are friends\\.)$`);
  let likes = 0, lives = 0, friends = 0;
  for (const s of a) {
    assert.match(s, re);
    if (s.includes(' likes ')) { likes++; assert.equal(checkFact(s), 'true'); }
    else if (s.includes(' lives ')) { lives++; assert.equal(checkFact(s), 'true'); }
    else {
      friends++;
      const [p, , q] = s.split(' ');
      assert.notEqual(p, q);
    }
  }
  assert.ok(likes > 0 && lives > 0 && friends > 0);
  assert.equal(worldText({n: 300, seed: 7}), a.join(' '));
});

test('world sentence mix is about 40/40/20', () => {
  const a = worldSentences({n: 20000, seed: 3});
  const share = f => a.filter(f).length / a.length;
  assert.ok(Math.abs(share(s => s.includes(' likes ')) - 0.4) < 0.015);
  assert.ok(Math.abs(share(s => s.includes(' lives ')) - 0.4) < 0.015);
  assert.ok(Math.abs(share(s => s.includes(' friends')) - 0.2) < 0.015);
});

test('checkFact', () => {
  assert.equal(checkFact('mia likes figs.'), 'true');
  assert.equal(checkFact('mia likes pears.'), 'false');
  assert.equal(checkFact('mia lives in the town.'), 'true');
  assert.equal(checkFact('mia lives by the sea.'), 'false');
  assert.equal(checkFact('mia lives by the town.'), 'other');
  assert.equal(checkFact('ana and ben are friends.'), 'other');
  assert.equal(checkFact('xyz likes pears.'), 'other');
  assert.equal(checkFact('mia likes figs'), 'other');
  assert.equal(checkFact(''), 'other');
});
