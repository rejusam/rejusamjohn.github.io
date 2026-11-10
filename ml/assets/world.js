// The invented world for the Level 5 lessons: eight made-up people, six foods, four places.
// The facts are rules of the page, not claims about anyone. Pure functions, no DOM.
import {mulberry32} from './maths.js';

export const WORLD = {
  people: ['ana', 'ben', 'eli', 'mia', 'noa', 'sam', 'tom', 'zoe'],
  foods: ['apples', 'pears', 'plums', 'figs', 'limes', 'grapes'],
  places: ['sea', 'hills', 'lake', 'town'],
  prep: {sea: 'by', hills: 'in', lake: 'by', town: 'in'},
  // Every person likes one food and lives in one place; every food and place is used.
  likes: {ana: 'apples', ben: 'pears', eli: 'plums', mia: 'figs',
          noa: 'limes', sam: 'grapes', tom: 'apples', zoe: 'pears'},
  livesIn: {ana: 'sea', ben: 'hills', eli: 'lake', mia: 'town',
            noa: 'sea', sam: 'hills', tom: 'lake', zoe: 'town'},
};

// Index = id. Space, full stop, then a to z.
export const ALPHABET = ' .abcdefghijklmnopqrstuvwxyz'.split('');

const pick = (xs, rng) => xs[Math.floor(rng() * xs.length)];

// Each sentence: 0.4 "<p> likes <food>.", 0.4 "<p> lives <prep> the <place>.",
// 0.2 "<p> and <q> are friends." (p and q different, both uniform).
export function worldSentences({n, seed}) {
  const rng = mulberry32(seed);
  const {people} = WORLD;
  const out = [];
  for (let i = 0; i < n; i++) {
    const r = rng();
    const p = pick(people, rng);
    if (r < 0.4) out.push(`${p} likes ${WORLD.likes[p]}.`);
    else if (r < 0.8) {
      const pl = WORLD.livesIn[p];
      out.push(`${p} lives ${WORLD.prep[pl]} the ${pl}.`);
    } else {
      let q = p;
      while (q === p) q = pick(people, rng);
      out.push(`${p} and ${q} are friends.`);
    }
  }
  return out;
}

export const worldText = ({n, seed}) => worldSentences({n, seed}).join(' ');

// 'true' or 'false' for the two fact templates, 'other' for anything else.
export function checkFact(sentence) {
  let m = /^([a-z]+) likes ([a-z]+)\.$/.exec(sentence);
  if (m) {
    if (!WORLD.people.includes(m[1]) || !WORLD.foods.includes(m[2])) return 'other';
    return WORLD.likes[m[1]] === m[2] ? 'true' : 'false';
  }
  m = /^([a-z]+) lives (by|in) the ([a-z]+)\.$/.exec(sentence);
  if (m) {
    if (!WORLD.people.includes(m[1]) || !WORLD.places.includes(m[3])) return 'other';
    if (WORLD.prep[m[3]] !== m[2]) return 'other';
    return WORLD.livesIn[m[1]] === m[3] ? 'true' : 'false';
  }
  return 'other';
}
