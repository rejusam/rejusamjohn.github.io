// Made-up points around one hump-and-dip of a sine wave, shared by
// Lessons 9 and 10. Pure: no DOM.
//
// For each point in turn, x ~ Uniform(−1, 1) is drawn first, then
// y = sin(πx) + sd × a standard normal draw, all from mulberry32(seed).

import { mulberry32, uniform, randn } from './maths.js';

export const truth = (x) => Math.sin(Math.PI * x);

export function sineData({ n, sd = 0.25, seed }) {
  const rng = mulberry32(seed);
  const xs = [];
  const ys = [];
  for (let i = 0; i < n; i++) {
    const x = uniform(rng, -1, 1);
    xs.push(x);
    ys.push(truth(x) + sd * randn(rng));
  }
  return { xs, ys };
}
