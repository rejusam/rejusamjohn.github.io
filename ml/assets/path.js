// The full learning path. L0 to L5 are live; L6 shows the road ahead
// without links. Lesson pages live at /ml/<slug>/.

export const PATH = [
  {
    level: 'L0',
    name: 'Start here',
    lessons: [{ slug: 'what-is-a-model', title: 'What is a model?' }],
  },
  {
    level: 'L1',
    name: 'Data & intuition',
    lessons: [
      { slug: 'data-as-numbers', title: 'Data as numbers' },
      { slug: 'centre-and-spread', title: 'Centre and spread' },
      { slug: 'measuring-error', title: 'Measuring error' },
      { slug: 'slopes', title: 'Slopes and derivatives' },
    ],
  },
  {
    level: 'L2',
    name: 'Learning from data',
    lessons: [
      { slug: 'linear-regression', title: 'Linear regression' },
      { slug: 'gradient-descent', title: 'Gradient descent' },
      { slug: 'loss-surface', title: 'Many features and the loss surface' },
      { slug: 'train-test', title: 'Train, validate, test' },
      { slug: 'overfitting', title: 'Overfitting and regularisation' },
    ],
  },
  {
    level: 'L3',
    name: 'Classic ML',
    lessons: [
      { slug: 'k-nearest-neighbours', title: 'k-nearest neighbours' },
      { slug: 'logistic-regression', title: 'Logistic regression' },
      { slug: 'naive-bayes', title: 'Naive Bayes' },
      { slug: 'support-vector-machines', title: 'Support vector machines' },
      { slug: 'decision-trees', title: 'Decision trees' },
      { slug: 'random-forests', title: 'Bagging and random forests' },
      { slug: 'gradient-boosting', title: 'Gradient boosting' },
      { slug: 'k-means', title: 'k-means clustering' },
      { slug: 'pca', title: 'Principal component analysis' },
    ],
  },
  {
    level: 'L4',
    name: 'Deep learning',
    lessons: [
      { slug: 'neurons', title: 'The artificial neuron' },
      { slug: 'hidden-layers', title: 'Hidden layers: bending the boundary' },
      { slug: 'backpropagation', title: 'Backpropagation' },
      { slug: 'training-networks', title: 'Training a network' },
      { slug: 'optimisers', title: 'Optimisers: momentum and Adam' },
      { slug: 'deep-networks', title: 'Going deeper: vanishing gradients' },
      { slug: 'convolutions', title: 'Convolutions' },
      { slug: 'cnns', title: 'Convolutional networks' },
      { slug: 'rnns', title: 'Recurrent networks' },
    ],
  },
  {
    level: 'L5',
    name: 'Modern AI',
    lessons: [
      { slug: 'tokens', title: 'Tokens: text as numbers' },
      { slug: 'embeddings', title: 'Embeddings: words as points' },
      { slug: 'language-models', title: 'Predicting the next token' },
      { slug: 'attention', title: 'Attention' },
      { slug: 'transformers', title: 'Transformers' },
      { slug: 'llms', title: 'Large language models, in miniature' },
      { slug: 'generative-models', title: 'Generative models' },
      { slug: 'diffusion', title: 'Diffusion: from noise to data' },
    ],
  },
  { level: 'L6', name: 'In practice', lessons: [], soon: true, topics: 'Evaluation, calibration, interpretability, fairness' },
];

export const TOTAL_LIVE = PATH.flatMap((l) => l.lessons).length;

export const lessonHref = (slug) => `/ml/${slug}/`;

// Finds a lesson and its level by slug, or null.
export function findLesson(slug) {
  for (const level of PATH) {
    const i = level.lessons.findIndex((l) => l.slug === slug);
    if (i !== -1) return { level, lesson: level.lessons[i] };
  }
  return null;
}

// The lesson after this one along the path, or null at the end.
export function nextLesson(slug) {
  const all = PATH.flatMap((l) => l.lessons);
  const i = all.findIndex((l) => l.slug === slug);
  return i === -1 || i === all.length - 1 ? null : all[i + 1];
}

// The first lesson along the path that is not done, or null when all are.
// isDone(slug) says whether a lesson is done.
export function firstNotDone(isDone) {
  for (const level of PATH) {
    for (const lesson of level.lessons) if (!isDone(lesson.slug)) return lesson;
  }
  return null;
}

// The landing page's one start button. With no lesson done it starts at L0;
// part way through it continues at the first lesson not done; with every
// lesson done it offers L0 again. isDone(slug) says whether a lesson is done.
export function startLink(isDone) {
  const all = PATH.flatMap((level) => level.lessons);
  const first = all[0];
  const atStart = { label: `Start at L0: ${first.title}`, href: lessonHref(first.slug) };
  if (!all.some((l) => isDone(l.slug))) return atStart;
  const next = firstNotDone(isDone);
  if (!next) return { label: `Start again at L0: ${first.title}`, href: lessonHref(first.slug) };
  return { label: `Continue: ${next.title}`, href: lessonHref(next.slug) };
}
