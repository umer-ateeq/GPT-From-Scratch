# Sample completions

From `weights8b_300epoch.pth` on CPU. Temperature 0.8, top-k 50, 60 new tokens,
conditioning context 128, seed 123.

Reproduce:

```bash
python generate.py --ckpt weights8b_300epoch.pth --out SAMPLES.md
```

Every prompt in `DEFAULT_PROMPTS` appears below, in source order, from one seeded
run. Nothing is selected after the fact.

**These are not good, and that is the point.** The model is locally fluent and
factually unreliable, and it loops ("the war in France, the wars in France").
That is exactly what 184.96 out-of-domain perplexity predicts, and it is what a
134M-parameter model trained on 1.23B tokens looks like. See
[docs/RESULTS.md](docs/RESULTS.md).

### 'The mitochondria is'

```
The mitochondria is responsible for converting ATP and oxygen to oxygen. The mitochondria are involved in maintaining the balance of cellular energy. However, these types of mitochondria are also involved in the production of ATP.
Understanding cell damage and growth is crucial for maintaining healthy cellular connections. To understand mitochondria, we must first
```

### 'In 1969, the first'

```
In 1969, the first company to develop a fully-equipped truck is the California Carpike. In the early 1970s, the California Carpike was one of the first and most well-known hybrids of the California Carpike, which was found on the California Carpike Company. The first carpike
```

### 'Photosynthesis is the process by which'

```
Photosynthesis is the process by which plants convert carbon dioxide into ATP or ATP. This enzyme is usually produced by the breaking down of DNA molecules. The more ATP is added to the amino acids, the more they can be produced.
The DNA has an electron structure, which is a physical structure that includes amino acids, amino acids,
```

### 'The main causes of the French Revolution were'

```
The main causes of the French Revolution were the war in France, the wars in France, the war in France and the French Revolution in France, the revolution in France, the wars of the Napoleonic Wars, the war in the French Revolution in France, the American Revolution in France and the Great War in France. This was the Great
```

### 'To solve a quadratic equation, you'

```
To solve a quadratic equation, you need to add an equation to your equation.
It is very important to know some of the equations. You need to know some key formulas in order to get them.
You can also use them as a reference when calculating a quadratic equation.
Let’s say that we get
```

### 'Water boils at'

```
Water boils at room temperature should be avoided and under extreme heat conditions.
The key to managing and reducing the symptoms of skin cancer is to stop the process of growing at high temperatures, to prevent the development of cancer and to reduce the risk of developing cancer and the progression of cancer. There are many different types of
```

