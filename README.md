To compute the dot product between two columns $\mathbf{p}_j$ and $\mathbf{p}_{j'}$ (positions $j$ and $j'$), we pairwise multiply elements and sum them:


$$\mathbf{p}_j$$

## Positional Encoding

### Calculation
The positional encoding $\text{PE}(i, j)$ is defined for each embedding dimension $i$ and token position $j$ as:

$$
\text{PE}(i, j) =
\begin{cases}
\sin\left(\frac{j}{10000^{i/d}}\right), & \text{if } i \text{ is even}, \\
\cos\left(\frac{j}{10000^{(i-1)/d}}\right), & \text{if } i \text{ is odd}.
\end{cases}
$$

Here:
- $j$ is the token's position in the sequence.
- $i$ is the embedding dimension index.
- $d$ is the embedding dimensionality.

This creates a matrix $\mathbf{P}$ where each column represents a token’s positional encoding.

---

### Why It Works

## 1. Column Construction (Position as Column Index)

Assume the positional encoding matrix $\mathbf{P}$ has:
- **$d$ rows** (each row corresponds to a dimension of the embedding),
- **$n$ columns** (each column corresponds to a position $j$ in the sequence).

For position $j$, the column $\mathbf{p}_j$ looks like

$$\mathbf{p}_j = \bigl[\sin\bigl(\tfrac{j}{\alpha_0}\bigr),\,\cos\bigl(\tfrac{j}{\alpha_0}\bigr),\,\sin\bigl(\tfrac{j}{\alpha_1}\bigr),\,\cos\bigl(\tfrac{j}{\alpha_1}\bigr),\,\dots\bigr]^\top$$

where each $\alpha_k$ (often $10000^{k/d}$) controls the wavelength of the $k$-th sine/cosine pair.

---

## 2. Dot Product Involves Sine-Cosine Products

To compute the dot product between two columns $\mathbf{p}_j$ and $\mathbf{p}_{j'}$ (positions $j$ and $j'$), we pairwise multiply elements and sum them:

$$\mathbf{p}_j^\top\mathbf{p}_{j'} = \sum_{k\in\text{(sine/cos pairs)}}\bigl[\sin\bigl(\tfrac{j}{\alpha_k}\bigr)\sin\bigl(\tfrac{j'}{\alpha_k}\bigr) + \cos\bigl(\tfrac{j}{\alpha_k}\bigr)\cos\bigl(\tfrac{j'}{\alpha_k}\bigr)\bigr].$$

Using the trigonometric identity

$$\sin(A)\sin(B)+\cos(A)\cos(B)=\cos(A-B),$$

each sine-cosine pair becomes

$$\cos\bigl(\tfrac{j}{\alpha_k}-\tfrac{j'}{\alpha_k}\bigr)=\cos\Bigl(\tfrac{j-j'}{\alpha_k}\Bigr).$$

---

## 3. Dependence on $(j-j')$

Summing across all frequencies $\alpha_k$ yields terms of the form $\cos\bigl(\tfrac{j-j'}{\alpha_k}\bigr)$. Hence, the dot product depends on the difference $(j-j')$:

$$\mathbf{p}_j \mathbf{p}_{j'} = \sum_{k}\cos\Bigl(\frac{j-j'}{\alpha_k}\Bigr).$$

Because this expression depends only on $(j-j')$ and **not** on $j$ or $j'$ separately, it encodes the **relative distance** between these two positions in the sequence.

---

## 4. Why This Matters

- **Relative Position Encoding**: Since $\mathbf{p}_j^\top\mathbf{p}_{j'}$ is a function of $(j-j')$, the model inherently captures how far apart two positions are.

- **Multi-Scale Representation**: Each frequency $\alpha_k$ contributes a different periodicity, enabling both **local** and **global** positional differences to be represented.

- **Generalization**: Because the encoding is based on sinusoids, a transformer can handle **varying sequence lengths** and still interpret relative positions consistently, even for positions not seen during training.

By aligning each **column** with a position $j$ in the sequence, we see that the **dot product** between columns depends on $(j-j')$. This emerges from the trigonometric identities that convert products of sine and cosine terms into functions of **phase differences**, effectively encoding **relative** positional information.---
