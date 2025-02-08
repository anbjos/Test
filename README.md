### Attenuation

$Y_M$ is the model's estimate of a noise-free signal in the MEL domain. To reconstruct a noise-free audio signal, we must find an attenuation vector 

$$
\text{attenuation}\in\mathbb{R}^m
$$

that can be applied to the original signal 

$$
X_{\mathbb{C}}\in\mathbb{C}^{m\times n}
$$

so that, when the same signal chain up to the model is applied, the result is identical to $Y_M$. This requirement is equivalent to:

$$
Y_P=M\cdot\text{Diagonal}(\text{attenuation})\cdot X_P
$$

Considering a pair of columns, $y_P$ and $x_P$, in $Y_P$ and $X_P$ respectively, we can write:

$$
y_P=M\cdot\text{Diagonal}(\text{attenuation})\cdot x_P
$$

However, since $y_P\in\mathbb{R}^{d_U}$ and $m\ge d_U$, our system is underdetermined; there are many attenuation vectors that can produce the desired outcome. One way to address this is by considering the MEL transformation as a dimensionality reduction from $m$ to $d_U$. In this view, each element in the MEL representation aggregates information from several bins of the original representation.

To pursue this approach, we first define the matrix 

$$
\tilde{M}=\text{RegularRows}(M),\quad\tilde{M}\in\mathbb{R}^{d_U\times\tilde{m}}

$$
by removing all all-zero columns from $M$, thereby excluding the bins that are not seen by the transformer, and similarly,

$$
\tilde{u}\_P=\text{RegularRows}(u_P,M),\quad\tilde{u}\_P\in\mathbb{R}^{\tilde{m}}

$$
where $\tilde{m}$ represents the number of nonzero rows in $M$.

Next, we introduce a vector 

$$
w\in\mathbb{R}^{\tilde{m}}
$$

such that $\tilde{M}\cdot w$ represents the desired attenuation. We express the relationship as:

$$
y=\tilde{M}\cdot\text{Diagonal}(\tilde{M}^{T}w)\cdot\tilde{u}\_P
$$

where $\text{Diagonal}(\tilde{M}^{T}w)$ applies an attenuation to the vector $\tilde{u}\_P$. Since $u_P$ represents power, this formulation ensures that the attenuated version maintains the same power output. We can rewrite this as a linear system:

$$
y=\tilde{M}\cdot\text{Diagonal}(\tilde{M}^{T}\tilde{u}\_P)\cdot w
$$

From $w$, we derive the desired attenuation:

$$
\text{PowerAttenuation}=\text{ReconstructRows}\Bigl(\text{clamp}(\tilde{M}^{T}w,0,1),\,M\Bigr)
$$

Here, the function $\text{clamp}(\cdot,0,1)$ ensures that the attenuation values remain between 0 and 1, and the function $\text{ReconstructRows}(\cdot,M)$ reinserts zeros at the positions of the discarded rows in $M$.

Finally, the overall attenuation is obtained by taking the element-wise square root of the power attenuation:

$$
\text{attenuation}=\bigl[\sqrt{\text{PowerAttenuation}_{i,j}}\bigr]
$$

which is then applied to the corresponding column of $X_{\mathbb{C}}$ to create $Y_{\mathbb{C}}$.






$$
p_j^{\top} p_{j'}  = \sum_{k} \cos\Bigl(\frac{j - j'}{\alpha_k}\Bigr).
$$  


$$
p_j^{\top} p_{j'}
$$

$$
p_j^T p_{j'}
$$



$$
p_j^{\top} p_{j'} = \sum_{k} \bigl[\sin\bigl(\tfrac{j}{\alpha_k}\bigr)\sin\bigl(\tfrac{j'}{\alpha_k}\bigr) + \cos\bigl(\tfrac{j}{\alpha_k}\bigr)\cos\bigl(\tfrac{j'}{\alpha_k}\bigr)\bigr].
$$

$$
\mathbf{p}_j = \bigl[\sin\bigl(\tfrac{j}{\alpha_0}\bigr),\,\cos\bigl(\tfrac{j}{\alpha_0}\bigr),\,\sin\bigl(\tfrac{j}{\alpha_1}\bigr),\,\cos\bigl(\tfrac{j}{\alpha_1}\bigr),\,\dots\bigr]^\top
$$  


The **Spectral Transformer** suppresses audio noise through the following pipeline:

1. **Embedding and Positional Encoding**

$$X=\text{PositionalEncoding}(\text{Embedding}(U))$$

2. **Transformer Processing and Output Projection**

$$Y=\text{OutputProjection}(\text{Transformer}(X))$$

**Key Points:**

- $W\_{\text{emb}}$ and $W\_{\text{out}}$ are **independently learned**.
- Masking is included to support future enhancements, such as handling variable-length inputs.

```julia
struct Model
    position_embedding
    projection
    withmask 
    transformer
    antiprojection
end
 
model=Model(position_embedding, projection_layer, withmask, encoding_transformer, antiprojection_layer)
@functor Model (projection, transformer, antiprojection)

function (model::Model)(input)
    position= model.position_embedding(input.hidden_state)
    projection = model.projection(input.hidden_state)
    transformed=model.transformer( (hidden_state=projection .+ position, attention_mask=input.attention_mask) )
    result=(hidden_state=model.antiprojection(transformed.hidden_state),attention_mask=transformed.attention_mask) 
    return result
end
```