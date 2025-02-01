## Model Summary


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