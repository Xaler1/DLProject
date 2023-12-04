# Observations
## 12-03-2023

Debugging thre transformer initial model. Noticed a bunch of peculiarities: 

 - The positional encoding requires the data to be sent as (seq_len, batch_size, embedding_dim/channels)
 - x.mean(dim = 1) in line 81 is actually across the samples, which is incorrect. 