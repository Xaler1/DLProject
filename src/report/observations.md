# Observations
## 12-03-2023

Debugging thre transformer initial model. Noticed a bunch of peculiarities: 

 - The positional encoding requires the data to be sent as (seq_len, batch_size, embedding_dim/channels)
 - x.mean(dim = 1) in line 81 is actually across the samples, which is incorrect. 

 ## 12-06-2023

 Exploring the dataset
 
 Explored the individual label inbalance. The structure of the labels is as follows:

1. Diagnostic: 44 total labels
    Main Classes, each with sub-labels
    - Normal
    - MI: Myocardial Infarction
    - CD: Conduction Disturbance
    - STTC: ST/T-Changes
    - HYP: Hypertrophy

2. Form: 19 form, 4 common with diagnostic

    - 

3. Rhythm: 12 labels:

    - Related to changes such as Arhythmia and Atrial Fibrillation

![]('/home/anirudhkailaje/Documents/01_UPenn/01_ESE5460/03_Project/src/report/LabelDistribution.png')

Distribution of Gender: 52% male and 48% female.

Benchmarking Our Models through macro-averaged ROC-AUC. Why? Two Reasons.

- There's a lot of class imbalance The same metric is used in the PTB-XL benchmarks.