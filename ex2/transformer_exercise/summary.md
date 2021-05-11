Student 1:
* Name: Morris Alper
* ID: 341438711
* Username: morrisalper

Student 2:
* Name: Naama Yochai
* ID: 208769737
* Username: naamayochai

Student 3:
* Name: Shlomo Tannor
* ID: 314389248
* Username: shlomotannor

Now, for each log file that you need to submit, you will need to write its last 3 lines. For example, this is what we got for `baseline_gen.log`:
```txt
2021-04-24 17:20:46 | INFO | fairseq_cli.generate | NOTE: hypothesis and token scores are output in base 2
2021-04-24 17:20:46 | INFO | fairseq_cli.generate | Translated 7,283 sentences (165,025 tokens) in 18.5s (394.00 sentences/s, 8927.61 tokens/s)
Generate valid with beam=5: BLEU4 = 33.39, 69.1/42.8/28.5/19.4 (BP=0.934, ratio=0.937, syslen=138824, reflen=148229)
```

3 last lines from the baseline_train.log file: 
```txt
2021-05-10 14:26:15 | INFO | fairseq_cli.train | end of epoch 50 (average epoch stats below)
2021-05-10 14:26:15 | INFO | train | epoch 050 | loss 3.943 | nll_loss 2.514 | ppl 5.71 | wps 35189.2 | ups 3.38 | wpb 10419.8 | bsz 422.8 | num_updates 18950 | lr 0.000229718 | gnorm 0.625 | train_wall 65 | gb_free 8.9 | wall 5706
2021-05-10 14:26:15 | INFO | fairseq_cli.train | done training in 5705.4 seconds
```

3 last lines from the baseline_gen.log file: 
```txt
2021-05-11 00:22:32 | INFO | fairseq_cli.generate | NOTE: hypothesis and token scores are output in base 2
2021-05-11 00:22:32 | INFO | fairseq_cli.generate | Translated 7,283 sentences (167,441 tokens) in 40.9s (177.85 sentences/s, 4088.99 tokens/s)
Generate valid with beam=5: BLEU4 = 33.46, 68.5/42.3/28.1/19.1 (BP=0.948, ratio=0.949, syslen=140718, reflen=148229)
```

3 last lines from the baseline_mask.log file: 
```txt
2021-05-11 00:30:46 | INFO | fairseq_cli.generate | NOTE: hypothesis and token scores are output in base 2
2021-05-11 00:30:46 | INFO | fairseq_cli.generate | Translated 7,283 sentences (168,548 tokens) in 59.9s (121.52 sentences/s, 2812.37 tokens/s)
Generate valid with beam=5: BLEU4 = 32.27, 67.1/40.8/26.6/17.8 (BP=0.956, ratio=0.957, syslen=141858, reflen=148229)
```

25 last lines from the check_all_masking_options.log file: 
```txt
2021-05-11 02:10:39 | INFO | fairseq_cli.generate | NOTE: hypothesis and token scores are output in base 2
2021-05-11 02:10:39 | INFO | fairseq_cli.generate | Translated 7,283 sentences (167,413 tokens) in 58.5s (124.53 sentences/s, 2862.48 tokens/s)
Generate valid with beam=5: BLEU4 = 33.28, 68.3/42.1/27.9/18.9 (BP=0.949, ratio=0.950, syslen=140802, reflen=148229)
table of score with masking enc-enc attention head
rows are transformer layer number and columns are head number
       0      1      2      3
0  33.36  33.33  33.27  33.22
1  33.24  33.37  33.37  33.19
2  33.34  32.89  32.15  33.35
3  33.31  33.12  29.32  33.43
table of score with masking enc-dec attention head
rows are transformer layer number and columns are head number
       0      1      2      3
0  33.32  33.24  17.89  33.12
1  33.02  33.26  33.11  33.27
2  33.18  33.24  32.93  32.72
3  32.27  32.88  32.52  32.62
table of score with masking dec-dec attention head
rows are transformer layer number and columns are head number
       0      1      2      3
0  33.38  33.35  33.34  33.34
1  33.37  33.34  33.28  33.36
2  33.34  33.39  32.95  33.36
3  33.38  33.43  33.32  33.28
```

3 last lines from the sandwich_train.log file: 
```txt
<write here>
```

3 last lines from the sandwich_gen.log file: 
```txt
<write here>
```
