
|     |     opt                 | STATUS | MT BLEU  | <-step | ASR BLEU  | <-step  | SLT BLEU  | <-step  |  train_logNAME.err                      |
|-----|-------------------------|--------|----------|--------|-----------|---------|-----------|---------|-----------------------------------------|
|  MT:|
| rnn |   ENtRNN2DEtRNN         |  done  |    21.68 |   140K |           |         |           |         |  rnn_ent2det_1584239.err                |
| txt |   ENDEtRNN2DEtRNN       |  done  |    20.07 |   130K |           |         |           |         |  rnn_ent2det_1584234.err                |
| --- |-------------------------|--------|----------|--------|-----------|---------|-----------|---------|-----------------------------------------|
|     |   ENt2DEt               |  done  |    22.28 |   120K |           |         |           |         |  trf_ent2det_1383686.err                |
| trf |   ENtDEt2DEt            |  done  |    22.31 |   120K |           |         |           |         |  trf_entdet2det_1415543.err             |
| txt |   ENDEt2DEt             |  done  |    25.17 |   120K |           |         |           |         |  trf_endet2det_1415546.err              |
|     |   ENDEt2DEt_Mtrf        |  done  |    13.62 |   210K |           |         |           |         |  trfM_endet2det_{1437364,1456588}.err   |
| --- |-------------------------|--------|----------|--------|-----------|---------|-----------|---------|-----------------------------------------|
|  MIKKOs
|     |   ENt2DEt.50m           | done  |    24.11  |  210K  |           |         |           |         |                                         |
|     |   ENt2DEt_Mtrf.50m      | done  |    14     |  210K  |           |         |           |         |                                         |
|     |   ENt2DEt.43m           | done  |    22.68  |  210K  |           |         |           |         |                                         | 
|     |   ENt2DEt_Mtrf.43m      | done  |    12     |  210K  |           |         |           |         |                                         |
|     |   ENDEt2DEt.50m         | done  |   21.01   |  ~150K |           |         |           |         |                                         |
|     |   ENDEt2DEt.43m         | done  |   21.42   |  ~150K |           |         |           |         |                                         |
|  UMUTs 
|     |   marian_en2de.50M     | done  |  ~35      |         |           |         |           |         |                                         |
|(mustC)|   marian_en2de.n00.  | done  |   27.16   |         |           |         |           |         |                                         |
|(mustC)|   marian_en2de.n15.  | done  |   28.44   |         |           |         |           |         |                                         |
|(mustC)|   marian_ende2de.n00.| done  |   28.27   |         |           |         |           |         |                                         |
|(mustC)|   marian_ende2de.n15.| done  |   29.31   |         |           |         |           |         |                                         |
| --- |------------------------|-------|-----------|---------|-----------|---------|-----------|---------|-----------------------------------------|
| ASR:|
|     |   ENa2ENt         |   done     |           |         |    1.51   | 130K    |           |         |  trf_ena2det_1533364.err                |
| trf |   ENa2ENt_posenc  | not-convrgd|           |         |    8.49   | 250K    |           |         |  trf_ena2entPE_1603257.err              |
| aud |   ENa2ENtRNN      |   done     |           |         |    2.10   | 150K    |           |         |  trfrnn_ena2det_1603275.err             |
|     |   ENaENt2ENt      | running    |           |         |    3.55   | 290K    |           |         |  trf_enaent2ent_1612343.err             |
| --- |-------------------|------------|-----------|---------|-----------|---------|-----------|---------|-----------------------------------------|
|     |   ENaRNN2ENt      |   done     |           |         |   30.40   | 170K    |           |         |  rnntrf_ena2det_1545021.err             |
|train0| ENaRNN2ENtRNN    |   done     |           |         |   32.74   | 150K    |           |         |  err_train_358994                       |
|     |  ENaRNNENt2ENt    | running    |           |         |           |         |           |         |  ntrftrf_enaent2ent__1629630.err        |
| --- |-------------------|------------|-----------|---------|-----------|---------|-----------|---------|-----------------------------------------|
| SLT:|                   |            |           |         |           |         |           |         |                                         |
|     |   train3          |   done     |           |         |           |         |   5.00    |   380K  |  err_train_                             |
|     |   train5shareEnc  |   done     |           |         |           |         |   4.94    |   250K  |  err_train_                             |
| rnn |   train2shareEnc  |   done     |           |         |           |         |   4.84    |   250K  |  err_train_                             |
| aud |   train4          |   done     |           |         |           |         |   4.50    |   300K  |  err_train_                             |
|     |   train1          |   done     |           |         |           |         |   4.30    |   190K  |  err_train_                             |
|     |   train4          |   done     |           |         |           |         |   3.62    |   190K  |  err_train_                             |




|     |     opt                 | STATUS | ASR BLEU  | MT BLEU  | <-step  |  train_logNAME.err                      |
|-----|-------------------------|--------|-----------|-----------|---------|-----------------------------------------|
|     | 3en.3dec.6skmel.100ah      |  done  |   8.49   |         | 250K |   atrf_3en.3dec.6skmel.100ah_1610713.err     |
|     | 3en.3dec.4skmel.100ah      |  done  |  12.03   |         | 250K |   atrf_3en.3dec.4skmel.50ah_1612139.err      |
|     | 2enc.3dec.4skmel.100ah     |  done  |  13.85   |         | 280K |  atrf_2enc.3dec.4skmel.100ah_1629127.err     |
|     | 2enc.3dec.4skmel.0ah       |  done  |  14.34   |         | 280K |   atrf_2enc.3dec.4skmel.0ah_1643986.err      |
|     | trf_ENaENt2DEtENt          |  done  |   0.51   |         | 270K |   trf_ENaENt2DEtENt.3en.3dec.4skmel.100ah    |
| --- |----------------------------|--------|----------|-------|-------------------------|
| R-Transformer:
|     | 2enc.3dec.4skmel.100ah.layernorm  |  done  |   0.86   |         | 240K |   atrf_2enc.3dec.4skmel.100ah_1655912.err     |                                               
|     | 2enc.3dec.4skmel.0ah       |  done  |   7.50   |         | 280K |   atrf_2enc.3dec.4skmel.0ah_1655914.err       |
|     | 2enc.3dec.4skmel.100ah     |  done  |          |         |      |   atrf_2enc.3dec.4skmel.100ah_1664046.err     |
|     | 2enc.3dec.4skmel.100ah     |  done  |   5.70   |         | 280K |   atrf_2enc.3dec.4skmel.100ah_1664046.err                                                    
|     |2enc.3dec.4skmel.100ah.accum|  done  |29.9/32.4 |         |44/112k|   atrf_2enc.3dec.4skmel.100ah.accum_{1675765,1676309}.err |
|     | FBKfairseq-MUSTc           |  done  |  56.73   |   0.0!? |      |bleu score in outfile: asr=61.31  slt=0.0        |
| all with accum:
|     | 3enc.3dec.4skmel.100ah     |  done  |  32.12   |         |  44K |   atrf_6enc.3dec.4skmel.100ah_1694492         |
|     | 3enc.6dec.4skmel.100ah     |  done  |  30.87   |         |  44K |   atrf_6enc.6dec.4skmel.100ah_1694493         |
