# Demo multi30k

Multilingual RNN model:

- basic RNN architecture, 2 layer LSTM, 4 encoders, 4 decoders, hidden size 512, embedding size 128, vocab max_size 20000

- trained on de-cs fr-cs de-en fr-en cs-de en-de fr-de cs-fr de-fr en-fr

- multi30k multilingual dataset for de, cs, fr, en, text only

- 50k iterations

## Live demo with our client

```
./demo-translation-client.sh --from en --to de hello world
en -> de:
der fallschirmspringer ist in der luft .
```



## Live demo -- easily from anywhere

The translation server runs on u-pl0.ms.mff.cuni.cz:5000. Use your CURL to access it:

```
en -> cs : 
curl -X POST -i http://u-pl0.ms.mff.cuni.cz:5000/translator/translate --data '[{"src": "hello world", "id":1}]'
<unk> závodního <unk> .
de -> cs : 
curl -X POST -i http://u-pl0.ms.mff.cuni.cz:5000/translator/translate --data '[{"src": "hello world", "id":0}]'
chlap na kolečkových bruslích .
fr -> cs : 
curl -X POST -i http://u-pl0.ms.mff.cuni.cz:5000/translator/translate --data '[{"src": "hello world", "id":2}]'
ve své <unk> jídlo .
cs -> en : 
curl -X POST -i http://u-pl0.ms.mff.cuni.cz:5000/translator/translate --data '[{"src": "hello world", "id":3}]'
a martial arts tournament .
de -> en : 
curl -X POST -i http://u-pl0.ms.mff.cuni.cz:5000/translator/translate --data '[{"src": "hello world", "id":4}]'
men in a skateboard match .
fr -> en : 
curl -X POST -i http://u-pl0.ms.mff.cuni.cz:5000/translator/translate --data '[{"src": "hello world", "id":5}]'
in the crowded stadium .
cs -> de : 
curl -X POST -i http://u-pl0.ms.mff.cuni.cz:5000/translator/translate --data '[{"src": "hello world", "id":6}]'
ärzte helfen einem neugeborenen fallschirm .
en -> de : 
curl -X POST -i http://u-pl0.ms.mff.cuni.cz:5000/translator/translate --data '[{"src": "hello world", "id":7}]'
pendler beim halt .
fr -> de : 
curl -X POST -i http://u-pl0.ms.mff.cuni.cz:5000/translator/translate --data '[{"src": "hello world", "id":8}]'
in der nacht im freien .
cs -> fr : 
curl -X POST -i http://u-pl0.ms.mff.cuni.cz:5000/translator/translate --data '[{"src": "hello world", "id":9}]'
un tournoi de natation .
en -> fr : 
curl -X POST -i http://u-pl0.ms.mff.cuni.cz:5000/translator/translate --data '[{"src": "hello world", "id":11}]'
la tête d&apos; un accident
de -> fr : 
curl -X POST -i http://u-pl0.ms.mff.cuni.cz:5000/translator/translate --data '[{"src": "hello world", "id":10}]'
des artistes en train de travailler .
```

## Launch the translation server

- clone this repo, install pytorch, OpenNMT-py in this version, and flask in your python environment

- save the model checkpoint file into the repo root dir, name it as indicated in `./available_models/conf.json`, or update the conf

- `python3 server.py` launches the server on your machine, port 5000

- access it through our client or CURL

