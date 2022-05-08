Reproduction of Deep learning Heathcare project

Citation to the original paper:

```
@ARTICLE{Patient2Vec,
      author={Zhang, Jinghe and Kowsari, Kamran and Harrison, James H and Lobo, Jennifer M and Barnes, Laura E},
      journal={IEEE Access},
      title={Patient2Vec: A Personalized Interpretable Deep Representation of the Longitudinal Electronic Health Record},
      year={2018},
      volume={6},
      pages={65333-65346},
      doi={10.1109/ACCESS.2018.2875677},
      ISSN={2169-3536}
}
```

original repo: https://github.com/BarnesLab/Patient2Vec

---
Dependencies: 
---

pytorch, pandas

---
Data download instruction: 
---

MIMIC III data, put unziped folder under project root.

---
Preprocessing code + command:
---

 `python3 patient2vec.py --preprocess_data`
 
---
Training code + command (if applicable): // Run after data preprocess
---

Patient2Vec Model: `python3 patient2vec.py --Patient2Vec` 
MLP Model: `python3 patient2vec.py --MLP` 
LR Model: `python3 patient2vec.py --LR` 
GRU based RNN Model: `python3 patient2vec.py --GRU` 
LSTM based RNN Model: `python3 patient2vec.py --LSTM` 
Bidirectional RNN Model: `python3 patient2vec.py --BiRNN` 

---
Evaluation code + command (if applicable): 
---

evaluation is done after training. Same as training cmds.

---
Table of results (no need to include additional experiments, but main reproducibility result should be included)
---

`acc: Accuracy, p: Precision, r: Recall, f: F-beta score, roc_auc: AUC score`

MLP: Validation acc: 0.71, p: 0.79, r:0.71, f: 0.75, roc_auc: 0.78

Logreg: Validation acc: 0.71, p: 0.79, r:0.71, f: 0.75, roc_auc: 0.77

GRU RNN: Validation acc: 0.70, p: 0.78, r:0.72, f: 0.75, roc_auc: 0.78

LSTM RNN Validation acc: 0.70, p: 0.78, r:0.72, f: 0.75, roc_auc: 0.76

Bidirectional RNN: Validation acc: 0.70, p: 0.81, r:0.68, f: 0.74, roc_auc: 0.78

Patient2Vec: Validation acc: 0.72, p: 0.82, r:0.69, f: 0.74, roc_auc: 0.78
