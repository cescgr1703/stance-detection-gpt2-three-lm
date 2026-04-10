# stance-detection-gpt2-three-lm

This project adapts the Authorial Language Model (ALM) approach of Huang et al. (2025), 
originally proposed for authorship attribution, to political stance detection. Instead of 
fine-tuning one GPT-2 per author, we fine-tune one per stance class (AGAINST, FAVOR, NEUTRAL) 
and classify by argmin perplexity. Tested on the Catalonia Independence Corpus (CIC-ES), 
the approach achieves an F1avg of 0.72, within 2.6 points of the best published benchmark (mBERT).

A key advantage over discriminative approaches is built-in token-level explainability: since 
classification is based on per-token log-likelihood, each token's contribution to the predicted 
class is directly interpretable at no additional computational cost.

## Results

F1avg is the average of F1-AGAINST and F1-FAVOR, following
the evaluation protocol of Zotova et al. (2021).

| model                           | f1avg       | macro_f1 | f1_against | f1_favor | f1_neutral |
| ------------------------------- | ----------- | -------- | ---------- | -------- | ---------- |
| GPT-2 Three-LM (ours)           | 0.7213      | 0.7425   | 0.7267     | 0.7159   | 0.7849     |
| TF-IDF + SVC (ours)             | 0.7184      | 0.7513   | 0.7216     | 0.7152   | 0.8170     |
| mBERT (Zotova et al.)           | 0.7472      | —        | 0.7517     | 0.7417   | —          |
| XLM-R (Zotova et al.)           | 0.7357      | —        | 0.7468     | 0.7245   | —          |
| FTEmb + fastText (Zotova et al.)| 0.7243      | —        | 0.7320     | 0.7113   | —          |
| TF-IDF + SVM (Zotova et al.)    | 0.7109      | —        | 0.7150     | 0.7109   | —          |

## Setup
The notebook is designed to run on Google Colab with a GPU runtime (tested on T4).

1. Clone the repo and open `gpt2_three_lm_argmin.ipynb` in Colab
2. Download the CIC-ES dataset from [Zotova et al. (2020)]((https://github.com/ixa-ehu/catalonia-independence-corpus)) and place the splits under `data/`
3. Update the path constants in the configuration cell before running

## Citation

```bibtex
@misc{gomez2026,
  author = {Cesc Gómez and Mustafa Can Buken and Ahmad Kamran and Shahmir Khan and Tuna Cemal Erdem},
  title  = {Stance Detection via Perplexity-Based Approach},
  year   = {2026},
  url    = {https://github.com/cescgr1703/stance-detection-gpt2-three-lm}
}
```
## Use of AI

This project was developed with the assistance of AI tools: Codex was used to generate a first working version of the code, and Claude (Anthropic) was used for code review, debugging, and discussion of methodological decisions.
