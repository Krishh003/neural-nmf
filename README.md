# Neural NMF

Run from the repository root using Python 3.10 and PowerShell.

```powershell
python -m pip install -r requirements.txt
python neural_nmf.py --dataset olist
python neural_nmf.py --dataset newsgroups
```

Jamie Haddock, Tyler Will, and colleagues, 2024 journal volume. [Original paper](https://link.springer.com/article/10.1007/s43670-023-00077-3) · [Open preprint](https://arxiv.org/abs/2303.00058) · [Authors' code](https://github.com/jvendrow/NeuralNMF) · [Group's Olist notebook](https://github.com/rushika1256/NeuralNMF)

`neural_nmf.py` uses the authors' differentiable nonnegative least-squares layers with two trainable dictionaries (ranks 10 and 5). Sequential NMF initializes the dictionaries, followed by one joint projected Adam update per epoch. The supervised heads start from a least-squares fit using observed training labels. It learns unsupervised, with 50% of the training labels, and with all training labels. It reports reconstruction error, top entries in each topic, and held-out accuracy for the two supervised modes. The unlabeled test orders are held out from all model fits. Unreviewed orders are omitted from the Olist sample.

The default uses 500 orders and eight epochs. To use the slides' sample size and epoch count, run `python neural_nmf.py --dataset olist --sample 5000 --epochs 20`. Accuracy still uses a held-out split, so the reported percentages from the group notebook use a different evaluation protocol. Run `python neural_nmf.py --dataset newsgroups` for a five-category experiment on the paper's 20 Newsgroups corpus. scikit-learn downloads the corpus on first use. For an existing extracted corpus, pass `--news-dir` with the folder containing `20news-bydate-train` and `20news-bydate-test`.

If moving the project to another machine, place the order-items, products, order-reviews, and product-category-translation CSVs from the [Olist dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) inside `data/olist`. The local copies came from [this public mirror](https://github.com/spdrio/Brazilian-E-Commerce-Public-Dataset-by-Olist/tree/master/files).
