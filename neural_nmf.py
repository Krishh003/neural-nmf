"""Hierarchical topic/basket modeling using the paper authors' NeuralNMF."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from NeuralNMF import Energy_Loss_Func, Neural_NMF
from sklearn.datasets import fetch_20newsgroups, load_files
from sklearn.datasets._twenty_newsgroups import (
    strip_newsgroup_footer, strip_newsgroup_header, strip_newsgroup_quoting,
)
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
NEWS = ["comp.graphics", "rec.autos", "rec.sport.baseball", "sci.med", "talk.politics.guns"]


def olist(sample):
    folder = ROOT / "data" / "olist"
    items = pd.read_csv(folder / "olist_order_items_dataset.csv",
                        usecols=["order_id", "product_id", "order_item_id"])
    products = pd.read_csv(folder / "olist_products_dataset.csv",
                           usecols=["product_id", "product_category_name"])
    translations = pd.read_csv(folder / "product_category_name_translation.csv")
    reviews = pd.read_csv(folder / "olist_order_reviews_dataset.csv",
                          usecols=["order_id", "review_score"]).drop_duplicates("order_id")
    joined = items.merge(products).merge(translations).merge(reviews)
    labels = joined[["order_id", "review_score"]].drop_duplicates("order_id")
    sampled = labels.sample(n=min(sample, len(labels)), random_state=42).set_index("order_id")
    categories = joined[joined.order_id.isin(sampled.index)].pivot_table(
        index="order_id", columns="product_category_name_english",
        values="order_item_id", aggfunc="count", fill_value=0)
    train, test = train_test_split(sampled, test_size=.2, stratify=sampled.review_score, random_state=42)
    return (categories.loc[train.index].to_numpy(), categories.loc[test.index].to_numpy(),
            train.review_score.to_numpy() - 1, test.review_score.to_numpy() - 1,
            categories.columns.to_numpy())


def newsgroups(sample, folder):
    options = {"categories": NEWS, "remove": ("headers", "footers", "quotes"), "random_state": 42}

    def read(split, limit):
        if folder is None:
            data = fetch_20newsgroups(subset=split, **options)
            return data.data[:limit], data.target[:limit]
        data = load_files(folder / f"20news-bydate-{split}", categories=NEWS,
                          load_content=False, random_state=42)
        documents = []
        for name in data.filenames[:limit]:
            text = Path(name).read_text(encoding="latin1")
            documents.append(strip_newsgroup_footer(strip_newsgroup_quoting(strip_newsgroup_header(text))))
        return documents, data.target[:limit]

    train_text, train_labels = read("train", sample)
    test_text, test_labels = read("test", max(sample // 4, 50))
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english", min_df=2)
    train_x = vectorizer.fit_transform(train_text).toarray()
    test_x = vectorizer.transform(test_text).toarray()
    return train_x, test_x, train_labels, test_labels, vectorizer.get_feature_names_out()


def run(dataset, sample, epochs, news_dir=None):
    train_x, test_x, train_y, test_y, terms = olist(sample) if dataset == "olist" else newsgroups(sample, news_dir)
    X = torch.tensor(train_x.T.copy(), dtype=torch.double)
    X_test = torch.tensor(test_x.T.copy(), dtype=torch.double)
    label = torch.as_tensor(train_y, dtype=torch.long)
    # Warm-start both dictionaries with fast sequential NMF before joint refinement.
    initial = []
    representation = train_x
    for rank in (10, 5):
        warm_start = NMF(n_components=rank, init="nndsvda", random_state=42, tol=.005)
        representation = warm_start.fit_transform(representation)
        basis = warm_start.components_.T
        scale = np.maximum(np.linalg.norm(basis, axis=0), 1e-12)
        basis /= scale
        representation *= scale
        initial.append(torch.tensor(basis, dtype=torch.double))
    results = {}
    for mode, fraction in (("unsupervised", 0), ("semi_supervised", .5), ("supervised", 1)):
        torch.manual_seed(42)
        model = Neural_NMF([len(terms), 10, 5], c=5 if fraction else None)
        with torch.no_grad():
            for layer, basis in zip(model.lsqnonneglst, initial):
                layer.A.copy_(basis)
        loss = Energy_Loss_Func(lambd=.1)
        if fraction:
            indices = np.random.default_rng(42).permutation(len(train_y))[:int(len(train_y) * fraction)]
            mask = torch.zeros((5, len(train_y)), dtype=torch.double)
            mask[:, indices] = 1
            with torch.no_grad():
                factors, _ = model(X)
                targets = torch.nn.functional.one_hot(label[indices], num_classes=5).double()
                head = torch.linalg.lstsq(factors[-1][:, indices].T, targets, driver="gelsd").solution
                model.linear.weight.copy_(head.T)
        # Use one joint update per epoch. The package's legacy trainer repeats the
        # expensive NNLS forward pass 30 times per epoch to update the label head.
        optimizer = torch.optim.Adam(model.parameters(), lr=.02)
        for _ in range(epochs):
            optimizer.zero_grad()
            output = model(X)
            objective = (loss(model, X, *output, label, mask)
                         if fraction else loss(model, X, output))
            objective.backward()
            optimizer.step()
            with torch.no_grad():
                for layer in model.lsqnonneglst:
                    layer.A.clamp_(min=0)
        with torch.no_grad():
            output = model(X_test)
            factors, predictions = output if fraction else (output, None)
            A, B = (layer.A for layer in model.lsqnonneglst)
            error = torch.linalg.vector_norm(X_test - A @ B @ factors[-1]) / torch.linalg.vector_norm(X_test)
            assert torch.isfinite(error) and torch.all(A >= 0) and torch.all(B >= 0)
            result = {
                "test_relative_reconstruction_error": float(error),
                "subtopics": terms[A.argsort(dim=0, descending=True)[:4].T.numpy()].tolist(),
                "supertopics_subtopic_indices": B.argsort(dim=0, descending=True)[:3].T.tolist(),
            }
            if fraction:
                result["test_accuracy"] = float(np.mean(predictions.argmax(dim=0).numpy() == test_y))
            results[mode] = result
    majority = np.bincount(train_y).argmax()
    return {"dataset": dataset, "epochs": epochs, "train_rows": len(train_y), "test_rows": len(test_y),
            "test_majority_accuracy": float(np.mean(test_y == majority)), "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("olist", "newsgroups"), default="olist")
    parser.add_argument("--news-dir", type=Path, help="Directory containing the extracted 20 Newsgroups splits.")
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    if args.sample < 100 or args.epochs < 1:
        parser.error("Require at least 100 rows and one epoch.")
    torch.set_num_threads(1)
    report = run(args.dataset, args.sample, args.epochs, args.news_dir)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    path = out / f"neural_nmf_{args.dataset}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path)
    for name, values in report["results"].items():
        print(name, values["test_relative_reconstruction_error"], values.get("test_accuracy"))
