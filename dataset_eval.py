from datasets import load_dataset
from utils import hybrid_summary, tfidf_summary, bart_summary, compute_rouge

dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:10]")

results = {
    "TF-IDF": [],
    "BART": [],
    "HYBRID": []
}

for sample in dataset:
    text = sample["article"]
    reference = sample["highlights"]

    tfidf = tfidf_summary(text)
    bart = bart_summary(text)
    hybrid = hybrid_summary(text)

    results["TF-IDF"].append(compute_rouge(reference, tfidf))
    results["BART"].append(compute_rouge(reference, bart))
    results["HYBRID"].append(compute_rouge(reference, hybrid))


def average(scores, metric):
    return sum(s[metric] for s in scores) / len(scores)


print("\n===== AVERAGE ROUGE SCORES =====\n")

for model in results:
    print(model)
    for metric in ["rouge1", "rouge2", "rougeL"]:
        print(metric, ":", round(average(results[model], metric), 4))
    print()