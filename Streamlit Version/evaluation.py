from rouge_score import rouge_scorer

def compute_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )

    scores = scorer.score(reference, generated)

    return {
        "ROUGE-1": scores['rouge1'].fmeasure,
        "ROUGE-2": scores['rouge2'].fmeasure,
        "ROUGE-L": scores['rougeL'].fmeasure
    }