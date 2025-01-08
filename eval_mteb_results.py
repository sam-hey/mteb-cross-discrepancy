import json
from sentence_transformers import CrossEncoder
import numpy as np
import torch.nn as nn

# Load data from files
def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Save data to a JSON file
def save_json_file(data, file_path):
    def convert_float32(obj):
        if isinstance(obj, np.float32):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, default=convert_float32)


# Main processing function
def process_files(
    corpus_file, queries_file, relevant_docs_file, results_file, output_file
):
    # Load input data
    corpus = load_json_file(corpus_file)
    queries = load_json_file(queries_file)
    relevant_docs = load_json_file(relevant_docs_file)
    results = load_json_file(results_file)

    missing_top = 5

    # Initialize cross-encoder model
    model = CrossEncoder("cross-encoder/msmarco-MiniLM-L6-en-de-v1", max_length=512)
    model.model = nn.DataParallel(model.model)
    processed_results = {}
    output_results = {}

    for query_id, relevant in relevant_docs.items():
        query_text = queries.get(query_id, "")
        if not query_text:
            print(f"Query ID {query_id} not found in queries.json.")
            continue

        # Prepare pairs for scoring
        pairs = [(query_text, corpus[doc_id]) for doc_id in corpus]
        scores = model.predict(pairs, batch_size=600, show_progress_bar=True)

        # Collect results with scores
        scored_results = {doc_id: score for doc_id, score in zip(corpus, scores)}

        # Sort results by score in descending order
        sorted_results = dict(
            sorted(scored_results.items(), key=lambda item: item[1], reverse=True)
        )

        # Mark missing relevant documents in the top missing_top= 5
        top_10_docs = list(sorted_results.keys())[:missing_top]
        missing_relevant_docs = [
            doc_id for doc_id in relevant if doc_id not in top_10_docs
        ]

        # Compare with results from results file
        comparison_results = {}
        for doc_id in sorted_results:
            cross_encoder_score = sorted_results[doc_id]
            original_score = results.get(query_id, {}).get(doc_id, None)

            query_doc_list = results.get(query_id, {})
            query_doc_list_sorted = sorted(
                query_doc_list.items(), key=lambda item: item[1], reverse=True
            )

            comparison_results[doc_id] = {
                "cross_encoder_score": cross_encoder_score,
                "original_score": original_score,
                "cross_encoder_index": list(sorted_results.keys()).index(doc_id),
                "original_index": [doc[0] for doc in query_doc_list_sorted].index(
                    doc_id
                )
                if original_score is not None
                else None,
                "mismatch": False,
            }

        # Check for mismatched order
        original_sorted_docs = sorted(
            results.get(query_id, {}).items(), key=lambda item: item[1], reverse=True
        )
        cross_encoder_sorted_docs = list(sorted_results.keys())

        for i, (doc_id, _) in enumerate(original_sorted_docs):
            if (
                i < len(cross_encoder_sorted_docs)
                and doc_id != cross_encoder_sorted_docs[i]
            ):
                comparison_results[doc_id]["mismatch"] = True

        # Add data to output
        processed_results[query_id] = {
            "comparison_results": comparison_results,
            "missing_relevant_docs_in_top_x": missing_relevant_docs,
        }

        cross_encoder_results = {
            query_id: {
                doc_id: scores["cross_encoder_score"]
                for doc_id, scores in comparison_results.items()
            }
        }
        processed_results[query_id]["cross_encoder_results"] = cross_encoder_results[
            query_id
        ]

        output_results[query_id] = {
            doc_id: scores["cross_encoder_score"]
            for doc_id, scores in comparison_results.items()
        }

    # Output the processed results
    for query_id, result in processed_results.items():
        print(f"Query ID: {query_id}")
        print("Comparison Results:")
        for doc_id, scores in result["comparison_results"].items():
            if scores["mismatch"]:
                print(
                    f"  \033[91mDoc ID: {doc_id}, Cross-Encoder Score: {scores['cross_encoder_score']} (Index: {scores['cross_encoder_index']}), MTEB Score: {scores['original_score']} (Index: {scores['original_index']})\033[0m"
                )
            else:
                print(
                    f"  Doc ID: {doc_id}, Cross-Encoder Score: {scores['cross_encoder_score']} (Index: {scores['cross_encoder_index']}), MTEB Score: {scores['original_score']} (Index: {scores['original_index']})"
                )
        print(f"Missing Relevant Docs in Top {missing_top}:")
        for doc_id in result["missing_relevant_docs_in_top_x"]:
            print(f"  Doc ID: {doc_id}")
        print()
    # Save the processed results
    save_json_file(output_results, output_file)
    save_json_file(processed_results, output_file.replace(".json", "_processed.json"))
    print(f"Processed results saved to {output_file}")


# File paths
corpus_file = "output/fmplus/corpus.json"
queries_file = "output/fmplus/queries.json"
relevant_docs_file = "output/fmplus/relevant_docs.json"
results_file = "output/GermanDPR_default_predictions.json"
output_file = "output/GermanDPR_default_predictions_results.json"

# Run the script
process_files(corpus_file, queries_file, relevant_docs_file, results_file, output_file)
