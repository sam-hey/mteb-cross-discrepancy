from mteb import MTEB
import mteb
from sentence_transformers import CrossEncoder
import argparse
import logging
from pathlib import Path
import torch.nn as nn


def main():
    # using other mteb version
    model_name = "cross-encoder/msmarco-MiniLM-L6-en-de-v1"
    output_foler = Path("output")
    output_foler.mkdir(exist_ok=True)
    tasks_names = ["GermanDPR"]
    eval_splits = ["test"]

    # not working for cross-encoder/msmarco-MiniLM-L6-en-de-v1 because of max_length = 512
    max_length = None
    if model_name == "cross-encoder/msmarco-MiniLM-L6-en-de-v1":
        max_length = 512
    cross_encoder = CrossEncoder(model_name, max_length=max_length)
    cross_encoder.model = nn.DataParallel(cross_encoder.model)
    tasks = mteb.get_tasks(tasks=tasks_names)
    print(tasks)
    evaluation = MTEB(
        tasks=tasks,
    )
    bm25 = mteb.get_model("bm25s")
    path_output = Path(output_foler) / "bm25"
    path_output.mkdir(parents=True, exist_ok=True)
    evaluation.run(
        bm25,
        eval_splits=eval_splits,
        output_folder=path_output.as_posix(),
        save_predictions=True,
    )

    evaluation.run(
        cross_encoder,
        eval_splits=eval_splits,
        output_folder=output_foler.as_posix(),
        save_predictions=True,
        top_k=100,
        previous_results=(
            path_output / (tasks_names[0] + "_default_predictions.json")
        ).as_posix(),
    )
    print("Evaluation finished")


if __name__ == "__main__":
    main()
    # https://github.com/embeddings-benchmark/mteb?tab=readme-ov-file#using-a-cross-encoder-for-reranking
