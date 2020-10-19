#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""


import os
import csv
import glob
import json
import gzip
import logging
import pickle
import time
from typing import List, Tuple, Dict, Iterator

import numpy as np
import torch
from torch import Tensor as T
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from dpr import setup_logger
from dpr.data.qa_validation import calculate_matches
from dpr.models import init_biencoder_components
from dpr.options import set_encoder_params_from_state
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
)
from dpr.indexer.faiss_indexers import (
    DenseIndexer,
    DenseHNSWFlatIndexer,
    DenseFlatIndexer,
)

from cli import get_dense_retriever_args

logger = logging.getLogger(__name__)
console = None


class DenseRetriever(object):
    """
    Does passage retrieving over the provided index and question encoder
    """

    def __init__(
        self,
        question_encoder: nn.Module,
        batch_size: int,
        tensorizer: Tensorizer,
        index: DenseIndexer,
    ):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.index = index

    def generate_question_vectors(self, questions: List[str], device="cuda") -> T:
        n = len(questions)
        bsz = self.batch_size
        query_vectors = []

        self.question_encoder.eval()

        with torch.no_grad():
            for j, batch_start in enumerate(range(0, n, bsz)):

                batch_token_tensors = [
                    self.tensorizer.text_to_tensor(q)
                    for q in questions[batch_start : batch_start + bsz]
                ]

                q_ids_batch = torch.stack(batch_token_tensors, dim=0).to(device)
                q_seg_batch = torch.zeros_like(q_ids_batch).to(device)
                q_attn_mask = self.tensorizer.get_attn_mask(q_ids_batch)
                _, out, _ = self.question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

                query_vectors.extend(out.cpu().split(1, dim=0))

                if len(query_vectors) % 100 == 0:
                    logger.info("Encoded queries %d", len(query_vectors))

        query_tensor = torch.cat(query_vectors, dim=0)

        logger.info("Total encoded queries tensor %s", query_tensor.size())

        assert query_tensor.size(0) == len(questions)
        return query_tensor

    def get_top_docs(
        self, query_vectors: np.array, top_docs: int = 100
    ) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        results = self.index.search_knn(query_vectors, top_docs)
        logger.info("index search time: %f sec.", time.time() - time0)
        return results


def parse_qa_csv_file(location) -> Iterator[Tuple[str, List[str]]]:
    with open(location) as ifile:
        reader = csv.reader(ifile, delimiter="\t")
        for row in reader:
            question = row[0]
            answers = eval(row[1])
            yield question, answers


def validate(
    passages: Dict[object, Tuple[str, str]],
    answers: List[List[str]],
    result_ctx_ids: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
) -> List[List[bool]]:
    match_stats = calculate_matches(
        passages, answers, result_ctx_ids, workers_num, match_type
    )
    top_k_hits = match_stats.top_k_hits

    logger.info("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    logger.info("Validation results: top k documents hits accuracy %s", top_k_hits)
    return match_stats.questions_doc_hits


def load_passages(ctx_file: str) -> Dict[object, Tuple[str, str]]:
    docs = {}
    logger.info("Reading data from: %s", ctx_file)
    if ctx_file.endswith(".gz"):
        with gzip.open(ctx_file, "rt") as tsvfile:
            reader = csv.reader(tsvfile, delimiter="\t",)
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != "id":
                    docs[row[0]] = (row[1], row[2])
    else:
        with open(ctx_file) as tsvfile:
            reader = csv.reader(tsvfile, delimiter="\t",)
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != "id":
                    docs[row[0]] = (row[1], row[2])
    return docs


def save_results(
    passages: Dict[object, Tuple[str, str]],
    questions: List[str],
    answers: List[List[str]],
    top_passages_and_scores: List[Tuple[List[object], List[float]]],
    per_question_hits: List[List[bool]],
    out_file: str,
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)

        merged_data.append(
            {
                "question": q,
                "answers": q_answers,
                "ctxs": [
                    {
                        "id": results_and_scores[0][c],
                        "title": docs[c][1],
                        "text": docs[c][0],
                        "score": scores[c],
                        "has_answer": hits[c],
                    }
                    for c in range(ctxs_num)
                ],
            }
        )

    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info("Saved results * scores  to %s", out_file)


def iterate_encoded_files(vector_files: list) -> Iterator[Tuple[object, np.array]]:
    for i, file in enumerate(vector_files):
        logger.info("Reading file %s", file)
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                db_id, doc_vector = doc
                yield db_id, doc_vector


if __name__ == "__main__":

    args = get_dense_retriever_args()

    console = setup_logger(
        logger, log_level=logging.DEBUG if args.debug else logging.INFO
    )

    saved_state = load_states_from_checkpoint(args.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, args)

    tensorizer, encoder, _ = init_biencoder_components(
        args.encoder_model_type, args, inference_only=True
    )

    encoder = encoder.question_model

    encoder, _ = setup_for_distributed_mode(
        encoder, None, args.device, args.n_gpu, args.local_rank, args.fp16
    )
    encoder.eval()

    # load weights from the model file
    logger.info("Loading saved model state ...")
    model_to_load = get_model_obj(encoder)

    prefix_len = len("question_model.")
    question_encoder_state = {
        key[prefix_len:]: value
        for (key, value) in saved_state.model_dict.items()
        if key.startswith("question_model.")
    }
    model_to_load.load_state_dict(question_encoder_state)

    vector_size = model_to_load.get_out_size()
    logger.info("Encoder vector_size=%d", vector_size)

    if args.hnsw_index:
        index = DenseHNSWFlatIndexer(vector_size, args.index_buffer)
    else:
        index = DenseFlatIndexer(vector_size, args.index_buffer)

    retriever = DenseRetriever(encoder, args.batch_size, tensorizer, index)

    # index all passages
    ctx_files_pattern = args.encoded_ctx_file
    input_paths = glob.glob(ctx_files_pattern)

    index_path = "_".join(input_paths[0].split("_")[:-1])
    if args.save_or_load_index and (
        os.path.exists(index_path) or os.path.exists(index_path + ".index.dpr")
    ):
        retriever.index.deserialize_from(index_path)
    else:
        logger.info("Reading all passages data from files: %s", input_paths)
        retriever.index.index_data(input_paths)
        if args.save_or_load_index:
            retriever.index.serialize(index_path)

    # get questions & answers
    questions = []
    question_answers = []

    for ds_item in parse_qa_csv_file(args.qa_file):
        question, answers = ds_item
        questions.append(question)
        question_answers.append(answers)

    questions_tensor = retriever.generate_question_vectors(questions)

    # get top k results
    top_ids_and_scores = retriever.get_top_docs(questions_tensor.numpy(), args.n_docs)

    all_passages = load_passages(args.ctx_file)

    if len(all_passages) == 0:
        raise RuntimeError(
            "No passages data found. Please specify ctx_file param properly."
        )

    questions_doc_hits = validate(
        all_passages,
        question_answers,
        top_ids_and_scores,
        args.validation_workers,
        args.match,
    )

    if args.out_file:
        save_results(
            all_passages,
            questions,
            question_answers,
            top_ids_and_scores,
            questions_doc_hits,
            args.out_file,
        )
