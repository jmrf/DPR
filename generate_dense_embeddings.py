#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool that produces embeddings for a large documents base based on
 the pretrained ctx & question encoders.
 Supposed to be used in a 'sharded' way to speed up the process.
"""
import os
import pathlib

import argparse
import csv
import logging
import pickle
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from dpr.models import init_biencoder_components
from dpr.options import (
    print_args,
    set_encoder_params_from_state,
)
from dpr import setup_logger
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
    move_to_device,
)
from cli import get_dense_embedding_args

logger = logging.getLogger(__name__)
console = None


def gen_ctx_vectors(
    ctx_rows: List[Tuple[object, str, str]],
    model: nn.Module,
    tensorizer: Tensorizer,
    insert_title: bool = True,
) -> List[Tuple[object, np.array]]:
    n = len(ctx_rows)
    bsz = args.batch_size
    total = 0
    results = []
    for j, batch_start in enumerate(range(0, n, bsz)):

        batch_token_tensors = [
            tensorizer.text_to_tensor(ctx[1], title=ctx[2] if insert_title else None)
            for ctx in ctx_rows[batch_start : batch_start + bsz]
        ]

        ctx_ids_batch = move_to_device(
            torch.stack(batch_token_tensors, dim=0), args.device
        )
        ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch), args.device)
        ctx_attn_mask = move_to_device(
            tensorizer.get_attn_mask(ctx_ids_batch), args.device
        )
        with torch.no_grad():
            _, out, _ = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)
        out = out.cpu()

        ctx_ids = [r[0] for r in ctx_rows[batch_start : batch_start + bsz]]

        assert len(ctx_ids) == out.size(0)

        total += len(ctx_ids)

        results.extend(
            [(ctx_ids[i], out[i].view(-1).numpy()) for i in range(out.size(0))]
        )

        if total % 10 == 0:
            logger.info("Encoded passages %d", total)

    return results


if __name__ == "__main__":

    args = get_dense_embedding_args()

    console = setup_logger(
        logger, log_level=logging.DEBUG if args.debug else logging.INFO
    )

    saved_state = load_states_from_checkpoint(args.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, args)
    print_args(args)

    tensorizer, encoder, _ = init_biencoder_components(
        args.encoder_model_type, args, inference_only=True
    )

    encoder = encoder.ctx_model

    encoder, _ = setup_for_distributed_mode(
        encoder,
        None,
        args.device,
        args.n_gpu,
        args.local_rank,
        args.fp16,
        args.fp16_opt_level,
    )
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info("Loading saved model state ...")
    logger.debug("saved model keys =%s", saved_state.model_dict.keys())

    prefix_len = len("ctx_model.")
    ctx_state = {
        key[prefix_len:]: value
        for (key, value) in saved_state.model_dict.items()
        if key.startswith("ctx_model.")
    }
    model_to_load.load_state_dict(ctx_state)

    logger.info("reading data from file=%s", args.ctx_file)

    rows = []
    with open(args.ctx_file) as tsvfile:
        reader = csv.reader(tsvfile, delimiter="\t")
        # file format: doc_id, doc_text, title
        rows.extend([(row[0], row[1], row[2]) for row in reader if row[0] != "id"])

    shard_size = int(len(rows) / args.num_shards)
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size

    logger.info(
        "Producing encodings for passages range: %d to %d (out of total %d)",
        start_idx,
        end_idx,
        len(rows),
    )
    rows = rows[start_idx:end_idx]

    data = gen_ctx_vectors(rows, encoder, tensorizer, True)

    file = args.out_file + "_" + str(args.shard_id) + ".pkl"
    pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
    logger.info("Writing results to %s" % file)
    with open(file, mode="wb") as f:
        pickle.dump(data, f)

    logger.info("Total passages processed %d. Written to %s", len(data), file)
