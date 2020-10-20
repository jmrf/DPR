import logging
import time
from typing import List
from typing import Tuple

import numpy as np
import torch
from torch import Tensor as T
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from typing import List
from typing import Tuple
from typing import Text

from dpr import setup_logger
from dpr.models import init_biencoder_components
from dpr.options import set_encoder_params_from_state
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import (
    move_to_device,
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


# TODO: Make explicit all parameters and stop passing around 'args'
# TODO:


class SingleEncoder:
    @classmethod
    def from_biencoder_ckpt(cls, biencoder_ckpt: Text, args) -> None:

        logger.info(f"Loading encoder '{cls.__name__}'")

        saved_state = load_states_from_checkpoint(biencoder_ckpt)
        set_encoder_params_from_state(saved_state.encoder_params, args)

        logger.info(f"Pretrained model tensorizer: {args.pretrained_model_cfg}")

        tensorizer, encoder, _ = init_biencoder_components(
            args.encoder_model_type, args, inference_only=True
        )

        encoder = getattr(encoder, cls.MODEL_TYPE)

        # load weights from the model file
        logger.info("Loading saved model state ...")
        model_to_load = get_model_obj(encoder)

        prefix_len = len(cls.MODEL_PREFIX)
        encoder_state = {
            key[prefix_len:]: value
            for (key, value) in saved_state.model_dict.items()
            if key.startswith(cls.MODEL_PREFIX)
        }
        model_to_load.load_state_dict(encoder_state)

        vector_size = model_to_load.get_out_size()
        logger.info("Encoder vector_size=%d", vector_size)

        return cls(encoder, tensorizer)


class QuestionEncoder(SingleEncoder):

    MODEL_TYPE = "question_model"
    MODEL_PREFIX = "question_model."

    def __init__(self, question_encoder: nn.Module, tensorizer: Tensorizer) -> None:
        self.question_encoder = question_encoder
        self.tensorizer = tensorizer

    def encode(
        self, questions: List[str], batch_size: int = 32, device: Text = "cuda"
    ) -> T:
        n = len(questions)
        query_vectors = []

        self.question_encoder.eval()

        with torch.no_grad():
            for _, batch_start in enumerate(range(0, n, batch_size)):

                batch_token_tensors = [
                    self.tensorizer.text_to_tensor(q)
                    for q in questions[batch_start : batch_start + batch_size]
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


class ContextEncoder(SingleEncoder):

    MODEL_TYPE = "ctx_model"
    MODEL_PREFIX = "ctx_model."

    def __init__(self, ctx_encoder: nn.Module, tensorizer: Tensorizer,) -> None:
        self.ctx_encoder = ctx_encoder
        self.tensorizer = tensorizer

    def gen_ctx_vectors(
        self,
        ctx_rows: List[Tuple[object, str, str]],
        insert_title: bool = True,
        batch_size: int = 32,
        device: Text = "cuda",
    ) -> List[Tuple[object, np.array]]:
        n = len(ctx_rows)
        total = 0
        results = []
        for _, batch_start in enumerate(range(0, n, batch_size)):

            batch_token_tensors = [
                self.tensorizer.text_to_tensor(
                    ctx[1], title=ctx[2] if insert_title else None
                )
                for ctx in ctx_rows[batch_start : batch_start + batch_size]
            ]

            ctx_ids_batch = move_to_device(
                torch.stack(batch_token_tensors, dim=0), device
            )
            ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch), device)
            ctx_attn_mask = move_to_device(
                self.tensorizer.get_attn_mask(ctx_ids_batch), device
            )
            with torch.no_grad():
                _, out, _ = self.ctx_encoder(
                    ctx_ids_batch, ctx_seg_batch, ctx_attn_mask
                )
            out = out.cpu()

            ctx_ids = [r[0] for r in ctx_rows[batch_start : batch_start + batch_size]]

            assert len(ctx_ids) == out.size(0)

            total += len(ctx_ids)

            results.extend(
                [(ctx_ids[i], out[i].view(-1).numpy()) for i in range(out.size(0))]
            )

            if total % 10 == 0:
                logger.info("Encoded passages %d", total)

        return results


class Retriever(object):
    """
    Does passage retrieving over the provided index
    """

    def __init__(
        self, index: DenseIndexer,
    ):
        self.index = index

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


def log_graph_to_tensorboard(model, tensorizer, text_input):

    t = tensorizer.text_to_tensor(text_input)

    q_ids_batch = torch.stack([t], dim=0).to("cpu")
    q_seg_batch = torch.zeros_like(q_ids_batch).to("cpu")
    q_attn_mask = model.tensorizer.get_attn_mask(q_ids_batch)

    # Log the graph to a Tensorboard compatible representation
    writer = SummaryWriter()
    writer.add_graph(
        model,
        input_to_model=[q_ids_batch, q_seg_batch, q_attn_mask],
        verbose=False,
    )
    writer.close()


if __name__ == "__main__":

    args = get_dense_retriever_args()

    console = setup_logger(
        logger, log_level=logging.DEBUG if args.debug else logging.INFO
    )

    # ******** Question Encoder********
    question_encoder = QuestionEncoder.from_biencoder_ckpt(args.model_file, args)

    # Encode test questions
    questions = [
        "Are all research code-bases such a mess every single time?",
        "Why is this not working so easily?",
    ]

    questions_tensor = question_encoder.encode(questions, device="cpu")
    logger.debug(f"question tensor shape: {questions_tensor.shape}")

    # ******** Context Encoder********
    ctx_encoder = ContextEncoder.from_biencoder_ckpt(args.model_file, args)

    # Encode test context passages
    contexts = [
        (
            0,
            "Usually research code bases are slightly messy in exchange of quick prototyping",
            "Research coding",
        ),
        (1, "To make AI research work good hardware is often required", "Requirements"),
    ]

    ctx_vectors = ctx_encoder.gen_ctx_vectors(contexts, device="cpu")
    logger.debug(
        f"Encoded {len(ctx_vectors)} context vectors "
        f"of shape: {ctx_vectors[0][1].shape}"
    )
