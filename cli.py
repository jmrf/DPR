import argparse

from dpr.options import add_cuda_params
from dpr.options import add_encoder_params
from dpr.options import add_tokenizer_params
from dpr.options import print_args
from dpr.options import setup_args_gpu


def get_dense_embedding_args():

    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument(
        "--ctx_file", type=str, default=None, help="Path to passages set .tsv file"
    )
    parser.add_argument(
        "--out_file",
        required=True,
        type=str,
        default=None,
        help="output file path to write results to",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="Number(0-based) of data shard to process",
    )
    parser.add_argument(
        "--num_shards", type=int, default=1, help="Total amount of data shards"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for the passage encoder forward pass",
    )
    parser.add_argument("--debug", action="store_true", help="set logging to DEBUG")

    args = parser.parse_args()

    assert (
        args.model_file
    ), "Please specify --model_file checkpoint to init model weights"

    setup_args_gpu(args)

    return args


def get_dense_retriever_args():
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument(
        "--qa_file",
        # required=True,
        type=str,
        default=None,
        help=(
            "Question and answers file of the format: "
            "question \\t ['answer1','answer2', ...]"
        ),
    )
    parser.add_argument(
        "--ctx_file",
        # required=True,
        type=str,
        default=None,
        help="All passages file in the tsv format: id \\t passage_text \\t title",
    )
    parser.add_argument(
        "--encoded_ctx_file",
        type=str,
        default=None,
        help="Glob path to encoded passages (from generate_dense_embeddings tool)",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default=None,
        help="output .json file path to write results to ",
    )
    parser.add_argument(
        "--match",
        type=str,
        default="string",
        choices=["regex", "string"],
        help="Answer matching logic type",
    )
    parser.add_argument(
        "--n-docs", type=int, default=200, help="Amount of top docs to return"
    )
    parser.add_argument(
        "--validation_workers",
        type=int,
        default=16,
        help="Number of parallel processes to validate results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for question encoder forward pass",
    )
    parser.add_argument(
        "--index_buffer",
        type=int,
        default=50000,
        help="Temporal memory data buffer size (in samples) for indexer",
    )
    parser.add_argument(
        "--hnsw_index",
        action="store_true",
        help="If enabled, use inference time efficient HNSW index",
    )
    parser.add_argument(
        "--save_or_load_index", action="store_true", help="If enabled, save index"
    )
    parser.add_argument("--debug", action="store_true", help="set logging to DEBUG")

    args = parser.parse_args()

    assert (
        args.model_file
    ), "Please specify --model_file checkpoint to init model weights"

    setup_args_gpu(args)
    print_args(args)

    return args
