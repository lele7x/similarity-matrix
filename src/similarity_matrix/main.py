
import os
import logging
import argparse
from dotenv import load_dotenv

from similarity_matrix.lib.database import Database
from similarity_matrix.lib.pipeline import Pipeline
from similarity_matrix.lib.logging import logger


# -----------------------------------------------------------------------
# Main sub-commands

def update_db_row_table(pipeline: Pipeline):
    logger.debug(f'Calling update_db_row_table with {pipeline}')
    return pipeline.update_db_row_table()


def update_db_column_table(pipeline: Pipeline):
    logger.debug(f'Calling update_db_column_table with {pipeline}')
    return pipeline.update_db_column_table()


def update_db_matrix(pipeline: Pipeline):
    logger.debug(f'Calling update_db_matrix with {pipeline}')
    return pipeline.update_db_matrix_table()


def compute_similarity(pipeline: Pipeline):
    logger.debug(f'Calling compute_similarity with {pipeline}')
    return pipeline.get_matrix()


# -----------------------------------------------------------------------
# Main

def main():
    # ---------------------------------------
    # Common arguments parser
    parser = argparse.ArgumentParser(
        description='ute similarity between two sets of texts')
    parser.add_argument(
        '--version',
        action='version',
        version='0.2.0')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='show debug messages')
    parser.add_argument(
        '--db-host',
        metavar='DB_HOST',
        default=os.environ.get('DB_HOST', 'localhost'),
        help='database host (default: localhost)')
    parser.add_argument(
        '--db-port',
        metavar='DB_PORT',
        default=os.environ.get('DB_PORT', '3306'),
        help='database port (default: 3306)')
    parser.add_argument(
        '--db-name',
        metavar='DB_DATABASE',
        default=os.environ.get('DB_DATABASE'),
        help='database name')
    parser.add_argument(
        '--db-user',
        metavar='DB_USERNAME',
        default=os.environ.get('DB_USERNAME'),
        help='database user (can be passed through .env DB_USERNAME)')
    parser.add_argument(
        '--db-password',
        metavar='DB_PASSWORD',
        default=os.environ.get('DB_PASSWORD'),
        help='database password (can be passed through .env DB_PASSWORD)')
    parser.add_argument(
        '--model-name',
        metavar='MODEL_NAME',
        default=os.environ.get('MODEL_NAME', 'jinaai/jina-embeddings-v3'),
        help='the name of the model to use for the similarity calculation (default: jinaai/jina-embeddings-v3, can be passed through .env MODEL_NAME)')
    parser.add_argument(
        '--pipeline-dir',
        default='./pipelines',
        help='path to the directory containing all pipelines (default: ./pipelines)')
    parser.add_argument(
        '--list-pipelines',
        action='store_true',
        help='list all the available pipelines')
    parser.add_argument(
        '--output-dir',
        default='matrices',
        help='path to the directory where the similarity matrices will be saved (default: matrices)')
    parser.add_argument(
        '--pipeline',
        default=None,
        help='the name of the pipeline to use (name of the .py file with no extension)')
    subparsers = parser.add_subparsers(
        dest='commands',
        help='which action to perform on the selected pipeline')

    # ---------------------------------------
    # Argument parser for the update_db_row_table sub-command
    upd_skill_parser = subparsers.add_parser(
        'update-db-row',
        help='update the row table')
    upd_skill_parser.set_defaults(func=update_db_row_table)

    # ---------------------------------------
    # Argument parser for the update_db_column_table sub-command
    upd_skill_parser = subparsers.add_parser(
        'update-db-column',
        help='update the column table')
    upd_skill_parser.set_defaults(func=update_db_column_table)

    # ---------------------------------------
    # Argument parser for the update_db_matrix sub-command
    upd_skill_parser = subparsers.add_parser(
        'update-db-matrix',
        help='update the column table')
    upd_skill_parser.set_defaults(func=update_db_matrix)

    # ---------------------------------------
    # Argument parser for the compute_similarity sub-command
    upd_skill_parser = subparsers.add_parser(
        'compute-similarity',
        help='update the column table')
    upd_skill_parser.set_defaults(func=compute_similarity)

    args = parser.parse_args()

    # ---------------------------------------
    # Load ENV
    load_dotenv()

    # ---------------------------------------
    # Debug
    if args.debug:
        logger.info('Debug mode')
        logger.setLevel(logging.DEBUG)
        for h in logger.handlers:
            h.setLevel(logging.DEBUG)

    # ---------------------------------------
    # Database
    try:
        args.db_port = int(args.db_port)
    except ValueError:
        logger.error(f'Invalid port number {args.db_port}')
        exit(1)

    # Configure database only if a username was provided
    if args.db_user is not None:
        db = Database(
            args.db_host,
            args.db_port,
            args.db_user,
            args.db_password,
            args.db_name)

        e = db.test_connection()
        if e is not None:
            logger.error('Error connecting to the database')
            raise e
    else:
        db = None
        logger.info(
            "Database username not set: ignoring database configuration")

    # ---------------------------------------
    # Pipeline

    # Load all the pipelines
    available_pipelines = Pipeline.load_default_pipelines()
    if os.path.isdir(args.pipeline_dir):
        available_pipelines.update(Pipeline.load_from_dir(args.pipeline_dir))

    logger.debug(f'Available pipelines: {available_pipelines}')

    # If was passed the argument --list-pipelines or the pipeline
    # was not passed, print the available pipelines
    if args.list_pipelines:
        logger.info('Available pipelines: ' +
                    ', '.join(available_pipelines.keys()))
        exit(0)
    elif args.pipeline is None:
        logger.error(
            'No pipeline was passed (--pipeline)! Available pipelines: ' +
            ', '.join(
                available_pipelines.keys()))
        exit(1)

    # Check if the pipeline exists
    if args.pipeline not in available_pipelines:
        logger.error(f'Pipeline {args.pipeline} not found')
        exit(1)

    # Initialize pipeline
    pipeline = available_pipelines[args.pipeline](
        name=args.pipeline,
        db=db,
        path=args.output_dir,
        model_name=args.model_name
    )

    # ---------------------------------------
    # Call the correct function
    if not hasattr(args, 'func'):
        parser.print_help()
        exit(1)

    args.func(pipeline=pipeline)

    logger.info('Done! ;)')
    exit(0)


# -----------------------------------------------------------------------
# Entrypoint

if __name__ == "__main__":
    main()
