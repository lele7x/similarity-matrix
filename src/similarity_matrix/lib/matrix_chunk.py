import numpy as np
import tempfile
import math
import os
from typing import List, Optional, Union
from pathlib import Path

from similarity_matrix.lib.logging import logger
from similarity_matrix.lib.utils import total_length_lists
from similarity_matrix.lib.model import initialize_model
from similarity_matrix.lib.windowing import compute_aggregation_tail_weight
from similarity_matrix.lib.similarity import cos_sim_mem
from similarity_matrix.lib.matrix import SimilarityMatrix


# -----------------------------------------------------------------------
# Chuncked Matrix class

class ChunkedSimilarityMatrix(SimilarityMatrix):
    """
    A memory-efficient subclass of SimilarityMatrix that processes the matrix
    in chunks to avoid filling RAM with large matrices.
    """

    def __init__(self,
                 row_ids: List[str],
                 column_ids: List[str],
                 name: str,
                 row_load_function: callable = None,
                 column_load_function: callable = None,
                 matrix: Optional[np.ndarray] = None,
                 model_name: str = os.environ.get('MODEL_NAME', 'jinaai/jina-embeddings-v3'),
                 row_chunk_size: int = 100,
                 column_chunk_size: int = 100,
                 temp_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the ChunkedSimilarityMatrix.

        Args:
            row_ids: List of row IDs
            column_ids: List of column IDs
            name: Name of the matrix
            row_load_function: Function to load row data
            column_load_function: Function to load column data
            matrix: Optional pre-computed matrix
            model_name: The name of the SentenceTransformer model to initialize
            row_chunk_size: Number of rows to process at once
            column_chunk_size: Number of columns to process at once
            temp_dir: Directory for temporary files (uses system temp if None)
        """
        super().__init__(row_ids, column_ids, name, row_load_function,
                         column_load_function, matrix, model_name)

        self.row_chunk_size = row_chunk_size
        self.column_chunk_size = column_chunk_size
        self.temp_dir = Path(temp_dir) if temp_dir else None

    def calculate(self, fake: int | None = None) -> None:
        """
        Calculate the similarity matrix in chunks to avoid memory issues.

        Args:
            fake: If provided, generates random values instead of actual embeddings
        """
        if fake is not None:
            # Use parent's fake implementation for testing
            super().calculate(fake)
            return

        # Initialize the embedding model
        model = initialize_model(model_name=self.model_name)

        # Load row and column data using the passed functions
        row_texts = self.row_load_function()
        column_texts = self.column_load_function()

        # Get rid of eventual pointers returned (actually stored internally)
        # and store them for later use
        row_texts, column_texts = self._detect_pointers(
            row_texts, column_texts)
        logger.info(
            f"Matrix shape (windows): {len(row_texts)} rows, "
            f"{len(column_texts)} columns")

        # Calculate number of chunks based on number of pointers
        if self._row_pointers is not None:
            self.num_row_chunks = math.ceil(
                len(self._row_pointers) / self.row_chunk_size)
        else:
            self.num_row_chunks = math.ceil(
                self.matrix.shape[0] / self.row_chunk_size)

        if self._column_pointers is not None:
            self.num_col_chunks = math.ceil(
                len(self._column_pointers) / self.column_chunk_size)
        else:
            self.num_col_chunks = math.ceil(
                self.matrix.shape[1] / self.column_chunk_size)

        # Display matrix shapes for debugging
        logger.info(
            f"Matrix shape (final): {len(self.row_ids)} rows, "
            f"{len(self.column_ids)} columns")
        logger.info(
            "Processing in chunks of "
            f"{self.row_chunk_size}x{self.column_chunk_size}")
        logger.info(
            f"Total chunks: {self.num_row_chunks} rows x " +
            f"{self.num_col_chunks} columns = " +
            f"{self.num_row_chunks * self.num_col_chunks} total chunks")

        # Create temporary directory for chunk storage
        with tempfile.TemporaryDirectory(dir=self.temp_dir) as temp_dir:
            temp_path = Path(temp_dir)

            # Process matrix in chunks (includes aggregation per chunk)
            self._process_chunks(model, row_texts, column_texts, temp_path)

            # Reassemble the aggregated matrix from chunks
            # At this point chunks are already aggregated to final size
            self._reassemble_matrix(temp_path)

        # Ensure values are in valid range [-1, 1]
        self.matrix = np.clip(self.matrix, -1, 1)

    def _process_chunks(self, model, row_texts: List[str],
                        column_texts: List[str], temp_path: Path) -> None:
        """
        Process the similarity matrix in chunks, apply aggregation per chunk, and save to temporary files.

        Args:
            model: The embedding model
            row_texts: List of row texts (windows, not final texts)
            column_texts: List of column texts (windows, not final texts)
            temp_path: Path to temporary directory
        """
        # Calculate chunk mappings for the windowed texts
        total_chunks = self.num_row_chunks * self.num_col_chunks
        processed_chunks = 0

        for row_chunk_idx in range(self.num_row_chunks):
            # Calculate row chunk indices
            if self._row_pointers is not None:
                # Since chunks are calculated on the pointers, we need to use them
                # to determine the start and end of the row chunk in windows
                # (texts)
                row_start = total_length_lists(
                    self._row_pointers[0:(row_chunk_idx * self.row_chunk_size)])
                row_size = total_length_lists(self._row_pointers[row_chunk_idx * self.row_chunk_size:
                                                                 (row_chunk_idx + 1) * self.row_chunk_size])
                row_end = min(row_start + row_size, len(row_texts))
            else:
                row_start = row_chunk_idx * self.row_chunk_size
                row_end = min(row_start + self.row_chunk_size, len(row_texts))

            # Extract the row chunk texts
            row_chunk = row_texts[row_start:row_end]

            for col_chunk_idx in range(self.num_col_chunks):
                # Calculate column chunk indices
                if self._column_pointers is not None:
                    # Since chunks are calculated on the pointers, we need to use them
                    # to determine the start and end of the column chunk in
                    # windows (texts)
                    col_start = total_length_lists(
                        self._column_pointers[0:(col_chunk_idx * self.column_chunk_size)])
                    col_size = total_length_lists(self._column_pointers[col_chunk_idx * self.column_chunk_size:
                                                                        (col_chunk_idx + 1) * self.column_chunk_size])
                    col_end = min(col_start + col_size, len(column_texts))
                else:
                    col_start = col_chunk_idx * self.column_chunk_size
                    col_end = min(
                        col_start + self.column_chunk_size,
                        len(column_texts))

                # Extract the column chunk texts
                col_chunk = column_texts[col_start:col_end]

                # Compute similarity for this chunk
                logger.debug(f"Processing chunk ({row_chunk_idx}, {col_chunk_idx}): "
                             f"rows {row_start}:{row_end}, cols {col_start}:{col_end}")

                chunk_matrix = cos_sim_mem(model, row_chunk, col_chunk)

                # Apply aggregation to this chunk if pointers exist
                if self._row_pointers is not None or self._column_pointers is not None:
                    # Extract relevant pointers for this chunk in pointers (not
                    # windows!)
                    if self._row_pointers:
                        row_start_pointers = row_chunk_idx * self.row_chunk_size
                        row_end_pointers = min(
                            row_start_pointers + self.row_chunk_size, len(self._row_pointers))
                        row_pointers_chunk = self._row_pointers[row_start_pointers:row_end_pointers]
                    else:
                        row_pointers_chunk = None

                    if self._column_pointers:
                        col_start_pointers = col_chunk_idx * self.column_chunk_size
                        col_end_pointers = min(
                            col_start_pointers + self.column_chunk_size, len(self._column_pointers))
                        col_pointers_chunk = self._column_pointers[col_start_pointers:col_end_pointers]
                    else:
                        col_pointers_chunk = None

                    # Apply aggregation with tail weight
                    chunk_matrix = compute_aggregation_tail_weight(
                        chunk_matrix, row_pointers_chunk, col_pointers_chunk, tail_weight=0.2)

                # Save aggregated chunk to temporary file
                chunk_fname = f"{self.name}_chunk_" +\
                    f"{row_chunk_idx}_{col_chunk_idx}.npy"
                chunk_path = temp_path / chunk_fname
                np.save(chunk_path, chunk_matrix)

                # Clear chunk from memory
                del chunk_matrix

                processed_chunks += 1
                if processed_chunks % 10 == 0 or processed_chunks == total_chunks:
                    logger.info(f"Processed {processed_chunks}/{total_chunks} chunks " +
                                f"({100 * processed_chunks / total_chunks:.1f}%)")

    def _reassemble_matrix(self, temp_path: Path) -> None:
        """
        Reassemble the aggregated matrix from saved chunks.

        Args:
            temp_path: Path to temporary directory containing chunks
        """
        logger.info("Reassembling aggregated matrix from chunks")

        # Initialize the final matrix (already aggregated size)
        self.matrix = np.zeros(
            (len(
                self.row_ids), len(
                self.column_ids)), dtype=float)

        current_final_row = 0

        for row_chunk_idx in range(self.num_row_chunks):
            current_final_col = 0
            chunk_row_height = 0

            for col_chunk_idx in range(self.num_col_chunks):
                # Load chunk from temporary file
                chunk_fname = f"{self.name}_chunk_{row_chunk_idx}_" + \
                    f"{col_chunk_idx}.npy"
                chunk_path = temp_path / chunk_fname
                chunk_matrix = np.load(chunk_path)

                # Get the size of this aggregated chunk
                chunk_rows, chunk_cols = chunk_matrix.shape

                # Place chunk in the final matrix
                self.matrix[current_final_row:current_final_row +
                            chunk_rows, current_final_col:current_final_col +
                            chunk_cols] = chunk_matrix

                current_final_col += chunk_cols

                # All chunks in a row should have same height
                chunk_row_height = chunk_rows

                # Clean up the temporary file
                os.unlink(chunk_path)

            # Move to next row position
            current_final_row += chunk_row_height

        logger.info("Matrix reassembly complete")

    def set_chunk_sizes(
            self,
            row_chunk_size: int,
            column_chunk_size: int) -> None:
        """
        Update the chunk sizes for processing.

        Args:
            row_chunk_size: Number of rows to process at once
            column_chunk_size: Number of columns to process at once
        """
        self.row_chunk_size = row_chunk_size
        self.column_chunk_size = column_chunk_size

    def estimate_memory_usage(self, embedding_dim: int = 768,
                              dtype_size: int = 4) -> dict:
        """
        Estimate memory usage for different components.

        Args:
            embedding_dim: Dimension of embeddings
            dtype_size: Size of float type in bytes (4 for float32, 8 for float64)

        Returns:
            Dictionary with memory estimates in MB
        """
        chunk_matrix_mb = (self.row_chunk_size *
                           self.column_chunk_size * dtype_size) / (1024**2)
        embeddings_mb = (max(self.row_chunk_size, self.column_chunk_size) *
                         embedding_dim * dtype_size) / (1024**2)
        full_matrix_mb = (len(self.row_ids) *
                          len(self.column_ids) * dtype_size) / (1024**2)

        return {
            'chunk_matrix_mb': chunk_matrix_mb,
            'embeddings_mb': embeddings_mb,
            'full_matrix_mb': full_matrix_mb,
            'peak_chunk_memory_mb': chunk_matrix_mb + embeddings_mb
        }
