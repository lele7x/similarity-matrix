import numpy as np
import json
import os
from typing import List, Optional, Union
from pathlib import Path

from similarity_matrix.lib.logging import logger
from similarity_matrix.lib.math import normalize_array
from similarity_matrix.lib.model import initialize_model
from similarity_matrix.lib.windowing import compute_aggregation_tail_weight
from similarity_matrix.lib.similarity import cos_sim_mem


# -----------------------------------------------------------------------
# Matrix class

class SimilarityMatrix:
    """
    A class to handle a complex data structure containing:
    - A numpy matrix of float numbers
    - A list of IDs corresponding to rows
    - A list of IDs corresponding to columns
    - A name for identification and file saving
    - A function to load a row given the list of row IDs
    - A function to load a column given the list of column IDs
    """

    def __init__(self,
                 row_ids: list,
                 column_ids: list,
                 name: str,
                 row_load_function: callable = None,
                 column_load_function: callable = None,
                 matrix: Optional[np.ndarray] = None,
                 model_name: str | None = None):
        """
        Initialize the SymilarityMatrix.

        Args:
            row_ids: List of row IDs (rows)
            column_ids: List of column IDs (columns)
            name: Name of the matrix (used for file saving)
            row_load_function: Function to load rows given the row IDs
            column_load_function: Function to load columns given the column IDs
            matrix: Optional pre-computed matrix. If None, creates empty matrix
            model_name: The name of the SentenceTransformer model to initialize.
        """
        self.row_ids = list(row_ids)
        self.column_ids = list(column_ids)
        self.name = name

        if row_load_function is None:
            self.row_load_function = lambda row_ids: [
                str(id) for id in row_ids]
        else:
            self.row_load_function = row_load_function

        if column_load_function is None:
            self.column_load_function = lambda column_ids: [
                str(id) for id in column_ids]
        else:
            self.column_load_function = column_load_function

        if matrix is not None:
            if matrix.shape != (len(row_ids), len(column_ids)):
                raise ValueError(
                    "Matrix shape " +
                    f"{matrix.shape} doesn't match dimensions (" +
                    f"{len(row_ids)}, {len(column_ids)})")
            self.matrix = matrix.astype(float)
        else:
            # Initialize empty matrix with zeros
            self.matrix = np.zeros(
                (len(row_ids), len(column_ids)), dtype=float)
            
        # Store the model name for later use
        self.model_name = model_name if model_name is not None else os.environ.get('MODEL_NAME', 'jinaai/jina-embeddings-v3')

        # These values are updated automatically right before matrix computation
        # do not set them manually! They are private for a reason
        self._row_pointers = None
        self._column_pointers = None

    @classmethod
    def create_empty(
            cls,
            row_ids: list,
            column_ids: list,
            name: str,
            row_load_function: callable = None,
            column_load_function: callable = None,
            model_name: str | None = None):
        """
        Create an empty SymilarityMatrix with just the row and column lists.

        Args:
            row_ids: List of row IDs
            column_ids: List of column IDs
            name: Name of the matrix
            model_name: The name of the SentenceTransformer model to initialize.

        Returns:
            SymilarityMatrix instance with zero-initialized matrix
        """
        return cls(
            row_ids,
            column_ids,
            name,
            row_load_function,
            column_load_function,
            model_name= model_name if model_name is not None else os.environ.get('MODEL_NAME', 'jinaai/jina-embeddings-v3'))

    def calculate(self, fake: int | None = None) -> None:
        """
        Calculate the similarity matrix based on the row and column lists.

        Args:
            fake: If provided, generates random values instead of actual embeddings
        """
        if fake is not None:
            # Generate random values with a normal distribution (mean, stddev)
            generator = np.random.default_rng(fake)
            self.matrix = generator.normal(
                0.35, 0.15, size=(len(self.row_ids), len(self.column_ids)))
            self.matrix.clip(-1, 1, out=self.matrix)
            return

        # Initialize the embedding model
        model = initialize_model(model_name=self.model_name)

        # Load row and column data using the passed functions
        row_texts = self.row_load_function()
        column_texts = self.column_load_function()

        # Get rid of eventual pointers returned and store them for later use
        row_texts, column_texts = self._detect_pointers(
            row_texts, column_texts)

        # Display matrix shapes for debugging
        logger.info(
            f"Matrix shape (final): {len(self.row_ids)} rows, " +
            f"{len(self.column_ids)} columns")
        logger.info(
            f"Matrix shape (windows): {len(row_texts)} rows, " +
            f"{len(column_texts)} columns")

        # Compute similarity matrix using memory-managed cosine similarity
        self.matrix = cos_sim_mem(model, row_texts, column_texts)

        # Apply aggregation if pointers were detected
        # TODO: de-hardcode the magic number for tail-weight
        self.matrix = compute_aggregation_tail_weight(
            self.matrix, self._row_pointers, self._column_pointers, tail_weight=0.2)

        # Ensure values are in valid range [-1, 1]
        # (should already be, but just in case)
        self.matrix = np.clip(self.matrix, -1, 1)

    def _detect_pointers(
        self,
        row_texts: list[str] | tuple[list[str], list[list[int]]],
        column_texts: list[str] | tuple[list[str], list[list[int]]]
    ) -> tuple[list[str], list[str]]:
        """
        This function eventually stores the passed pointers
        for later use to aggregate results and returns the texts
        deprived of the pointers.
        """
        # Store pointers for later use
        if isinstance(row_texts, tuple):
            row_texts, self._row_pointers = row_texts
        if isinstance(column_texts, tuple):
            column_texts, self._column_pointers = column_texts

        # Return just the texts
        return row_texts, column_texts

    def save(self, directory_path: Union[str, Path]) -> None:
        """
        Save the matrix and indices to files in the specified directory.

        Args:
            directory_path: Directory path where files will be saved
        """
        directory_path = Path(directory_path)
        directory_path.mkdir(parents=True, exist_ok=True)

        # Save matrix as .npy file using the name
        matrix_path = directory_path / f"{self.name}.npy"
        np.save(matrix_path, self.matrix)

        # Save indices as .json file using the name
        indices_path = directory_path / f"{self.name}.json"
        indices_data = {
            'name': self.name,
            'row_ids': self.row_ids,
            'column_ids': self.column_ids
        }

        with open(indices_path, 'w') as f:
            json.dump(indices_data, f, indent=2)

        logger.debug(f"Saved matrix to {matrix_path}")
        logger.debug(f"Saved indices to {indices_path}")

    @classmethod
    def load(cls, 
             directory_path: Union[str, Path],
             name: str, 
             model_name: str = os.environ.get('MODEL_NAME', 'jinaai/jina-embeddings-v3')) -> 'SimilarityMatrix':
        """
        Load the matrix and indices from files in the specified directory.

        Args:
            directory_path: Directory path where files are located
            name: Name of the matrix files to load
            model_name: The name of the SentenceTransformer model to initialize.

        Returns:
            SymilarityMatrix instance loaded from files
        """
        directory_path = Path(directory_path)

        # Load matrix from .npy file
        matrix_path = directory_path / f"{name}.npy"
        if not matrix_path.exists():
            raise FileNotFoundError(f"Matrix file not found: {matrix_path}")
        matrix = np.load(matrix_path)

        # Load indices from .json file
        indices_path = directory_path / f"{name}.json"
        if not indices_path.exists():
            raise FileNotFoundError(f"Indices file not found: {indices_path}")

        with open(indices_path, 'r') as f:
            indices_data = json.load(f)

        loaded_name = indices_data.get('name', name)
        row_ids = indices_data['row_ids']
        column_ids = indices_data['column_ids']

        return cls(row_ids, column_ids, loaded_name, matrix=matrix, model_name=model_name)

    def get_value(self, row_id: str, column_id: str) -> float:
        """
        Get a specific value from the matrix using row ID and column ID.

        Args:
            row_id: Row identifier
            column_id: Column identifier

        Returns:
            The value at the specified position
        """
        row_idx = self.row_ids.index(row_id)
        column_idx = self.column_ids.index(column_id)
        return self.matrix[row_idx, column_idx]

    def set_value(self, row_id: str, column_id: str, value: float) -> None:
        """
        Set a specific value in the matrix using row ID and column ID.

        Args:
            row_id: Row identifier
            column_id: Column identifier
            value: Value to set
        """
        row_idx = self.row_ids.index(row_id)
        column_idx = self.column_ids.index(column_id)
        self.matrix[row_idx, column_idx] = value

    def get_row(self, row_id: str) -> np.ndarray:
        """Get all column values for a specific row."""
        try:
            row_idx = self.row_ids.index(row_id)
        except ValueError:
            raise ValueError(f"'{row_id}' is not in list")

        return self.matrix[row_idx, :]

    def get_column(self, column_id: str) -> np.ndarray:
        """Get all row values for a specific column."""
        try:
            column_idx = self.column_ids.index(column_id)
        except ValueError:
            raise ValueError(f"'{column_id}' is not in list")

        return self.matrix[:, column_idx]

    def normalized(self,
                   min_v: Union[float, None] = None,
                   max_v: Union[float, None] = None
                   ) -> 'SimilarityMatrix':
        """
        Normalize the matrix values.

        Args:
            min_v(float|None): Minimum value for normalization (default: None, means use min value)
            max_v(float|None): Maximum value for normalization (default: None, means use max value)

        Returns:
            Normalized matrix (copy)
        """
        return SimilarityMatrix(
            row_ids=self.row_ids,
            column_ids=self.column_ids,
            name=self.name,
            matrix=normalize_array(self.matrix, min_v, max_v),
            row_load_function=self.row_load_function,
            column_load_function=self.column_load_function,
            model_name=self.model_name)

    @property
    def shape(self) -> tuple:
        """Get the shape of the matrix."""
        return self.matrix.shape

    @property
    def num_rows(self) -> int:
        """Get the number of rows."""
        return len(self.row_ids)

    @property
    def num_columns(self) -> int:
        """Get the number of column_ids."""
        return len(self.column_ids)

    def __iter__(self):
        """
        Iterate over (row_id, column_id, similarity_value) tuples.

        Yields:
            Tuple of (row_id, column_id, similarity_value)
        """
        for i, row_id in enumerate(self.row_ids):
            for j, col_id in enumerate(self.column_ids):
                yield (row_id, col_id, self.matrix[i, j])

    def __str__(self) -> str:
        return "SymilarityMatrix('" + \
            f"{self.name}', " + \
            f"{self.num_rows} rows, " + \
            f"{self.num_columns} columns)"

    def __repr__(self) -> str:
        return "SymilarityMatrix(name='" +\
            f"{self.name}', row_ids=" + \
            f"{self.row_ids}, column_ids=" + \
            f"{self.column_ids}, matrix.shape=" + \
            f"{self.matrix.shape})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SimilarityMatrix):
            return False
        return self.name == other.name and \
            self.row_ids == other.row_ids and \
            self.column_ids == other.column_ids and \
            np.array_equal(self.matrix, other.matrix)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __getitem__(self, key: Union[int, str]) -> Union[float, np.ndarray]:
        if isinstance(key, int):
            return self.matrix[key]
        elif isinstance(key, str):
            return self.get_row(key)
        else:
            raise TypeError("Invalid key type. Expected int or str.")
