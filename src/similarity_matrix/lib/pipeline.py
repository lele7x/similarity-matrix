import os
import ast
import numpy as np
import importlib.util
from pathlib import Path
from abc import ABC, abstractmethod

from similarity_matrix.lib.logging import logger
from similarity_matrix.lib.database import Database
from similarity_matrix.lib.matrix import SimilarityMatrix
from similarity_matrix.lib.matrix_chunk import ChunkedSimilarityMatrix
from similarity_matrix.lib.utils import camel_to_snake


class Pipeline(ABC):
    """
    This is an abstract class used to create a structure for a general
    pipeline to follow when it is necessary to compute a similarity
    matrix.

    It has some method to gather the raw data to produce the similarity
    matrix, being:
    - `get_row_ids` and `get_column_ids`: these functions are used to
    list all the instances that will be used.
    - `get_row_values` and `get_column_values`: these functions are used
    to retrieve the actual texts associated with the isntances on the
    rows and columns of the matrix.
    - `get_matrix`: is a concrete method that computes the similarity
    matrix using the results of the previous functions; this method will
    calculate the similarity matrix and save it in the specified path. If
    the matrix was already computed, it will load it from the specified
    path.

    There are other methods that are useful to update the state of the
    databse. It is assumed that the database is composed by three tables:
    a table contining the data relative to the instances on the rows of the
    matrix, lets call this table `A`, a table contining the data relative
    to the instances on the columns of the matrix, lets call this table `B`
    and a table containing the similarity matrix, lets call this table `AB`.

    The function provided to update the database are:
    - `update_db_row_table`: this function should update the table `A` in
    the database.
    - `update_db_column_table`: this function should update the table `B`
    in the database.
    - `update_db_similarity_matrix`: this function should update the table
    `AB` in the database.

    All of this functions should be implemented in the subclasses.

    Optionally a `postprocess_matrix` method can be implemented to
    perform some postprocessing on the similarity matrix. Please, keep in mind
    that this method will be called after the matrix has been computed but
    before it is saved on disk.
    """

    def __init__(
            self,
            name: str,
            db: Database,
            path: str = './matrices',
            chunk_size: int | None = None,
            model_name: str | None = None):
        self.name = name
        self.db = db
        self.path = path
        self.chunk_size = chunk_size
        self._sm = None
        self.model_name = model_name if model_name is not None else os.environ.get('MODEL_NAME', 'jinaai/jina-embeddings-v3')

        if not os.path.isdir(self.path):
            os.mkdir(self.path)

    # -----------------------------------------------------
    # Getters

    @abstractmethod
    def get_row_ids(self) -> list:
        """
        This function should implement a method to retrieve the row IDs.

        This could be a database query or any other method to retrieve the IDs.
        """
        pass

    @abstractmethod
    def get_column_ids(self) -> list:
        """
        This function should implement a method to retrieve the column IDs.

        This could be a database query or any other method to retrieve the IDs.
        """
        pass

    @abstractmethod
    def get_row_values(self) -> list[str] | tuple[list[str], list[list[int]]]:
        """
        This function should implement a method to retrieve the row values.

        This could be a database query or any other method to retrieve the values.

        The values should be the strings that will be used to create the similarity matrix.

        This function can return a tuple. This is useful when you have to compare big
        texts and want to compare windows of the texts instead of taking it into account
        as a whole. If that is the case then the second element of the tuple should
        be a list of pointers that make it possible to reconstruct the original
        texts and aggregate the similarity scores. This allow to get at the end a
        matrix which has the expected shape even if the number of similarities
        calculated was much greate.
        """
        pass

    @abstractmethod
    def get_column_values(
            self) -> list[str] | tuple[list[str], list[list[int]]]:
        """
        This function should implement a method to retrieve the column values.

        This could be a database query or any other method to retrieve the values.

        The values should be the strings that will be used to create the similarity matrix.

        This function can return a tuple. This is useful when you have to compare big
        texts and want to compare windows of the texts instead of taking it into account
        as a whole. If that is the case then the second element of the tuple should
        be a list of pointers that make it possible to reconstruct the original
        texts and aggregate the similarity scores. This allow to get at the end a
        matrix which has the expected shape even if the number of similarities
        calculated was much greate.
        """
        pass

    def _init_matrix(self) -> SimilarityMatrix:
        if self.chunk_size is not None:
            logger.info(
                'Using ChunkedSimilarityMatrix with chunk size %d',
                self.chunk_size)
            return ChunkedSimilarityMatrix(
                row_ids=self.get_row_ids(),
                column_ids=self.get_column_ids(),
                name=self.name,
                row_load_function=self.get_row_values,
                column_load_function=self.get_column_values,
                model_name=self.model_name,
                row_chunk_size=self.chunk_size,
                column_chunk_size=self.chunk_size)
        else:
            logger.info('Using SimilarityMatrix')
            return SimilarityMatrix.create_empty(
                row_ids=self.get_row_ids(),
                column_ids=self.get_column_ids(),
                name=self.name,
                row_load_function=self.get_row_values,
                column_load_function=self.get_column_values,
                model_name=self.model_name)

    def get_matrix(self) -> SimilarityMatrix:
        """
        Get the similarity matrix.

        The matrix is computed and saved on disk if it doesn't exist in the file system.
        """
        if os.path.isfile(os.path.join(self.path, self.name + '.npy')):
            logger.info('Loading matrix from file...')
            return SimilarityMatrix.load(self.path, self.name)

        if self._sm is None:
            # Create the empty matrix
            sm: SimilarityMatrix = self._init_matrix()

            # Compute the matrix
            logger.info('Computing matrix...')
            sm.calculate()

            # Keep the matrix in memory
            self._sm = sm

            # Postprocess the matrix
            self._sm = self.postprocess_matrix()

        # Save the matrix
        logger.info('Saving matrix to file...')
        self._sm.save(self.path)

        return self._sm

    # -----------------------------------------------------
    # Database

    @abstractmethod
    def update_db_row_table(self):
        """
        This function should implement a method to update the row table.

        By row table we mean the table that contains the row IDs and the row values.
        """
        pass

    @abstractmethod
    def update_db_column_table(self):
        """
        This function should implement a method to update the column table.

        By column table we mean the table that contains the column IDs and the column values.
        """
        pass

    @abstractmethod
    def update_db_matrix_table(self):
        """
        This function should implement a method to update the similarity matrix table.

        By matrix table we mean the table that contains the matrix.
        """
        pass

    # -----------------------------------------------------
    # Transform resulted matrix

    def postprocess_matrix(self) -> SimilarityMatrix:
        """
        This function should implement a method to postprocess the similarity matrix.

        This function is called right after the similarity matrix is computed and
        should override the values in `self.matrix` or modify it in-place.

        NOTE: Transformation to the matrix are computed before the matrix is saved to disk.

        This is not marked as abstract because it's not mandatory to implement it.
        """
        return self._sm

    # -----------------------------------------------------
    # Printing

    def __str__(self):
        return f'{self.__class__.__name__}(name=\'{self.name}\', ' +\
            f'{"not " if not self._sm else ""}computed)'

    def __repr__(self):
        return str(self)

    # -----------------------------------------------------
    # Utils class methods

    @classmethod
    def load_from_dir(cls, directory_path: str) -> dict:
        """
        Searches for all .py files in a directory (excluding __init__.py) and returns
        a dictionary mapping snake_case class names to class objects that can be instantiated
        and end with the suffix `Pipeline`.

        Args:
            directory_path (str): Path to the directory to search

        Returns:
            dict: Dictionary with snake_case class name as key and class object as value
        """
        result = {}
        directory = Path(directory_path)

        # Find all .py files excluding __init__.py
        py_files = [f for f in directory.glob(
            "*.py") if f.name != "__init__.py"]

        for py_file in py_files:
            try:
                # Parse the file to find class definitions
                with open(py_file, 'r', encoding='utf-8') as file:
                    tree = ast.parse(file.read())

                # Extract class names from AST
                class_names = [node.name for node in ast.walk(
                    tree) if isinstance(node, ast.ClassDef)]

                if class_names:
                    # Import the module dynamically
                    module_name = py_file.stem
                    spec = importlib.util.spec_from_file_location(
                        module_name, py_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # For each class found, add it to the result
                    for class_name in class_names:
                        if hasattr(
                                module,
                                class_name) and class_name.endswith('Pipeline'):
                            class_obj = getattr(module, class_name)
                            # Use snake_case version of class name as key
                            key = camel_to_snake(
                                class_name).replace('_pipeline', '')
                            result[key] = class_obj

            except Exception as e:
                logger.error(f"Error processing {py_file}: {e}")
                continue

        return result

    @classmethod
    def load_default_pipelines(cls) -> dict:
        """
        Searches for all .py files in the default pipeline directory (excluding
        __init__.py) and returns a dictionary mapping filename to class
        objects that can be instantiated that end with the suffix `Pipeline`.

        Returns:
            dict: Dictionary with filename as key and class object as value
        """
        system_pipelines = Path(
            os.path.dirname(
                importlib.import_module('similarity_matrix').__file__)) / 'pipelines'
        return cls.load_from_dir(system_pipelines)
