import os
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from similarity_matrix.lib.matrix import SimilarityMatrix
from similarity_matrix.lib.matrix_chunk import ChunkedSimilarityMatrix
from similarity_matrix.lib.pipeline import Pipeline
from similarity_matrix.lib.database import Database

@pytest.fixture
def basic_ids():
    return ["r1"],["c1"]

@pytest.fixture
def fake_load_functions():
    return (
        lambda: ["text1"],
        lambda: ["text2"]
    )

class ConcretePipeline(Pipeline):
    """Concrete implementation of Pipeline for testing purposes."""

    def __init__(
            self, 
            name: str, 
            db: Database, 
            path: str = './matrices', 
            chunk_size: int | None = None,
            model_name: str = 'jinaai/jina-embeddings-v3'):
        super().__init__(name, db, path, chunk_size=chunk_size, model_name=model_name)
        self.row_ids = [1, 2, 3]
        self.column_ids = [10, 20, 30]
        self.row_values = ["text1", "text2", "text3"]
        self.column_values = ["textA", "textB", "textC"]

    def get_row_ids(self) -> list:
        return self.row_ids

    def get_column_ids(self) -> list:
        return self.column_ids

    def get_row_values(self) -> list[str]:
        return self.row_values

    def get_column_values(self) -> list[str]:
        return self.column_values

    def update_db_row_table(self):
        # Just mock method
        pass

    def update_db_column_table(self):
        # Just mock method
        pass

    def update_db_matrix_table(self):
        # Just mock method
        pass

class TestModelName:
    ## --------------------------------------------------------------
    ## SimilarityMatrix model_name tests
    ## --------------------------------------------------------------

    # Verify that model_name was appropriately stored by the 
    # SimilarityMatrix class.
    def test_similarity_matrix_model_name_stored(self, basic_ids):
        row_ids, column_ids = basic_ids
        matrix = SimilarityMatrix(
            row_ids=row_ids,
            column_ids=column_ids,
            name="test",
            model_name="test-model"
        )

        assert matrix.model_name == "test-model"

    # Ensure that when no model_name is provided, SimilarityMatrix 
    # assigns the correct default value.
    def test_similarity_matrix_default_model_name(self, basic_ids):
        row_ids, column_ids = basic_ids
        matrix = SimilarityMatrix(
            row_ids=row_ids,
            column_ids=column_ids,
            name="test"
        )

        assert matrix.model_name == "jinaai/jina-embeddings-v3"

    # Verify that model_name is passed to initialize_model() when 
    # calculate() is called.
    @patch("similarity_matrix.lib.matrix.cos_sim_mem")
    @patch("similarity_matrix.lib.matrix.initialize_model")
    def test_similarity_matrix_model_name_passed_to_initialize(
            self,
            mock_initialize,
            mock_cos_sim,
            basic_ids,
            fake_load_functions):

        mock_initialize.return_value = MagicMock()
        mock_cos_sim.return_value = np.array([[0.5]])

        row_ids, column_ids = basic_ids
        row_loader, column_loader = fake_load_functions

        matrix = SimilarityMatrix(
            row_ids=row_ids,
            column_ids=column_ids,
            name="test",
            row_load_function=row_loader,
            column_load_function=column_loader,
            model_name="test-model"
        )

        matrix.calculate()

        mock_initialize.assert_called_once_with(model_name="test-model")

    # Checks that an empty matrix is created with zero values, 
    # correct IDs, and a custom model_name
    def test_create_empty_with_custom_model_name(self, basic_ids):
        row_ids, column_ids = basic_ids

        matrix = SimilarityMatrix.create_empty(
            row_ids=row_ids,
            column_ids=column_ids,
            name="empty_custom",
            model_name="test-model"
        )

        # Controlli
        assert matrix.shape == (1, 1)
        assert np.all(matrix.matrix == 0)
        assert matrix.row_ids == row_ids
        assert matrix.column_ids == column_ids
        assert matrix.model_name == "test-model"

    # Checks that an empty matrix is created with zero values, 
    # correct IDs, and the default model_name
    def test_create_empty_with_default_model_name(self, basic_ids):
        row_ids, column_ids = basic_ids

        matrix = SimilarityMatrix.create_empty(
            row_ids=row_ids,
            column_ids=column_ids,
            name="empty_default"
        )

        # Controlli
        assert matrix.shape == (1, 1)
        assert np.all(matrix.matrix == 0)
        assert matrix.row_ids == row_ids
        assert matrix.column_ids == column_ids
        assert matrix.model_name == "jinaai/jina-embeddings-v3"

    # Verifies that loading a saved matrix restores data and IDs while ù
    # keeping a custom model_name
    def test_load_preserves_custom_model_name(self, tmp_path, basic_ids):
        # Setup matrix e JSON file
        row_ids, column_ids = basic_ids
        matrix_data = np.array([[0.42]])
        name = "test_matrix"
        
        # Save matrix and json in a temporary directory
        np.save(tmp_path / f"{name}.npy", matrix_data)
        with open(tmp_path / f"{name}.json", "w") as f:
            json.dump({
                "name": name,
                "row_ids": row_ids,
                "column_ids": column_ids
            }, f)
        
        # Load matrix with custom model_name
        loaded_matrix = SimilarityMatrix.load(
            directory_path=tmp_path,
            name=name,
            model_name="test-model"
        )
        
        assert loaded_matrix.model_name == "test-model"

        # Check that the loaded matrix data and IDs match what was saved
        assert np.array_equal(loaded_matrix.matrix, matrix_data)
        assert loaded_matrix.row_ids == row_ids
        assert loaded_matrix.column_ids == column_ids

    # Verifies that loading a saved matrix without specifying model_name 
    # assigns the default value while restoring data and IDs
    def test_load_uses_default_model_name_if_not_specified(self, tmp_path, basic_ids):
        # Setup matrix e file JSON
        row_ids, column_ids = basic_ids
        matrix_data = np.array([[0.42]])
        name = "test_matrix_default"
        
        np.save(tmp_path / f"{name}.npy", matrix_data)
        with open(tmp_path / f"{name}.json", "w") as f:
            json.dump({
                "name": name,
                "row_ids": row_ids,
                "column_ids": column_ids
            }, f)
        
        # Carica senza specificare model_name
        loaded_matrix = SimilarityMatrix.load(
            directory_path=tmp_path,
            name=name
        )
        
        assert loaded_matrix.model_name == "jinaai/jina-embeddings-v3"
        assert np.array_equal(loaded_matrix.matrix, matrix_data)
        assert loaded_matrix.row_ids == row_ids
        assert loaded_matrix.column_ids == column_ids
    
    ## --------------------------------------------------------------
    ## ChunkedSimilarityMatrix model_name tests
    ## --------------------------------------------------------------

    # Verify that model_name was appropriately stored by the 
    # ChunkedSimilarityMatrix class.
    def test_chunked_matrix_model_name_stored(self, basic_ids):
        row_ids, column_ids = basic_ids

        matrix = ChunkedSimilarityMatrix(
            row_ids=row_ids,
            column_ids=column_ids,
            name="test",
            model_name="custom-model"
        )

        assert matrix.model_name == "custom-model"

    # Ensure that when no model_name is provided, ChunkedSimilarityMatrix 
    # assigns the correct default value.
    def test_chunked_matrix_default_model_name(self, basic_ids):
        """Verifica valore di default."""
        row_ids, column_ids = basic_ids

        matrix = ChunkedSimilarityMatrix(
            row_ids=row_ids,
            column_ids=column_ids,
            name="test"
        )

        assert matrix.model_name == "jinaai/jina-embeddings-v3"

    # Verify that model_name is passed to initialize_model() when 
    # calculate() is called.
    @patch("similarity_matrix.lib.matrix_chunk.cos_sim_mem")
    @patch("similarity_matrix.lib.matrix_chunk.initialize_model")
    def test_chunked_matrix_model_name_passed_to_initialize(
            self,
            mock_initialize,
            mock_cos_sim,
            basic_ids,
            fake_load_functions):

        mock_initialize.return_value = MagicMock()
        mock_cos_sim.return_value = np.array([[0.5]])

        row_ids, column_ids = basic_ids
        row_loader, column_loader = fake_load_functions

        matrix = ChunkedSimilarityMatrix(
            row_ids=row_ids,
            column_ids=column_ids,
            name="test",
            row_load_function=row_loader,
            column_load_function=column_loader,
            model_name="test-model"
        )

        matrix.calculate()

        mock_initialize.assert_called_once_with(model_name="test-model")

    ## --------------------------------------------------------------
    ## Pipiline model_name tests
    ## --------------------------------------------------------------

    # Verify that model_name is stored in the Pipeline
    def test_pipeline_model_name_stored(self):
        db_mock = MagicMock()
        pipeline = ConcretePipeline(
            name="test",
            db=db_mock,
            model_name="test-model"
        )

        assert pipeline.model_name == "test-model"

    # Verify that if no model_name is provided, the Pipeline assigns 
    # the default value
    def test_pipeline_default_model_name(self):
        db_mock = MagicMock()
        pipeline = ConcretePipeline(
            name="test",
            db=db_mock
        )

        assert pipeline.model_name == "jinaai/jina-embeddings-v3"

    # Verify that when a Pipeline is initialized, it passes the model_name
    # to the SimilarityMatrix it creates (when chunk_size is None)
    @patch("similarity_matrix.lib.pipeline.SimilarityMatrix.create_empty")
    def test_pipeline_passes_model_name_to_similarity_matrix(self, mock_create):
        db_mock = MagicMock()

        pipeline = ConcretePipeline(
            name="test",
            db=db_mock,
            model_name="test-model"
        )

        pipeline.chunk_size = None
        pipeline._init_matrix()

        _, kwargs = mock_create.call_args
        assert kwargs["model_name"] == "test-model"

    # Verify that when a Pipeline is initialized, it passes the model_name
    # to the ChunkedSimilarityMatrix it creates (when chunk_size is set)
    @patch("similarity_matrix.lib.pipeline.ChunkedSimilarityMatrix")
    def test_pipeline_passes_model_name_to_chunked_matrix(self, mock_chunked):
        db_mock = MagicMock()

        pipeline = ConcretePipeline(
            name="test",
            db=db_mock,
            model_name="test-model",
            chunk_size=10
        )

        pipeline._init_matrix()

        _, kwargs = mock_chunked.call_args
        assert kwargs["model_name"] == "test-model"