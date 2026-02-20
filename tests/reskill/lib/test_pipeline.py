import os
import pytest
import shutil
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, call
import tempfile

from similarity_matrix.lib.pipeline import Pipeline
from similarity_matrix.lib.database import Database


class ConcretePipeline(Pipeline):
    """Concrete implementation of Pipeline for testing purposes."""

    def __init__(self, name: str, db: Database, path: str = './matrices'):
        super().__init__(name, db, path)
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


class TestPipeline:
    """Test cases for the Pipeline class."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database."""
        return Mock(spec=Database)

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def pipeline(self, mock_db, temp_dir):
        """Create a concrete pipeline instance for testing."""
        return ConcretePipeline("test_pipeline", mock_db, temp_dir)

    def test_init_creates_directory(self, mock_db, temp_dir):
        """Test that __init__ creates the matrices directory if it doesn't exist."""
        matrix_path = os.path.join(temp_dir, "new_matrices")
        assert not os.path.exists(matrix_path)

        pipeline = ConcretePipeline("test", mock_db, matrix_path)

        assert os.path.exists(matrix_path)
        assert os.path.isdir(matrix_path)
        assert pipeline.name == "test"
        assert pipeline.db == mock_db
        assert pipeline.path == matrix_path

    def test_init_existing_directory(self, mock_db, temp_dir):
        """Test that __init__ works with existing directory."""
        pipeline = ConcretePipeline("test", mock_db, temp_dir)

        assert pipeline.name == "test"
        assert pipeline.db == mock_db
        assert pipeline.path == temp_dir

    def test_abstract_methods_must_be_implemented(self, mock_db):
        """Test that Pipeline cannot be instantiated without implementing abstract methods."""
        with pytest.raises(TypeError):
            Pipeline("test", mock_db)

    def test_get_row_ids(self, pipeline):
        """Test get_row_ids method."""
        result = pipeline.get_row_ids()
        assert result == [1, 2, 3]

    def test_get_column_ids(self, pipeline):
        """Test get_column_ids method."""
        result = pipeline.get_column_ids()
        assert result == [10, 20, 30]

    def test_get_row_values(self, pipeline):
        """Test get_row_values method."""
        result = pipeline.get_row_values()
        assert result == ["text1", "text2", "text3"]

    def test_get_column_values(self, pipeline):
        """Test get_column_values method."""
        result = pipeline.get_column_values()
        assert result == ["textA", "textB", "textC"]

    def test_get_path_property(self, pipeline):
        """Test get_path method (assuming it exists)."""
        # Note: get_path() is called in get_matrix() but not defined in the class
        # This might be a missing method or inherited from a parent class
        with pytest.raises(AttributeError):
            pipeline.get_path()

    @patch('similarity_matrix.lib.pipeline.SimilarityMatrix')
    @patch('similarity_matrix.lib.pipeline.logger')
    def test_get_matrix_loads_existing_file(
            self, mock_logger, mock_similarity_matrix, pipeline):
        """Test that get_matrix loads existing matrix file."""
        # Create a mock matrix file
        matrix_file = os.path.join(pipeline.path, pipeline.name + '.npy')
        np.save(matrix_file, np.array([[1, 2], [3, 4]]))

        # Mock the SimilarityMatrix.load method
        mock_matrix_instance = Mock()
        mock_matrix_instance.matrix = np.array([[1, 2], [3, 4]])
        mock_similarity_matrix.load.return_value = mock_matrix_instance

        # Add get_path method to pipeline for this test
        pipeline.get_path = Mock(return_value=pipeline.path)

        result = pipeline.get_matrix()

        mock_logger.info.assert_called_with('Loading matrix from file...')
        mock_similarity_matrix.load.assert_called_once_with(
            pipeline.path, pipeline.name)
        np.testing.assert_array_equal(
            result.matrix, np.array([[1, 2], [3, 4]]))

    @patch('similarity_matrix.lib.pipeline.SimilarityMatrix')
    @patch('similarity_matrix.lib.pipeline.logger')
    def test_get_matrix_computes_new_matrix(
            self, mock_logger, mock_similarity_matrix, pipeline):
        """Test that get_matrix computes and saves new matrix when file doesn't exist."""
        # Mock the SimilarityMatrix class
        mock_matrix_instance = Mock()
        mock_matrix_instance.matrix = np.array([[0.5, 0.8], [0.3, 0.9]])
        mock_similarity_matrix.create_empty.return_value = mock_matrix_instance

        # Add get_path method to pipeline for this test
        pipeline.get_path = Mock(return_value=pipeline.path)

        result = pipeline.get_matrix()

        # Verify the matrix creation process
        mock_similarity_matrix.create_empty.assert_called_once_with(
            row_ids=[1, 2, 3],
            column_ids=[10, 20, 30],
            name=pipeline.name,
            row_load_function=pipeline.get_row_values,
            column_load_function=pipeline.get_column_values,
            model_name=pipeline.model_name
        )

        # Verify the matrix was calculated and saved
        mock_matrix_instance.calculate.assert_called_once()
        mock_matrix_instance.save.assert_called_once_with(pipeline.path)

        # Verify logging
        expected_calls = [
            call('Computing matrix...'),
            call('Saving matrix to file...')
        ]
        mock_logger.info.assert_has_calls(expected_calls)

        np.testing.assert_array_equal(
            result.matrix, np.array([[0.5, 0.8], [0.3, 0.9]]))

    def test_postprocess_matrix_default_implementation(self, pipeline):
        """Test that postprocess_matrix has a default implementation that does nothing."""
        # Should not raise any exception
        result = pipeline.postprocess_matrix()
        assert result is None

    def test_database_update_methods_are_abstract(self, pipeline):
        """Test that database update methods are implemented in concrete class."""
        # These should not raise exceptions since they're implemented in
        # ConcretePipeline
        pipeline.update_db_row_table()
        pipeline.update_db_column_table()
        pipeline.update_db_matrix_table()

    def test_str_representation_not_computed(self, pipeline):
        """Test that __str__ method returns expected string representation."""
        expected_str = (
            f"ConcretePipeline(name='{pipeline.name}', "
            f"not computed)"
        )
        assert str(pipeline) == expected_str

    def test_str_representation_computed(self, pipeline):
        """Test that __str__ method returns expected string representation."""
        pipeline._sm = 'some value'
        expected_str = (
            f"ConcretePipeline(name='{pipeline.name}', "
            f"computed)"
        )
        assert str(pipeline) == expected_str


class TestPipelineWithCustomPostprocess:
    """Test Pipeline with custom postprocess implementation."""

    class CustomPipeline(ConcretePipeline):
        def __init__(self, name: str, db: Database, path: str = './matrices'):
            super().__init__(name, db, path)
            self.postprocess_called = False

        def postprocess_matrix(self):
            self.postprocess_called = True
            return "postprocessed"

    @pytest.fixture
    def mock_db(self):
        return Mock(spec=Database)

    @pytest.fixture
    def temp_dir(self):
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def custom_pipeline(self, mock_db, temp_dir):
        return self.CustomPipeline("custom_test", mock_db, temp_dir)

    def test_custom_postprocess_matrix(self, custom_pipeline):
        """Test that custom postprocess_matrix implementation works."""
        result = custom_pipeline.postprocess_matrix()
        assert custom_pipeline.postprocess_called is True
        assert result == "postprocessed"


class TestPipelineLoadFromDir:

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def sample_py_files(self, temp_dir):
        """Create sample Python files for testing"""
        files = {}

        # File with a single class
        files['single_class.py'] = '''
class SingleClassPipeline:
    def __init__(self):
        self.value = "single"
'''

        # File with multiple classes (should return first one)
        files['multiple_classes.py'] = '''
class FirstClassPipeline:
    def __init__(self):
        self.value = "first"

class SecondClassPipeline:
    def __init__(self):
        self.value = "second"
'''

        # File with no classes
        files['no_classes.py'] = '''
def some_function():
    return "no classes here"

SOME_CONSTANT = 42
'''

        # File with syntax error
        files['syntax_error.py'] = '''
class BrokenClassPipeline:
    def __init__(self
        # Missing closing parenthesis and colon
'''

        # __init__.py file (should be excluded)
        files['__init__.py'] = '''
class InitClassPipeline:
    pass
'''

        # Create the files
        for filename, content in files.items():
            file_path = Path(temp_dir) / filename
            with open(file_path, 'w') as f:
                f.write(content)

        return files

    def test_load_from_dir_with_valid_files(self, temp_dir, sample_py_files):
        """Test loading classes from directory with valid Python files"""
        result = Pipeline.load_from_dir(temp_dir)

        # Should find classes from files that have them
        assert 'first_class' in result
        assert 'second_class' in result
        assert 'single_class' in result

        # Should not include files without classes
        assert 'no_classes' not in result

        # Should not include __init__.py
        assert '__init__' not in result

        # Verify the classes can be instantiated
        single_instance = result['single_class']()
        assert single_instance.value == "single"

        multiple_instance = result['first_class']()
        assert multiple_instance.value == "first"

        multiple_instance = result['second_class']()
        assert multiple_instance.value == "second"

    def test_load_from_dir_excludes_init_py(self, temp_dir, sample_py_files):
        """Test that __init__.py files are excluded"""
        result = Pipeline.load_from_dir(temp_dir)
        assert '__init__' not in result

    def test_load_from_dir_handles_syntax_errors(
            self, temp_dir, sample_py_files):
        """Test that files with syntax errors are handled gracefully"""
        # Should not raise an exception
        result = Pipeline.load_from_dir(temp_dir)

        # Should not include the file with syntax error
        assert 'syntax_error' not in result

        # Should still include valid files
        assert 'single_class' in result

    def test_load_from_dir_empty_directory(self, temp_dir):
        """Test loading from an empty directory"""
        result = Pipeline.load_from_dir(temp_dir)
        assert result == {}

    def test_load_from_dir_nonexistent_directory(self):
        """Test loading from a nonexistent directory"""
        result = Pipeline.load_from_dir("/nonexistent/directory")
        assert result == {}

    def test_load_from_dir_directory_with_only_init(self, temp_dir):
        """Test directory with only __init__.py file"""
        init_path = Path(temp_dir) / "__init__.py"
        with open(init_path, 'w') as f:
            f.write("class InitClass:\n    pass\n")

        result = Pipeline.load_from_dir(temp_dir)
        assert result == {}

    def test_load_from_dir_file_without_classes(self, temp_dir):
        """Test file with functions but no classes"""
        file_path = Path(temp_dir) / "functions_only.py"
        with open(file_path, 'w') as f:
            f.write('''
def function1():
    return "hello"

def function2():
    return "world"

CONSTANT = 123
''')

        result = Pipeline.load_from_dir(temp_dir)
        assert 'functions_only' not in result
        assert result == {}

    def test_load_from_dir_class_with_inheritance(self, temp_dir):
        """Test loading classes that inherit from other classes"""
        file_path = Path(temp_dir) / "inheritance.py"
        with open(file_path, 'w') as f:
            f.write('''
class BaseClassPipeline:
    def __init__(self):
        self.base_value = "base"

class DerivedClassPipeline(BaseClassPipeline):
    def __init__(self):
        super().__init__()
        self.derived_value = "derived"
''')

        result = Pipeline.load_from_dir(temp_dir)
        assert 'base_class' in result
        assert 'derived_class' in result

        # Should get the first class (BaseClass)
        instance = result['derived_class']()
        assert hasattr(instance, 'base_value')
        assert instance.base_value == "base"

    def test_load_from_dir_returns_class_objects(self, temp_dir):
        """Test that the method returns actual class objects, not strings"""
        file_path = Path(temp_dir) / "test_class.py"
        with open(file_path, 'w') as f:
            f.write('''
class TestClassPipeline:
    def test_method(self):
        return "test_result"
''')

        result = Pipeline.load_from_dir(temp_dir)

        # Verify it's a class object
        assert 'test_class' in result
        test_class = result['test_class']
        assert callable(test_class)
        assert hasattr(test_class, '__name__')
        assert test_class.__name__ == 'TestClassPipeline'

        # Verify we can instantiate and call methods
        instance = test_class()
        assert instance.test_method() == "test_result"

    @patch('ast.parse', side_effect=PermissionError("Permission denied"))
    def test_load_from_dir_permission_error(self, mock_open_func, temp_dir):
        """Test handling of permission errors when reading files"""
        # Create a file first
        file_path = Path(temp_dir) / "test.py"
        with open(file_path, 'w') as f:
            f.write("class TestClassPipeline:\n    pass\n")

        # Now the mocked open will raise PermissionError
        result = Pipeline.load_from_dir(temp_dir)

        # Should handle the error gracefully and return empty dict
        assert result == {}

    def test_load_from_dir_import_error(self, temp_dir):
        """Test handling of import errors"""
        file_path = Path(temp_dir) / "import_error.py"
        with open(file_path, 'w') as f:
            f.write('''
import nonexistent_module

class TestClass:
    pass
''')

        # Should handle import error gracefully
        result = Pipeline.load_from_dir(temp_dir)
        assert 'import_error' not in result


class TestPipelineErrorHandling:
    """Test error handling in Pipeline class."""

    @pytest.fixture
    def mock_db(self):
        return Mock(spec=Database)

    def test_init_with_invalid_path_permissions(self, mock_db):
        """Test initialization with path that cannot be created."""
        # This test might be platform-specific
        with patch('os.mkdir', side_effect=PermissionError("Permission denied")):
            with patch('os.path.isdir', return_value=False):
                with pytest.raises(PermissionError):
                    ConcretePipeline("test", mock_db, "/root/invalid_path")

    @patch('similarity_matrix.lib.pipeline.SimilarityMatrix')
    def test_get_matrix_with_similarity_matrix_error(
            self, mock_similarity_matrix, mock_db):
        """Test get_matrix when SimilarityMatrix operations fail."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = ConcretePipeline("test", mock_db, temp_dir)
            pipeline.get_path = Mock(return_value=temp_dir)

            # Mock SimilarityMatrix.create_empty to raise an exception
            mock_similarity_matrix.create_empty.side_effect = Exception(
                "Matrix creation failed")

            with pytest.raises(Exception, match="Matrix creation failed"):
                pipeline.get_matrix()


if __name__ == "__main__":
    pytest.main([__file__])
