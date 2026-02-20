import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
import argparse

# Import the main module (adjust import path as needed)
from similarity_matrix.main import (
    main,
    update_db_row_table,
    update_db_column_table,
    update_db_matrix,
    compute_similarity
)


class TestSubCommands:
    """Test the sub-command functions"""

    def test_update_db_row_table(self):
        """Test update_db_row_table function"""
        mock_pipeline = Mock()
        mock_pipeline.update_db_row_table.return_value = "success"

        result = update_db_row_table(mock_pipeline)

        mock_pipeline.update_db_row_table.assert_called_once()
        assert result == "success"

    def test_update_db_column_table(self):
        """Test update_db_column_table function"""
        mock_pipeline = Mock()
        mock_pipeline.update_db_column_table.return_value = "success"

        result = update_db_column_table(mock_pipeline)

        mock_pipeline.update_db_column_table.assert_called_once()
        assert result == "success"

    def test_update_db_matrix(self):
        """Test update_db_matrix function"""
        mock_pipeline = Mock()
        mock_pipeline.update_db_matrix_table.return_value = "success"

        result = update_db_matrix(mock_pipeline)

        mock_pipeline.update_db_matrix_table.assert_called_once()
        assert result == "success"

    def test_compute_similarity(self):
        """Test compute_similarity function"""
        mock_pipeline = Mock()
        mock_pipeline.get_matrix.return_value = "matrix_data"

        result = compute_similarity(mock_pipeline)

        mock_pipeline.get_matrix.assert_called_once()
        assert result == "matrix_data"


class TestMainFunction:
    """Test the main function with various scenarios"""

    @patch('similarity_matrix.main.load_dotenv')
    @patch('similarity_matrix.main.Database')
    @patch('similarity_matrix.main.Pipeline')
    def test_main_with_valid_args_and_update_db_row_table(
        self, mock_pipeline_class, mock_database_class, mock_load_dotenv
    ):
        """Test main function with valid arguments for update-db-row"""
        # Setup
        test_args = [
            'similarity_matrix',
            '--pipeline', 'test_pipeline',
            '--db-user', 'testuser',
            '--db-password', 'testpass',
            'update-db-row'
        ]

        # Mock database
        mock_db = Mock()
        mock_db.test_connection.return_value = None
        mock_database_class.return_value = mock_db

        # Mock pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.update_db_row_table.return_value = "success"
        mock_available_pipelines = {
            'test_pipeline': Mock(
                return_value=mock_pipeline_instance)}
        mock_pipeline_class.load_default_pipelines.return_value = mock_available_pipelines

        # Test
        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()

        # Assertions
        assert exc_info.value.code == 0
        mock_load_dotenv.assert_called_once()
        mock_database_class.assert_called_once_with(
            'localhost', 3306, 'testuser', 'testpass', None
        )
        mock_db.test_connection.assert_called_once()
        mock_pipeline_class.load_default_pipelines.assert_called_once()
        mock_pipeline_instance.update_db_row_table.assert_called_once()

    @patch('similarity_matrix.main.load_dotenv')
    def test_main_no_command_shows_help(self, mock_load_dotenv):
        """Test main function with no command shows help and exits"""
        with patch('sys.argv', ['similarity_matrix', '--db-username', 'testuser', '--db-password', 'testpass']):
            with pytest.raises(SystemExit):
                main()

    @patch('similarity_matrix.main.load_dotenv')
    @patch('os.environ.get')
    def test_main_db_credentials_from_env(
            self, mock_env_get, mock_load_dotenv):
        """Test main function gets DB credentials from environment"""
        test_args = [
            'similarity_matrix',
            '--pipeline', 'test_pipeline',
            'update-db-row'
        ]

        # Mock environment variables
        def env_side_effect(key, default=None):
            if key == 'DB_USERNAME':
                return 'env_user'
            elif key == 'DB_PASSWORD':
                return 'env_pass'
            elif key == 'DB_DATABASE':
                return 'env_db_name'
            return default

        mock_env_get.side_effect = env_side_effect

        with patch('similarity_matrix.main.Database') as mock_database_class, \
                patch('similarity_matrix.main.Pipeline') as mock_pipeline_class, \
                patch('sys.argv', test_args):

            # Mock database
            mock_db = Mock()
            mock_db.test_connection.return_value = None
            mock_database_class.return_value = mock_db

            # Mock pipeline
            mock_pipeline_instance = Mock()
            mock_available_pipelines = {
                'test_pipeline': Mock(
                    return_value=mock_pipeline_instance)}
            mock_pipeline_class.load_from_dir.return_value = mock_available_pipelines

            with pytest.raises(SystemExit):
                main()

            mock_database_class.assert_called_once_with(
                'localhost', 3306, 'env_user', 'env_pass', 'env_db_name'
            )

    @patch('similarity_matrix.main.load_dotenv')
    def test_main_missing_db_credentials_exits(self, mock_load_dotenv):
        """Test main function exits when DB credentials are missing"""
        test_args = [
            'similarity_matrix',
            '--pipeline', 'test_pipeline',
            'update-db-row'
        ]

        with patch('similarity_matrix.main.Database') as mock_database_class, \
                patch('similarity_matrix.main.Pipeline') as mock_pipeline_class, \
                patch('sys.argv', test_args):

            # Mock pipeline
            mock_pipeline_instance = Mock()
            mock_available_pipelines = {
                'test_pipeline': Mock(
                    return_value=mock_pipeline_instance)}
            mock_pipeline_class.load_from_dir.return_value = mock_available_pipelines

            with pytest.raises(SystemExit):
                main()

            mock_database_class.assert_not_called()

    @patch('similarity_matrix.main.load_dotenv')
    @patch('similarity_matrix.main.Database')
    def test_main_database_connection_error(
            self, mock_database_class, mock_load_dotenv):
        """Test main function handles database connection errors"""
        test_args = [
            'similarity_matrix',
            '--pipeline', 'test_pipeline',
            '--db-user', 'testuser',
            '--db-password', 'testpass',
            'update-db-row'
        ]

        # Mock database connection error
        mock_db = Mock()
        connection_error = Exception("Connection failed")
        mock_db.test_connection.return_value = connection_error
        mock_database_class.return_value = mock_db

        with patch('sys.argv', test_args):
            with pytest.raises(Exception) as exc_info:
                main()

        assert str(exc_info.value) == "Connection failed"

    @patch('similarity_matrix.main.load_dotenv')
    @patch('similarity_matrix.main.Database')
    @patch('similarity_matrix.main.Pipeline')
    def test_main_list_pipelines(
        self, mock_pipeline_class, mock_database_class, mock_load_dotenv
    ):
        """Test main function with --list-pipelines argument"""
        test_args = [
            'similarity_matrix',
            '--db-user', 'testuser',
            '--db-password', 'testpass',
            '--list-pipelines'
        ]

        # Mock database
        mock_db = Mock()
        mock_db.test_connection.return_value = None
        mock_database_class.return_value = mock_db

        # Mock pipelines
        mock_available_pipelines = {'pipeline1': Mock(), 'pipeline2': Mock()}
        mock_pipeline_class.load_from_dir.return_value = mock_available_pipelines

        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 0

    @patch('similarity_matrix.main.load_dotenv')
    @patch('similarity_matrix.main.Database')
    @patch('similarity_matrix.main.Pipeline')
    def test_main_missing_pipeline_argument(
        self, mock_pipeline_class, mock_database_class, mock_load_dotenv
    ):
        """Test main function exits when pipeline argument is missing"""
        test_args = [
            'similarity_matrix',
            '--db-user', 'testuser',
            '--db-password', 'testpass',
            'update-db-row'
        ]

        # Mock database
        mock_db = Mock()
        mock_db.test_connection.return_value = None
        mock_database_class.return_value = mock_db

        # Mock pipelines
        mock_available_pipelines = {'pipeline1': Mock()}
        mock_pipeline_class.load_from_dir.return_value = mock_available_pipelines

        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1

    @patch('similarity_matrix.main.load_dotenv')
    @patch('similarity_matrix.main.Database')
    @patch('similarity_matrix.main.Pipeline')
    def test_main_pipeline_not_found(
        self, mock_pipeline_class, mock_database_class, mock_load_dotenv
    ):
        """Test main function exits when specified pipeline is not found"""
        test_args = [
            'similarity_matrix',
            '--pipeline', 'nonexistent_pipeline',
            '--db-user', 'testuser',
            '--db-password', 'testpass',
            'update-db-row'
        ]

        # Mock database
        mock_db = Mock()
        mock_db.test_connection.return_value = None
        mock_database_class.return_value = mock_db

        # Mock pipelines
        mock_available_pipelines = {'pipeline1': Mock()}
        mock_pipeline_class.load_default_pipelines.return_value = {}
        mock_pipeline_class.load_from_dir.return_value = mock_available_pipelines

        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1

    @patch('similarity_matrix.main.load_dotenv')
    @patch('similarity_matrix.main.Database')
    @patch('similarity_matrix.main.Pipeline')
    def test_main_debug_mode(
        self, mock_pipeline_class, mock_database_class, mock_load_dotenv
    ):
        """Test main function with debug mode enabled"""
        test_args = [
            'similarity_matrix',
            '--debug',
            '--pipeline', 'test_pipeline',
            '--db-user', 'testuser',
            '--db-password', 'testpass',
            'update-db-row'
        ]

        # Mock database
        mock_db = Mock()
        mock_db.test_connection.return_value = None
        mock_database_class.return_value = mock_db

        # Mock pipeline
        mock_pipeline_instance = Mock()
        mock_available_pipelines = {
            'test_pipeline': Mock(
                return_value=mock_pipeline_instance)}
        mock_pipeline_class.load_default_pipelines.return_value = mock_available_pipelines

        with patch('similarity_matrix.main.logger') as mock_logger, \
                patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0
            # Verify debug mode was set
            mock_logger.setLevel.assert_called()

    @patch('similarity_matrix.main.load_dotenv')
    @patch('similarity_matrix.main.Database')
    @patch('similarity_matrix.main.Pipeline')
    def test_main_custom_database_settings(
        self, mock_pipeline_class, mock_database_class, mock_load_dotenv
    ):
        """Test main function with custom database settings"""
        test_args = [
            'similarity_matrix',
            '--db-host', 'custom-host',
            '--db-port', '5432',
            '--db-name', 'custom_db',
            '--pipeline', 'test_pipeline',
            '--db-user', 'testuser',
            '--db-password', 'testpass',
            'update-db-row'
        ]

        # Mock database
        mock_db = Mock()
        mock_db.test_connection.return_value = None
        mock_database_class.return_value = mock_db

        # Mock pipeline
        mock_pipeline_instance = Mock()
        mock_available_pipelines = {
            'test_pipeline': Mock(
                return_value=mock_pipeline_instance)}
        mock_pipeline_class.load_default_pipelines.return_value = mock_available_pipelines

        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 0
        mock_database_class.assert_called_once_with(
            'custom-host', 5432, 'testuser', 'testpass', 'custom_db'
        )

    @patch('similarity_matrix.main.load_dotenv')
    @patch('similarity_matrix.main.Database')
    @patch('similarity_matrix.main.Pipeline')
    def test_main_all_subcommands(
        self, mock_pipeline_class, mock_database_class, mock_load_dotenv
    ):
        """Test main function with all different subcommands"""
        subcommands = [
            'update-db-row',
            'update-db-column',
            'update-db-matrix',
            'compute-similarity'
        ]

        for subcommand in subcommands:
            test_args = [
                'similarity_matrix',
                '--pipeline', 'test_pipeline',
                '--db-user', 'testuser',
                '--db-password', 'testpass',
                subcommand
            ]

            # Mock database
            mock_db = Mock()
            mock_db.test_connection.return_value = None
            mock_database_class.return_value = mock_db

            # Mock pipeline
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.update_db_row_table.return_value = "success"
            mock_pipeline_instance.update_db_column_table.return_value = "success"
            mock_pipeline_instance.update_db_matrix_table.return_value = "success"
            mock_pipeline_instance.get_matrix.return_value = "matrix"

            mock_available_pipelines = {
                'test_pipeline': Mock(
                    return_value=mock_pipeline_instance)}
            mock_pipeline_class.load_default_pipelines.return_value = mock_available_pipelines

            with patch('sys.argv', test_args):
                with pytest.raises(SystemExit) as exc_info:
                    main()

            assert exc_info.value.code == 0


class TestArgumentParsing:
    """Test argument parsing scenarios"""

    def test_version_argument(self):
        """Test --version argument"""
        with patch('sys.argv', ['similarity_matrix', '--version']):
            with pytest.raises(SystemExit) as exc_info:
                main()

            # argparse raises SystemExit with code 0 for --version
            assert exc_info.value.code == 0

    def test_help_argument(self):
        """Test --help argument"""
        with patch('sys.argv', ['similarity_matrix', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main()

            # argparse raises SystemExit with code 0 for --help
            assert exc_info.value.code == 0


@pytest.fixture
def mock_environment():
    """Fixture to provide a clean environment for tests"""
    with patch.dict(os.environ, {}, clear=True):
        yield


class TestIntegration:
    """Integration-style tests that test multiple components together"""

    @patch('similarity_matrix.main.load_dotenv')
    @patch('similarity_matrix.main.Database')
    @patch('similarity_matrix.main.Pipeline')
    def test_full_pipeline_execution_flow(
        self, mock_pipeline_class, mock_database_class, mock_load_dotenv
    ):
        """Test the complete flow from argument parsing to pipeline execution"""

        # Mock database
        mock_db = Mock()
        mock_db.test_connection.return_value = None
        mock_database_class.return_value = mock_db

        # Mock pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.update_db_row_table.return_value = "pipeline_success"
        mock_pipeline_constructor = Mock(return_value=mock_pipeline_instance)
        mock_available_pipelines = {'test_pipeline': mock_pipeline_constructor}
        mock_pipeline_class.load_default_pipelines.return_value = {}
        mock_pipeline_class.load_from_dir.return_value = mock_available_pipelines

        # Test arguments
        test_args = [
            'similarity_matrix',
            '--debug',
            '--db-host', 'test-host',
            '--db-port', '3307',
            '--db-name', 'test_db',
            '--db-user', 'test_user',
            '--db-password', 'test_pass',
            '--model-name', 'test-model',
            '--pipeline-dir', 'test/pipelines',
            '--output-dir', 'test/output',
            '--pipeline', 'test_pipeline',
            'update-db-row'
        ]

        with patch('os.path.isdir', return_value=True):
            with patch('sys.argv', test_args):
                with pytest.raises(SystemExit) as exc_info:
                    main()

                # Verify successful execution
                assert exc_info.value.code == 0

                # Verify all components were called correctly
                mock_load_dotenv.assert_called_once()
                mock_database_class.assert_called_once_with(
                    'test-host', 3307, 'test_user', 'test_pass', 'test_db'
                )
                mock_db.test_connection.assert_called_once()
                mock_pipeline_class.load_from_dir.assert_called_once_with(
                    'test/pipelines')
                mock_pipeline_constructor.assert_called_once_with(
                    name='test_pipeline',
                    db=mock_db,
                    path='test/output',
                    model_name='test-model'
                )
                mock_pipeline_instance.update_db_row_table.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
