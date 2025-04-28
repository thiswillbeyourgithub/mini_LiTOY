import pytest
import json

# import toml # Use rtoml instead # No longer needed
import rtoml  # Use rtoml for consistency
import uuid6
import copy
import shutil  # Add this import
from pathlib import Path
from mini_LiTOY.mini_LiTOY import mini_LiTOY, LockedDict, recovery_dir


# Fixtures
@pytest.fixture
def sample_entry_text():
    """Provides sample entry text."""
    return [f"Entry {i}" for i in range(10)]


@pytest.fixture
def sample_input_file(tmp_path, sample_entry_text):
    """Creates a temporary input file with sample entries."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("\n".join(sample_entry_text))
    return input_file


@pytest.fixture
def sample_output_json_file(tmp_path):
    """Provides the path for a temporary JSON output file."""
    return tmp_path / "output.json"


@pytest.fixture
def sample_output_toml_file(tmp_path):
    """Provides the path for a temporary TOML output file."""
    return tmp_path / "output.toml"


@pytest.fixture
def existing_output_json_file(tmp_path, sample_entry_text):
    """Creates a temporary existing JSON output file with some data."""
    output_file = tmp_path / "existing_output.json"
    data = []
    for i, text in enumerate(sample_entry_text[:5]):  # Only first 5 entries
        entry = copy.deepcopy(mini_LiTOY.default_dict)
        entry["entry"] = text
        entry["id"] = str(uuid6.uuid6())
        entry["g_ELO"] = mini_LiTOY.ELO_default + i  # Slightly different ELOs
        for q in mini_LiTOY.questions:
            entry["all_ELO"][q]["q_ELO"] = mini_LiTOY.ELO_default + i
        data.append(entry)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)
    return output_file


@pytest.fixture
def existing_output_toml_file(tmp_path, sample_entry_text):
    """Creates a temporary existing TOML output file with some data."""
    output_file = tmp_path / "existing_output.toml"
    data = []
    for i, text in enumerate(sample_entry_text[:5]):  # Only first 5 entries
        # Create plain dicts as they would be after loading/before dumping TOML
        entry = copy.deepcopy(mini_LiTOY.default_dict)
        entry["entry"] = text
        entry["id"] = str(uuid6.uuid6())
        entry["g_ELO"] = mini_LiTOY.ELO_default + i  # Slightly different ELOs
        for q in mini_LiTOY.questions:
            entry["all_ELO"][q]["q_ELO"] = mini_LiTOY.ELO_default + i
        # Convert LockedDicts within the structure to plain dicts for TOML saving
        entry_plain = {
            k: (v if not isinstance(v, LockedDict) else dict(v))
            for k, v in entry.items()
        }
        if "all_ELO" in entry_plain:
            entry_plain["all_ELO"] = {
                q: dict(elo_data) for q, elo_data in entry_plain["all_ELO"].items()
            }
        data.append(entry_plain)  # Append the plain dict

    # Wrap the list in a dictionary for TOML structure
    data_wrapper = {"entries": data}
    with open(output_file, "w") as f:
        rtoml.dump(data_wrapper, f, pretty=True)  # Use rtoml here
    return output_file


@pytest.fixture(autouse=True)
def clean_recovery_dir_fixture():
    """Cleans the recovery directory before and after each test function."""
    # Clean before test
    for item in recovery_dir.glob("*"):
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    yield  # Test runs here
    # Clean after test
    for item in recovery_dir.glob("*"):
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


# Tests for LockedDict
def test_locked_dict_init():
    """Test LockedDict initialization and initial locking."""
    d = LockedDict({"a": 1, "b": 2})
    assert d["a"] == 1
    with pytest.raises(KeyError):
        d["c"] = 3  # Cannot add new keys


def test_locked_dict_modify():
    """Test modifying existing keys in LockedDict."""
    d = LockedDict({"a": 1})
    d["a"] = 100
    assert d["a"] == 100


def test_locked_dict_deepcopy():
    """Test deepcopying LockedDict."""
    d1 = LockedDict({"a": 1, "nested": {"x": 10}})
    d2 = copy.deepcopy(d1)
    assert d1 == d2
    assert id(d1) != id(d2)
    assert id(d1["nested"]) != id(d2["nested"])
    d2["a"] = 2
    d2["nested"]["x"] = 20
    assert d1["a"] == 1
    assert d1["nested"]["x"] == 10
    # Check if the copied dict is also locked
    with pytest.raises(KeyError):
        d2["new_key"] = 5


# Tests for mini_LiTOY class
def test_init_no_files():
    """Test initialization fails with no input or output file."""
    with pytest.raises(
        ValueError, match="Either input_file or output_file must be provided"
    ):
        mini_LiTOY()


def test_init_only_input_file(sample_input_file):
    """Test initialization fails with only input file."""
    with pytest.raises(ValueError, match="output_file must be provided"):
        mini_LiTOY(input_file=sample_input_file)


def test_init_only_output_file_json(sample_output_json_file):
    """Test initialization with only an output JSON file (creates empty data)."""
    # Mock run_comparison_loop to prevent infinite loop
    mini_LiTOY.run_comparison_loop = lambda self: None
    instance = mini_LiTOY(output_file=sample_output_json_file)
    assert instance.output_file == sample_output_json_file
    assert instance.output_format == "json"
    assert instance.alldata == []
    assert instance.callback is None
    assert not sample_output_json_file.exists()  # Should not create file yet


def test_init_only_output_file_toml(sample_output_toml_file):
    """Test initialization with only an output TOML file (creates empty data)."""
    mini_LiTOY.run_comparison_loop = lambda self: None
    instance = mini_LiTOY(output_file=sample_output_toml_file)
    assert instance.output_file == sample_output_toml_file
    assert instance.output_format == "toml"
    assert instance.alldata == []


def test_init_with_input_and_output_json(
    sample_input_file, sample_output_json_file, sample_entry_text
):
    """Test initialization with input and new output JSON file."""
    mini_LiTOY.run_comparison_loop = lambda self: None
    instance = mini_LiTOY(
        input_file=sample_input_file, output_file=sample_output_json_file
    )
    assert len(instance.alldata) == len(sample_entry_text)
    assert instance.alldata[0]["entry"] == sample_entry_text[0]
    assert instance.alldata[-1]["entry"] == sample_entry_text[-1]
    assert instance.alldata[0]["id"] is not None
    assert instance.alldata[0]["g_ELO"] == mini_LiTOY.ELO_default
    assert not sample_output_json_file.exists()  # Should not create file yet


def test_init_with_input_and_output_toml(
    sample_input_file, sample_output_toml_file, sample_entry_text
):
    """Test initialization with input and new output TOML file."""
    mini_LiTOY.run_comparison_loop = lambda self: None
    instance = mini_LiTOY(
        input_file=sample_input_file, output_file=sample_output_toml_file
    )
    assert len(instance.alldata) == len(sample_entry_text)
    assert instance.output_format == "toml"


def test_init_loading_existing_json(existing_output_json_file, sample_entry_text):
    """Test initialization loading data from an existing JSON file."""
    mini_LiTOY.run_comparison_loop = lambda self: None
    instance = mini_LiTOY(output_file=existing_output_json_file)
    assert len(instance.alldata) == 5  # Only 5 entries were in the existing file
    assert instance.alldata[0]["entry"] == sample_entry_text[0]
    assert instance.alldata[0]["g_ELO"] == mini_LiTOY.ELO_default + 0
    assert instance.alldata[4]["entry"] == sample_entry_text[4]
    assert instance.alldata[4]["g_ELO"] == mini_LiTOY.ELO_default + 4
    # Check if loaded entries are LockedDicts
    assert isinstance(instance.alldata[0], LockedDict)
    assert isinstance(
        instance.alldata[0]["all_ELO"][mini_LiTOY.questions[0]], LockedDict
    )


def test_init_loading_existing_toml(existing_output_toml_file, sample_entry_text):
    """Test initialization loading data from an existing TOML file."""
    mini_LiTOY.run_comparison_loop = lambda self: None
    instance = mini_LiTOY(output_file=existing_output_toml_file)
    assert len(instance.alldata) == 5  # Only 5 entries were in the existing file
    assert instance.output_format == "toml"
    assert instance.alldata[0]["entry"] == sample_entry_text[0]
    assert instance.alldata[0]["g_ELO"] == mini_LiTOY.ELO_default + 0
    assert instance.alldata[4]["entry"] == sample_entry_text[4]
    assert instance.alldata[4]["g_ELO"] == mini_LiTOY.ELO_default + 4
    # Check if loaded entries are LockedDicts
    assert isinstance(instance.alldata[0], LockedDict)
    assert isinstance(
        instance.alldata[0]["all_ELO"][mini_LiTOY.questions[0]], LockedDict
    )


def test_init_loading_existing_and_adding_new_json(
    existing_output_json_file, sample_input_file, sample_entry_text
):
    """Test initialization loading existing JSON and adding new entries from input."""
    mini_LiTOY.run_comparison_loop = lambda self: None
    instance = mini_LiTOY(
        input_file=sample_input_file, output_file=existing_output_json_file
    )
    assert len(instance.alldata) == len(
        sample_entry_text
    )  # All 10 entries should be present
    # Check existing entries are loaded correctly
    assert instance.alldata[0]["entry"] == sample_entry_text[0]
    assert instance.alldata[0]["g_ELO"] == mini_LiTOY.ELO_default + 0
    # Check new entries are added correctly
    assert instance.alldata[5]["entry"] == sample_entry_text[5]
    assert instance.alldata[5]["g_ELO"] == mini_LiTOY.ELO_default
    assert instance.alldata[9]["entry"] == sample_entry_text[9]
    assert instance.alldata[9]["g_ELO"] == mini_LiTOY.ELO_default


def test_init_loading_existing_and_adding_new_toml(
    existing_output_toml_file, sample_input_file, sample_entry_text
):
    """Test initialization loading existing TOML and adding new entries from input."""
    mini_LiTOY.run_comparison_loop = lambda self: None
    instance = mini_LiTOY(
        input_file=sample_input_file, output_file=existing_output_toml_file
    )
    assert len(instance.alldata) == len(
        sample_entry_text
    )  # All 10 entries should be present
    assert instance.output_format == "toml"
    # Check existing entries are loaded correctly
    assert instance.alldata[0]["entry"] == sample_entry_text[0]
    assert instance.alldata[0]["g_ELO"] == mini_LiTOY.ELO_default + 0
    # Check new entries are added correctly
    assert instance.alldata[5]["entry"] == sample_entry_text[5]
    assert instance.alldata[5]["g_ELO"] == mini_LiTOY.ELO_default
    assert instance.alldata[9]["entry"] == sample_entry_text[9]
    assert instance.alldata[9]["g_ELO"] == mini_LiTOY.ELO_default
    # Check types after loading and merging
    assert isinstance(instance.alldata[0], LockedDict)
    assert isinstance(instance.alldata[5], LockedDict)


def test_update_elo():
    """Test the ELO update calculation."""
    # Use class attribute directly
    # Create a dummy instance to access constants and call the method
    # Although update_elo doesn't use self, calling it via an instance is cleaner
    dummy_instance = mini_LiTOY.__new__(
        mini_LiTOY
    )  # Create instance without calling __init__
    elo_norm = dummy_instance.ELO_norm
    elo1, elo2 = 100, 100
    k1, k2 = 30, 30  # Initial K values

    # Test case 1: Player 1 wins decisively (answer=1 -> score1=(5-1)/5=0.8, score2=0.2)
    new_elo1, new_elo2 = dummy_instance.update_elo(1, elo1, elo2, k1, k2)
    expected_score1 = 1 / (1 + 10 ** ((elo2 - elo1) / elo_norm))  # 0.5
    expected_score2 = 1 - expected_score1  # 0.5
    assert new_elo1 == round(
        elo1 + k1 * (0.8 - expected_score1)
    )  # 100 + 30 * 0.3 = 109
    assert new_elo2 == round(
        elo2 + k2 * (0.2 - expected_score2)
    )  # 100 + 30 * -0.3 = 91

    # Test case 2: Player 2 wins decisively (answer=5 -> score1=(5-5)/5=0.0, score2=1.0)
    new_elo1, new_elo2 = dummy_instance.update_elo(5, elo1, elo2, k1, k2)
    expected_score1 = 1 / (1 + 10 ** ((elo2 - elo1) / elo_norm))  # 0.5
    expected_score2 = 1 - expected_score1  # 0.5
    assert new_elo1 == round(
        elo1 + k1 * (0.0 - expected_score1)
    )  # 100 + 30 * -0.5 = 85
    assert new_elo2 == round(
        elo2 + k2 * (1.0 - expected_score2)
    )  # 100 + 30 * 0.5 = 115

    # Test case 3: Draw (answer=3 -> score1=(5-3)/5=0.4, score2=0.6) - Note: ELO uses 5 steps, 3 is slightly favoring player 2
    new_elo1, new_elo2 = dummy_instance.update_elo(3, elo1, elo2, k1, k2)
    expected_score1 = 1 / (1 + 10 ** ((elo2 - elo1) / elo_norm))  # 0.5
    expected_score2 = 1 - expected_score1  # 0.5
    assert new_elo1 == round(
        elo1 + k1 * (0.4 - expected_score1)
    )  # 100 + 30 * -0.1 = 97
    assert new_elo2 == round(
        elo2 + k2 * (0.6 - expected_score2)
    )  # 100 + 30 * 0.1 = 103

    # Test case 4: Higher ELO player wins as expected
    elo1, elo2 = 120, 80
    k1, k2 = 20, 20  # Assume K decreased
    expected_score1 = 1 / (
        1 + 10 ** ((elo2 - elo1) / elo_norm)
    )  # 1 / (1 + 10**(-40 / 40)) = 1 / (1 + 10**(-1)) = 1 / 1.1 = ~0.909
    expected_score2 = 1 - expected_score1  # ~0.091
    # Player 1 wins (answer=2 -> score1=(5-2)/5=0.6, score2=0.4) - Less gain as it was expected
    new_elo1, new_elo2 = dummy_instance.update_elo(2, elo1, elo2, k1, k2)
    assert new_elo1 == round(
        elo1 + k1 * (0.6 - expected_score1)
    )  # 120 + 20 * (0.6 - 0.909) = 120 + 20 * ~-0.309 = ~114
    assert new_elo2 == round(
        elo2 + k2 * (0.4 - expected_score2)
    )  # 80 + 20 * (0.4 - 0.091) = 80 + 20 * ~0.309 = ~86

    # Test case 5: Lower ELO player wins (answer=4 -> score1=(5-4)/5=0.2, score2=0.8) - More gain
    new_elo1, new_elo2 = dummy_instance.update_elo(4, elo1, elo2, k1, k2)
    expected_score1 = 1 / (1 + 10 ** ((elo2 - elo1) / elo_norm))  # ~0.909
    expected_score2 = 1 - expected_score1  # ~0.091
    assert new_elo1 == round(
        elo1 + k1 * (0.2 - expected_score1)
    )  # 120 + 20 * (0.2 - 0.909) = 120 + 20 * ~-0.709 = ~106
    assert new_elo2 == round(
        elo2 + k2 * (0.8 - expected_score2)
    )  # 80 + 20 * (0.8 - 0.091) = 80 + 20 * ~0.709 = ~94


def test_store_data_json(tmp_path):
    """Test storing data to a JSON file."""
    mini_LiTOY.run_comparison_loop = lambda self: None  # Prevent loop
    output_file = tmp_path / "store_test.json"
    instance = mini_LiTOY(output_file=output_file)  # Start with empty data

    # Manually add some data
    entry1 = copy.deepcopy(mini_LiTOY.default_dict)
    entry1["entry"] = "Test Entry 1"
    entry1["id"] = str(uuid6.uuid6())
    entry2 = copy.deepcopy(mini_LiTOY.default_dict)
    entry2["entry"] = "Test Entry 2"
    entry2["id"] = str(uuid6.uuid6())
    instance.alldata = [entry1, entry2]

    instance.store_data()

    assert output_file.exists()
    with open(output_file, "r") as f:
        loaded_data = json.load(f)
    assert len(loaded_data) == 2
    assert loaded_data[0]["entry"] == "Test Entry 1"
    assert loaded_data[1]["entry"] == "Test Entry 2"
    assert loaded_data[0]["id"] == entry1["id"]
    # Check recovery file also exists and contains the saved data
    recovery_files = list(recovery_dir.glob("*"))
    assert len(recovery_files) == 1
    with open(recovery_files[0], "r") as f:
        recovery_data = json.load(f)
    assert recovery_data == loaded_data


def test_store_data_toml(tmp_path):
    """Test storing data to a TOML file."""
    mini_LiTOY.run_comparison_loop = lambda self: None  # Prevent loop
    output_file = tmp_path / "store_test.toml"
    instance = mini_LiTOY(output_file=output_file)  # Start with empty data

    # Manually add some data
    entry1 = copy.deepcopy(mini_LiTOY.default_dict)
    entry1["entry"] = "Test Entry 1 TOML"
    entry1["id"] = str(uuid6.uuid6())
    instance.alldata = [entry1]

    instance.store_data()

    assert output_file.exists()
    with open(output_file, "r") as f:
        # Load the wrapper dict and extract the list
        loaded_data_wrapper = rtoml.load(f)  # Use rtoml here
    assert "entries" in loaded_data_wrapper
    loaded_data = loaded_data_wrapper["entries"]
    assert isinstance(loaded_data, list)
    assert len(loaded_data) == 1
    assert loaded_data[0]["entry"] == "Test Entry 1 TOML"
    assert loaded_data[0]["id"] == entry1["id"]
    # Check recovery file also exists and contains the saved data
    recovery_files = list(recovery_dir.glob("*"))
    assert len(recovery_files) == 1
    with open(recovery_files[0], "r") as f:
        # Use rtoml to load recovery file as well, expecting the wrapper
        recovery_data_wrapper = rtoml.load(f)  # Use rtoml here
    assert "entries" in recovery_data_wrapper
    recovery_data = recovery_data_wrapper["entries"]
    assert isinstance(recovery_data, list)
    # Convert loaded TOML data (which will be plain dicts) back to LockedDicts
    # for comparison if necessary, or compare structure/values directly.
    # For simplicity, let's compare the plain dict versions.
    # The main code converts to plain dicts before dumping to TOML,
    # so loaded_data and recovery_data should both be lists of plain dicts.
    assert recovery_data == loaded_data


# Test pick_two_entries
from unittest.mock import patch


def test_pick_two_entries(tmp_path):
    """Test the logic for picking two entries, favoring lower n_comparison."""
    mini_LiTOY.run_comparison_loop = lambda self: None  # Prevent loop
    output_file = tmp_path / "pick_test.json"
    instance = mini_LiTOY(output_file=output_file)

    # Create 5 dummy entries with varying n_comparison
    entries = []
    for i in range(5):
        entry = copy.deepcopy(mini_LiTOY.default_dict)
        entry["entry"] = f"Entry {i}"
        entry["id"] = str(uuid6.uuid6())
        entry["g_n_comparison"] = i * 2  # 0, 2, 4, 6, 8
        entries.append(entry)
    instance.alldata = entries

    # Mock random.sample to return predictable indices
    # Scenario 1: entry2 (idx 1, n=2) vs entry3 (idx 2, n=4) -> pick entry2
    with patch("random.sample", return_value=[0, 1, 2]) as mock_sample:
        picked1, picked2 = instance.pick_two_entries()
        mock_sample.assert_called_once_with(range(5), 3)
        assert picked1 == entries[0]  # First sampled is always picked1
        assert picked2 == entries[1]  # Lower n_comparison between entry[1] and entry[2]

    # Scenario 2: entry2 (idx 3, n=6) vs entry3 (idx 1, n=2) -> pick entry3
    with patch("random.sample", return_value=[0, 3, 1]) as mock_sample:
        picked1, picked2 = instance.pick_two_entries()
        mock_sample.assert_called_once_with(range(5), 3)
        assert picked1 == entries[0]  # First sampled is always picked1
        assert picked2 == entries[1]  # Lower n_comparison between entry[3] and entry[1]

    # Scenario 3: Test with less than 5 entries raises ValueError
    instance.alldata = entries[:4]
    with pytest.raises(ValueError, match="You need at least 5 entries"):
        instance.pick_two_entries()


# TODO: Add tests for run_comparison_loop (needs mocking prompt_toolkit and potentially callback)
# TODO: Add tests for handling invalid data in output files
# TODO: Add tests for callback functionality
