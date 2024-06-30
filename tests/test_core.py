import pytest
import sys
import os

# Add the root directory of the project to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mmlu.core import MMLU

def test_example_function(capfd):
    mmlu = MMLU()
    mmlu.example_function()
    captured = capfd.readouterr()
    assert captured.out == "This is an example function in the MMLU library.\n"

def test_read_file(tmpdir):
    mmlu = MMLU()
    test_file = tmpdir.join("test.txt")
    test_file.write("Hello, world!")
    content = mmlu.read_file(str(test_file))
    assert content == "Hello, world!"

def test_write_file(tmpdir):
    mmlu = MMLU()
    test_file = tmpdir.join("test.txt")
    mmlu.write_file(str(test_file), "Hello, world!")
    content = test_file.read()
    assert content == "Hello, world!"

def test_append_to_file(tmpdir):
    mmlu = MMLU()
    test_file = tmpdir.join("test.txt")
    test_file.write("Hello")
    mmlu.append_to_file(str(test_file), ", world!")
    content = test_file.read()
    assert content == "Hello, world!"

def test_search_files(tmpdir):
    mmlu = MMLU()
    test_dir = tmpdir.mkdir("test_dir")
    test_file1 = test_dir.join("file1.txt")
    test_file2 = test_dir.join("file2.txt")
    test_file3 = test_dir.join("another_file.txt")
    test_file1.write("Content of file1")
    test_file2.write("Content of file2")
    test_file3.write("Content of another file")
    matches = mmlu.search_files(str(test_dir), "*.txt")
    expected_matches = [str(test_file1), str(test_file2), str(test_file3)]
    assert sorted(matches) == sorted(expected_matches)
