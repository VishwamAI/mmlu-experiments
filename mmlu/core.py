class MMLU:
    def __init__(self):
        pass

    def example_function(self):
        """
        This is an example function in the MMLU library.

        It prints a message to the console.
        """
        print("This is an example function in the MMLU library.")

    def read_file(self, file_path):
        """
        Reads the contents of a file.

        Parameters:
        file_path (str): The path to the file to be read.

        Returns:
        str: The contents of the file.
        """
        with open(file_path, 'r') as file:
            return file.read()

    def write_file(self, file_path, content):
        """
        Writes content to a file.

        Parameters:
        file_path (str): The path to the file to be written.
        content (str): The content to write to the file.
        """
        with open(file_path, 'w') as file:
            file.write(content)

    def append_to_file(self, file_path, content):
        """
        Appends content to the end of a file.

        Parameters:
        file_path (str): The path to the file to be appended.
        content (str): The content to append to the file.
        """
        with open(file_path, 'a') as file:
            file.write(content)

    def search_files(self, directory, pattern):
        """
        Searches for files in a directory that match a given pattern.

        Parameters:
        directory (str): The directory to search in.
        pattern (str): The pattern to match files against.

        Returns:
        list: A list of file paths that match the pattern.
        """
        import os
        import fnmatch

        matches = []
        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, pattern):
                matches.append(os.path.join(root, filename))
        return matches
