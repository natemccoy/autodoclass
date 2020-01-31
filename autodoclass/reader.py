import os


class Reader:
    def __init__(self, folder):
        self.folder = folder

    def yield_paths(self):
        """ Return a list of absolute paths for files in the folder.

        :yield str: absolute file paths
        """
        for filename in os.listdir(self.folder):
            yield os.path.join(self.folder, filename)


class DocumentReader(Reader):
    def yield_texts(self):
        """ Yield the texts on the document level
        from the folder.

        :yield str:
        """
        for path in self.yield_paths():
            with open(path) as f:
                yield f.read()


class LineReader(Reader):
    def yield_texts(self):
        """ Yield the texts on the line level
        from the folder. Strips line endings only.

        :yield str:
        """
        for path in self.yield_paths():
            with open(path) as f:
                for line in f:
                    yield line.rstrip('\n').rstrip('\r')
