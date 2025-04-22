import random
from torch.utils.data import Sampler

class ChunkedSampler(Sampler):
    """
    Splits the dataset into N chunks of chunk_size each and yields them
    in sequence. On each epoch, the entire dataset is shuffled, re-chunked,
    then yielded again.

    This version also stores the list of indices for each chunk in `self.epoch_chunks`,
    so you can retrieve them later (e.g., in your model).
    """
    def __init__(self, data_source, chunk_size=1000, shuffle=True):
        super().__init__(data_source)
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.shuffle = shuffle

        # For simplicity, require perfect divisibility
        assert len(self.data_source) % chunk_size == 0, \
            "Dataset size must be divisible by num_chunks."
        self.num_chunks = len(self.data_source) // chunk_size

        # Initialize with an empty list
        self.epoch_chunks = []
        
        # Pre-initialize chunks with a basic partition to avoid the first-access error
        self._initialize_chunks()
        
    def _initialize_chunks(self):
        """Pre-initialize chunks with a basic ordering"""
        indices = list(range(len(self.data_source)))
        if self.shuffle:
            random.shuffle(indices)
            
        self.epoch_chunks = []
        for chunk_idx in range(self.num_chunks):
            start = chunk_idx * self.chunk_size
            end = start + self.chunk_size
            self.epoch_chunks.append(indices[start:end])

    def __len__(self):
        """
        Return the total number of samples that will be yielded by this sampler.
        """
        return len(self.data_source)
    
    def __iter__(self):
        """
        Yield indices in chunks.
        """
        # Start with a fresh shuffling for this epoch
        self._initialize_chunks()
        
        # Yield all indices in order of chunks
        for chunk in self.epoch_chunks:
            for idx in chunk:
                yield idx

    def get_chunk_indices(self, chunk_id):
        """
        Return the list of indices belonging to the chunk_id-th chunk (0-based).
        """
        # If epoch_chunks is not initialized yet, initialize it
        if not self.epoch_chunks:
            self._initialize_chunks()
            
        return self.epoch_chunks[chunk_id]