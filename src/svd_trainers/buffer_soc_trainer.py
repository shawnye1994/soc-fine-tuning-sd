import torch
import gc
from svd_trainers.soc_trainer import SOCTrainer

class BufferSOCTrainer(SOCTrainer):
    """
    Buffer-based Stochastic Optimal Control Trainer for diffusion models.
    
    This class extends SOCTrainer with an efficient memory buffer mechanism that:
    - Pre-computes and caches training trajectories in fixed-size chunks
    - Supports multiple passes through each buffer before recomputation
    - Implements memory-efficient reshuffling to increase data diversity
    - Distributes buffer computation across multiple GPUs
    
    The buffer approach significantly improves training efficiency by:
    1. Reducing redundant trajectory computation
    2. Optimizing GPU memory utilization
    3. Balancing compute time between trajectory generation and model updates
    4. Supporting larger effective batch sizes through accumulation
    
    Key parameters:
    - buffer_size: Total number of examples in the buffer
    - passes_per_buffer: Number of training passes before buffer refresh
    - buffer_device: Which device to store the buffer on
    """
    def __init__(self, config):
        super().__init__(config)
        self.buffer_size = config.buffer_size
        self.buffer_device = config.buffer_device
        self.iterations_per_chunk = self.buffer_size // (torch.cuda.device_count() * config.batch_size)
        self.passes_per_buffer = config.passes_per_buffer

        # Calculate how often to update the buffer
        self.buffer_update_frequency = self.iterations_per_chunk * config.passes_per_buffer

        self.buffer_initialized = False
        self.current_pass = 0  # Track which pass through the buffer we're on

    def on_fit_start(self):
        super().on_fit_start()
        # Initialize the buffer at the start of training
        if not self.buffer_initialized:
            self.recompute_buffer(1)  # Fill with first chunk
                # offload the vae to cpu
        print('offloading vae to cpu')
        self.soc_pipeline.vae.encoder.to('cpu')
        self.soc_pipeline.vae.decoder.to('cpu')
        torch.cuda.empty_cache()
        gc.collect()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # If we just finished all passes through a buffer:
        if (batch_idx + 1) % self.buffer_update_frequency == 0:
            # num_chunks = len(data_source) // chunk_size, where chunk_size = buffer_size
            current_chunk = ((batch_idx + 1) // self.buffer_update_frequency) + 1
            self.recompute_buffer(current_chunk)
            self.current_pass = 0
        # Or if we finished a pass but not all passes:
        elif (batch_idx + 1) % self.iterations_per_chunk == 0:
            self.current_pass += 1
            self.reshuffle_buffer()
            # Print the current pass number
            print(f"Buffer reshuffled. Starting pass {self.current_pass + 1}/{self.passes_per_buffer} through the current buffer")

    def recompute_buffer(self, chunk_id):
        """The whole train dataset is divided into num_chunks = len(data_source) // buffer_size
        For example, len(train_source) = 8, buffer_size = 4, then num_chunks = 2
        epoch_chunks would be some random exmaple indices like [[4,1,5,2], [0,3,6,7]] (two chunks, each chunk has buffer_size examples)
        get_chunk_indices(zero_based_chunk) will fetch a list of examples from epoch_chunks, like [4, 1, 5, 2], as the local_indices for all GPUs to train (the current buffer)
        """
        if next(self.soc_pipeline.vae.encoder.parameters()).device == torch.device('cpu'):
            print('move vae to gpu')
            self.soc_pipeline.vae.encoder.to(self.device)
            self.soc_pipeline.vae.decoder.to(self.device)
            
        # chunk_id is 1-based in our quick calculation above
        # but our sampler's chunk indexing is 0-based
        zero_based_chunk = chunk_id - 1

        # Access the DataModule
        datamodule = self.trainer.datamodule
        sampler = datamodule.sampler

        # Retrieve all indices for this chunk
        chunk_indices = sampler.get_chunk_indices(zero_based_chunk)

        # Get local rank (GPU ID) and world size (total GPUs)
        local_rank = self.global_rank
        world_size = torch.cuda.device_count()
        print(f'len(chunk_indices): {len(chunk_indices)}, world_size: {world_size}, local_rank: {local_rank}')
        
        # Distribute indices across GPUs
        per_gpu_count = len(chunk_indices) // world_size
        start_idx = local_rank * per_gpu_count
        end_idx = start_idx + per_gpu_count if local_rank < world_size - 1 else len(chunk_indices)

        # Get this GPU's subset of indices
        local_indices = chunk_indices[start_idx:end_idx]
        
        # Define the batch size for processing
        processing_batch_size = self.config.batch_size
        
        # Initialize accumulators for each GPU's results
        self.accumulator = {}
        for var in self.buffer_variables:
            self.accumulator[var] = []
        
        # Process in batches to avoid memory issues
        for batch_start in range(0, len(local_indices), processing_batch_size):
            batch_end = min(batch_start + processing_batch_size, len(local_indices))
            batch_indices = local_indices[batch_start:batch_end]
            
            # Get corresponding prompts for this batch
            batch_vid_image = [datamodule.train_dataset[i] for i in batch_indices] # a list of a tuple (video_tensor, PIL_Image(init_frame))
            print(f"GPU {local_rank}: Processing batch {batch_start//processing_batch_size + 1}/{(len(local_indices)-1)//processing_batch_size + 1} for chunk {chunk_id}")
            print(f'len(local_indices): {len(local_indices)}, processing_batch_size: {processing_batch_size}')
            
            # Process this batch
            data_dict = {'gt_video': torch.stack([p[0] for p in batch_vid_image]), 'init_frame': torch.stack([p[1] for p in batch_vid_image])}
            # trajectories, adjoints, rewards, random_noises, noise_preds, noise_preds_init, prompt_embeds, negative_prompt_embeds = self.collect_data(prompts_dict, batch_start)
            collected_data = self.collect_data(data_dict['init_frame'], batch_start)

            # move the collected data to the buffer device if buffer device is cpu
            for var in self.buffer_variables:
                data = collected_data[var]
                if isinstance(data, torch.Tensor) and self.buffer_device == 'cpu':
                    collected_data[var] = data.to(self.buffer_device, non_blocking=True).pin_memory()
            
            # Accumulate results
            for var in self.buffer_variables:
                self.accumulator[var].append(collected_data[var])
            
            # Free memory
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        
        # Store lists directly in buffer - no concatenation needed
        for var in self.buffer_variables:
            self.buffer[var] = self.accumulator[var]
        
        self.buffer_initialized = True
        print(f"GPU {local_rank}: Buffer updated for chunk {chunk_id} with {len(self.accumulator['image_latents'])} videos in {len(self.accumulator['trajectories'])} batches")
        
        # Minimal synchronization to ensure all GPUs have finished
        torch.distributed.barrier()

    def reshuffle_buffer(self):
        """
        Reshuffles the trajectories and corresponding data in the buffer
        to create new training batches while maintaining data alignment.
        """
        if not self.buffer_initialized:
            print("Buffer not initialized yet, nothing to reshuffle")
            return
        
        # Count the number of batches in the buffer
        num_batches = len(self.buffer['trajectories'])
        if num_batches <= 1:
            print("Only one batch in buffer, no reshuffling needed")
            return
        
        # Create a shuffled permutation of batch indices
        # Use the same seed across all GPUs to ensure consistent shuffling
        device = self.buffer['trajectories'][0].device
        seed = seed = self.config.seed + self.global_step * 100 + self.current_pass * 10000
        
        # Set the RNG state for consistent shuffling across GPUs
        orig_rng_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        
        # Generate permutation indices
        perm_indices = torch.randperm(num_batches)
        
        # Restore original RNG state
        torch.random.set_rng_state(orig_rng_state)
        
        # Perform in-place shuffling for each variable
        for var in self.buffer_variables:
            if self.buffer[var] is not None:
                # Create a reference copy of the original list
                original = self.buffer[var].copy()  # This is a shallow copy of the list, not the tensors
                
                # Shuffle in-place
                for i, idx in enumerate(perm_indices):
                    self.buffer[var][i] = original[idx]
        
        # Force garbage collection to clean up references
        del original
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"GPU {self.global_rank}: Buffer reshuffled with {num_batches} batches (memory-efficient)")
        
        # Minimal synchronization to ensure all GPUs have finished
        torch.distributed.barrier()