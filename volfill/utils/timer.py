import torch


class CUDATimer:
    def __init__(self, name="Block", enabled=True, num_frames=None, measure_memory=False):
        self.name = name
        self.enabled = enabled
        self.num_frames = num_frames
        self.measure_memory = measure_memory
        # Results will be populated after the context exits
        self.elapsed_time_ms = None
        self.avg_time_ms = None
        self.memory_allocated_mb = None
        self.memory_reserved_mb = None
        self.memory_peak_mb = None

        if self.enabled:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if self.enabled:
            torch.cuda.synchronize()  # Make sure everything before is done
            self.start_event.record()
        
            if self.measure_memory:
                torch.cuda.reset_peak_memory_stats()
                self.start_memory_allocated = torch.cuda.memory_allocated()
                self.start_memory_reserved = torch.cuda.memory_reserved()
        
        # Returning self allows callers to access measured times after the block
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.enabled:
            self.end_event.record()
            torch.cuda.synchronize()  # Wait for everything to finish

            if self.measure_memory:
                end_memory_allocated = torch.cuda.memory_allocated()
                end_memory_reserved = torch.cuda.memory_reserved()
                peak_memory = torch.cuda.max_memory_allocated()

            elapsed_time_ms = self.start_event.elapsed_time(self.end_event)
            self.elapsed_time_ms = elapsed_time_ms
            print(f"{self.name}: {elapsed_time_ms:.3f} ms")

            if self.num_frames is not None:
                avg_time_ms = elapsed_time_ms / self.num_frames
                self.avg_time_ms = avg_time_ms
                print(f"{self.name}: {avg_time_ms:.3f} ms per frame")
        
            if self.measure_memory:
                self.memory_allocated_mb = (end_memory_allocated - self.start_memory_allocated) / (1024 ** 2)
                self.memory_reserved_mb = (end_memory_reserved - self.start_memory_reserved) / (1024 ** 2)
                self.memory_peak_mb = peak_memory / (1024 ** 2)
                
                print(f"{self.name} Memory - Allocated: {self.memory_allocated_mb:+.2f} MB, "
                    f"Reserved: {self.memory_reserved_mb:+.2f} MB, Peak: {self.memory_peak_mb:.2f} MB")