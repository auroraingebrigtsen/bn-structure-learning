import tracemalloc

class MemoryTracer:
    def __init__(self):
        self.current_memory = 0
        self.peak_memory = 0

    def start(self):
        tracemalloc.start()

    def stop(self):
        self.current_memory, self.peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self._print_usage()

    def _print_usage(self):
        print(f"\nCurrent memory usage: {self.current_memory / 1_048_576:.2f} MB")
        print(f"Peak memory usage: {self.peak_memory / 1_048_576:.2f} MB")
