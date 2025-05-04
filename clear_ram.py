import gc

def clear_ram():
    # Force garbage collection
    gc.collect()
    print("RAM cleared.")

if __name__ == "__main__":
    clear_ram()
