
import pickle
from pathlib import Path
import sys

def count_total_chunks(export_dir_str: str):
    """
    Loads all worker output files and counts the total number of chunks.
    """
    try:
        export_dir = Path(export_dir_str)
        emb_dir = export_dir / "_index" / "embeddings"
        total_chunks = 0

        if not emb_dir.is_dir():
            print(f"Error: Embeddings directory not found at {emb_dir}")
            return

        pickle_files = sorted(emb_dir.glob("worker_*_batch_*.pkl"))

        if not pickle_files:
            print("No processed batch files found yet.")
            return

        for pkl_file in pickle_files:
            try:
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f)
                    # The 'chunks' key holds a list of the actual chunk dictionaries
                    num_chunks_in_batch = len(data.get("chunks", []))
            except Exception as e:
                print(f"Could not process file {pkl_file}: {e}", file=sys.stderr)

        print(f"Total chunks created so far: {total_chunks}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    # The path to the main export directory is passed as an argument
    if len(sys.argv) > 1:
        outlook_export_path = sys.argv
        count_total_chunks(outlook_export_path)
    else:
        print("Error: Please provide the path to the Outlook export directory.", file=sys.stderr)
