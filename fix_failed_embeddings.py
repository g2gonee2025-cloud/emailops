#!/usr/bin/env python3
"""Re-embed chunks that have zero vectors from failed batches"""
import pickle
import numpy as np
import json
from pathlib import Path
import sys

# Add emailops to path
sys.path.insert(0, str(Path(__file__).parent))

from emailops.llm_client import embed_texts

def find_and_fix_failed_batches(embeddings_dir: Path, provider: str = "vertex"):
    """Find batches with zero vectors and re-embed them"""
    pickle_files = list(embeddings_dir.glob('*.pkl'))
    
    print(f"Scanning {len(pickle_files)} pickle files for zero vectors...")
    
    total_fixed = 0
    
    for pkl_file in pickle_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            embs = np.array(data['embeddings'], dtype='float32')
            chunks = data['chunks']
            
            # Find rows that are all zeros
            zero_mask = np.all(embs == 0, axis=1)
            num_zeros = int(np.sum(zero_mask))
            
            if num_zeros > 0:
                print(f"\n{'='*70}")
                print(f"Found {num_zeros} zero vectors in {pkl_file.name}")
                print(f"{'='*70}")
                
                # Extract texts for failed chunks
                failed_indices = np.where(zero_mask)[0]
                texts_to_embed = [chunks[i]['text'] for i in failed_indices]
                
                print(f"Re-embedding {len(texts_to_embed)} chunks...")
                
                # Re-embed in one batch
                try:
                    new_embeddings = embed_texts(texts_to_embed, provider=provider)
                    new_embs_array = np.array(new_embeddings, dtype='float32')
                    
                    # Replace zero vectors with new embeddings
                    for idx, failed_idx in enumerate(failed_indices):
                        embs[failed_idx] = new_embs_array[idx]
                    
                    # Save updated pickle
                    data['embeddings'] = embs
                    with open(pkl_file, 'wb') as f:
                        pickle.dump(data, f)
                    
                    print(f"✅ Successfully re-embedded {num_zeros} chunks")
                    total_fixed += num_zeros
                    
                except Exception as e:
                    print(f"❌ Failed to re-embed: {e}")
                    
        except Exception as e:
            print(f"Error processing {pkl_file.name}: {e}")
    
    print(f"\n{'='*70}")
    print(f"SUMMARY: Fixed {total_fixed} zero vectors across {len([1 for f in pickle_files if num_zeros > 0])} files")
    print(f"{'='*70}")
    
    return total_fixed

if __name__ == "__main__":
    embeddings_dir = Path('C:/Users/ASUS/Desktop/Outlook/_index/embeddings')
    fixed = find_and_fix_failed_batches(embeddings_dir)
    sys.exit(0 if fixed > 0 else 1)
