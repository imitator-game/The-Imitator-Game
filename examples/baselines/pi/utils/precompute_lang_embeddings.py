"""
Precompute T5 Language Embeddings for RDT Training
This script generates and saves T5-XXL embeddings for all language descriptions
"""

import os
import json
import torch
from pathlib import Path
from tqdm import tqdm
import argparse

from examples.baselines.rdt.models.multimodal_encoder.t5_encoder import T5Embedder


def precompute_language_embeddings(
    lang_desc_path: str,
    output_dir: str,
    text_encoder: str = "google/t5-v1_1-xxl",
    max_length: int = 77,
    device: str = "cuda"
):
    """
    Precompute T5 embeddings for all language descriptions

    Args:
        lang_desc_path: Path to lang_descs.json
        output_dir: Directory to save precomputed embeddings
        text_encoder: T5 model name
        max_length: Maximum sequence length
        device: Device to use for encoding
    """

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load language descriptions
    print(f"Loading language descriptions from {lang_desc_path}")
    with open(lang_desc_path, 'r') as f:
        data = json.load(f)
    lang_descriptions = data['descriptions']

    print(f"Found {len(lang_descriptions)} tasks")

    # Initialize T5 encoder
    print(f"Initializing T5 encoder: {text_encoder}")
    t5_embedder = T5Embedder(
        from_pretrained=text_encoder,
        model_max_length=max_length,
        device=device
    )
    t5_embedder.model.eval()

    # Index for mapping text to file
    embedding_index = []

    # Process each task
    total_embeddings = 0
    with torch.no_grad():
        for task_name, videos in tqdm(lang_descriptions.items(), desc="Processing tasks"):
            # Create task directory
            task_dir = output_dir / task_name
            task_dir.mkdir(exist_ok=True)

            # Process each video
            for video_idx, video_data in enumerate(videos):
                # Process train descriptions
                if 'train' in video_data:
                    for desc_idx, text in enumerate(video_data['train']):
                        if not text or len(text.strip()) == 0:
                            continue

                        # Encode text
                        embeddings, mask = t5_embedder.get_text_embeddings([text])

                        # Save embedding
                        filename = f"v{video_idx}_train{desc_idx}.pt"
                        filepath = task_dir / filename

                        torch.save({
                            'embedding': embeddings[0].cpu(),
                            'mask': mask[0].cpu(),
                            'text': text
                        }, filepath)

                        # Add to index
                        embedding_index.append({
                            'text': text,
                            'task_name': task_name,
                            'video_idx': video_idx,
                            'desc_type': 'train',
                            'desc_idx': desc_idx,
                            'filename': filename
                        })

                        total_embeddings += 1

                # Process test descriptions
                if 'test' in video_data:
                    for desc_idx, text in enumerate(video_data['test']):
                        if not text or len(text.strip()) == 0:
                            continue

                        # Encode text
                        embeddings, mask = t5_embedder.get_text_embeddings([text])

                        # Save embedding
                        filename = f"v{video_idx}_test{desc_idx}.pt"
                        filepath = task_dir / filename

                        torch.save({
                            'embedding': embeddings[0].cpu(),
                            'mask': mask[0].cpu(),
                            'text': text
                        }, filepath)

                        # Add to index
                        embedding_index.append({
                            'text': text,
                            'task_name': task_name,
                            'video_idx': video_idx,
                            'desc_type': 'test',
                            'desc_idx': desc_idx,
                            'filename': filename
                        })

                        total_embeddings += 1

                # Process robot_prompt if exists
                if 'robot_prompt' in video_data and video_data['robot_prompt']:
                    text = video_data['robot_prompt']

                    # Encode text
                    embeddings, mask = t5_embedder.get_text_embeddings([text])

                    # Save embedding
                    filename = f"v{video_idx}_robot_prompt.pt"
                    filepath = task_dir / filename

                    torch.save({
                        'embedding': embeddings[0].cpu(),
                        'mask': mask[0].cpu(),
                        'text': text
                    }, filepath)

                    # Add to index
                    embedding_index.append({
                        'text': text,
                        'task_name': task_name,
                        'video_idx': video_idx,
                        'desc_type': 'robot_prompt',
                        'desc_idx': 0,
                        'filename': filename
                    })

                    total_embeddings += 1

    # Save index
    index_path = output_dir / "embedding_index.json"
    with open(index_path, 'w') as f:
        json.dump(embedding_index, f, indent=2)

    print(f"\n✓ Precomputed {total_embeddings} embeddings")
    print(f"✓ Saved to: {output_dir}")
    print(f"✓ Index saved to: {index_path}")

    # Print statistics
    print("\n=== Statistics ===")
    desc_types = {}
    for item in embedding_index:
        desc_type = item['desc_type']
        desc_types[desc_type] = desc_types.get(desc_type, 0) + 1

    for desc_type, count in desc_types.items():
        print(f"  {desc_type}: {count}")

    print(f"\nTotal tasks: {len(lang_descriptions)}")
    print(f"Total embeddings: {total_embeddings}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute T5 language embeddings")
    parser.add_argument(
        "--lang_desc_path",
        type=str,
        default="examples/baselines/imitators/processors/lang_descs.json",
        help="Path to language descriptions JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="precomputed_embeddings/lang",
        help="Output directory for precomputed embeddings"
    )
    parser.add_argument(
        "--text_encoder",
        type=str,
        default="google/t5-v1_1-xxl",
        help="T5 model name"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=77,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )

    args = parser.parse_args()

    precompute_language_embeddings(
        lang_desc_path=args.lang_desc_path,
        output_dir=args.output_dir,
        text_encoder=args.text_encoder,
        max_length=args.max_length,
        device=args.device
    )