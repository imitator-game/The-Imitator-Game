"""
Precomputed Language Embedding Loader
Loads and manages precomputed T5 embeddings for RDT training/evaluation
"""

import os
import json
import torch
import numpy as np
from typing import Dict, Optional, List, Tuple
from pathlib import Path


class PrecomputedLangEmbedding:
    """Manages precomputed language embeddings for RDT"""

    def __init__(self, precomputed_dir: str, device: str = "cuda"):
        self.precomputed_dir = Path(precomputed_dir)
        self.device = device
        self.embeddings_cache = {}
        self.text_to_embedding = {}

        # Load mapping index
        self.index_file = self.precomputed_dir / "embedding_index.json"
        if not self.index_file.exists():
            raise FileNotFoundError(f"Embedding index not found: {self.index_file}")

        with open(self.index_file, 'r') as f:
            self.embedding_index = json.load(f)

        print(f"Loaded precomputed embeddings index with {len(self.embedding_index)} entries")
        self._build_text_mapping()

    def _build_text_mapping(self):
        """Build mapping from text to embedding file path"""
        for mapping in self.embedding_index:
            text = mapping['text']
            task_name = mapping['task_name']
            video_idx = mapping['video_idx']
            desc_type = mapping['desc_type']
            desc_idx = mapping['desc_idx']

            # Create file path
            filename = f"v{video_idx}_{desc_type}{desc_idx}.pt"
            filepath = self.precomputed_dir / task_name / filename

            # Store mapping
            self.text_to_embedding[text] = {
                'filepath': filepath,
                'task_name': task_name,
                'video_idx': video_idx,
                'desc_type': desc_type,
                'desc_idx': desc_idx
            }

    def get_embedding(self, text: str, return_mask: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get embedding for given text

        Args:
            text: Text to get embedding for
            return_mask: Whether to return attention mask

        Returns:
            Tuple of (embedding, mask) if return_mask=True, else just embedding
        """
        # Check cache first
        cache_key = text
        if cache_key in self.embeddings_cache:
            embedding_data = self.embeddings_cache[cache_key]
            if return_mask:
                return embedding_data['embedding'], embedding_data['mask']
            else:
                return embedding_data['embedding']

        # Check if we have precomputed embedding for this text
        if text in self.text_to_embedding:
            mapping = self.text_to_embedding[text]
            filepath = mapping['filepath']

            if filepath.exists():
                # Load precomputed embedding
                data = torch.load(filepath, map_location=self.device)
                embedding = data['embedding'].to(self.device)

                # Create attention mask (all tokens are valid)
                seq_len = embedding.shape[0]
                mask = torch.ones(seq_len, dtype=torch.bool, device=self.device)

                # Cache the result
                self.embeddings_cache[cache_key] = {
                    'embedding': embedding,
                    'mask': mask
                }

                if return_mask:
                    return embedding, mask
                else:
                    return embedding

        # If no precomputed embedding found, return None
        # The calling code should handle this by falling back to real-time encoding
        return None, None if return_mask else None

    def get_embedding_by_task_and_type(
        self,
        task_name: str,
        desc_type: str = "train",
        video_idx: int = 0,
        desc_idx: int = 0,
        return_mask: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get embedding by task name and description type

        Args:
            task_name: Task name (e.g., "human_pick_red_cube_place_plate")
            desc_type: Description type ("train" or "test")
            video_idx: Video index
            desc_idx: Description index within video
            return_mask: Whether to return attention mask

        Returns:
            Tuple of (embedding, mask) if return_mask=True, else just embedding
        """
        # Create filename
        filename = f"v{video_idx}_{desc_type}{desc_idx}.pt"
        filepath = self.precomputed_dir / task_name / filename

        # Check cache first
        cache_key = str(filepath)
        if cache_key in self.embeddings_cache:
            embedding_data = self.embeddings_cache[cache_key]
            if return_mask:
                return embedding_data['embedding'], embedding_data['mask']
            else:
                return embedding_data['embedding']

        if filepath.exists():
            # Load precomputed embedding
            data = torch.load(filepath, map_location=self.device)
            embedding = data['embedding'].to(self.device)

            # Create attention mask (all tokens are valid)
            seq_len = embedding.shape[0]
            mask = torch.ones(seq_len, dtype=torch.bool, device=self.device)

            # Cache the result
            self.embeddings_cache[cache_key] = {
                'embedding': embedding,
                'mask': mask
            }

            if return_mask:
                return embedding, mask
            else:
                return embedding
        else:
            print(f"Warning: Precomputed embedding not found: {filepath}")
            return None, None if return_mask else None

    def get_batch_embeddings(
        self,
        texts: List[str],
        max_length: int = 77,
        return_mask: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get embeddings for a batch of texts

        Args:
            texts: List of texts to get embeddings for
            max_length: Maximum sequence length for padding
            return_mask: Whether to return attention masks

        Returns:
            Tuple of (batch_embeddings, batch_masks) if return_mask=True, else just embeddings
            batch_embeddings: [B, max_length, hidden_dim]
            batch_masks: [B, max_length] if return_mask=True
        """
        batch_embeddings = []
        batch_masks = []

        for text in texts:
            embedding, mask = self.get_embedding(text, return_mask=True)

            if embedding is None:
                # Fallback: create zero embedding
                embedding = torch.zeros(max_length, 4096, device=self.device)  # T5-XXL dim
                mask = torch.zeros(max_length, dtype=torch.bool, device=self.device)
                print(f"Warning: No precomputed embedding for text: '{text[:50]}...', using zero embedding")

            # Pad or truncate to max_length
            seq_len = embedding.shape[0]
            if seq_len > max_length:
                # Truncate
                embedding = embedding[:max_length]
                mask = mask[:max_length]
            elif seq_len < max_length:
                # Pad
                pad_len = max_length - seq_len
                embedding = torch.cat([
                    embedding,
                    torch.zeros(pad_len, embedding.shape[1], device=self.device)
                ], dim=0)
                if mask is not None:
                    mask = torch.cat([
                        mask,
                        torch.zeros(pad_len, dtype=torch.bool, device=self.device)
                    ], dim=0)

            batch_embeddings.append(embedding)
            batch_masks.append(mask)

        # Stack into batch
        batch_embeddings = torch.stack(batch_embeddings, dim=0)  # [B, max_length, hidden_dim]

        if return_mask:
            batch_masks = torch.stack(batch_masks, dim=0)  # [B, max_length]
            return batch_embeddings, batch_masks
        else:
            return batch_embeddings

    def has_embedding(self, text: str) -> bool:
        """Check if we have precomputed embedding for given text"""
        return text in self.text_to_embedding

    def get_task_texts(self, task_name: str, desc_type: str = "train") -> List[str]:
        """Get all texts for a given task and description type"""
        texts = []
        for mapping in self.embedding_index:
            if mapping['task_name'] == task_name and mapping['desc_type'] == desc_type:
                texts.append(mapping['text'])
        return texts

    def get_available_tasks(self) -> List[str]:
        """Get list of available task names"""
        tasks = set()
        for mapping in self.embedding_index:
            tasks.add(mapping['task_name'])
        return sorted(list(tasks))

    def get_stats(self) -> Dict:
        """Get statistics about loaded embeddings"""
        stats = {
            'total_embeddings': len(self.embedding_index),
            'tasks': len(self.get_available_tasks()),
            'cached_embeddings': len(self.embeddings_cache),
        }

        # Count by description type
        desc_types = {}
        for mapping in self.embedding_index:
            desc_type = mapping['desc_type']
            desc_types[desc_type] = desc_types.get(desc_type, 0) + 1
        stats['desc_types'] = desc_types

        return stats


def create_fallback_embedder(text_encoder_path: str, device: str, max_length: int = 77):
    """Create fallback T5 embedder for texts not in precomputed embeddings"""
    try:
        from examples.baselines.rdt.models.multimodal_encoder.t5_encoder import T5Embedder
        return T5Embedder(
            from_pretrained=text_encoder_path,
            model_max_length=max_length,
            device=device
        )
    except ImportError:
        print("Warning: T5Embedder not available, fallback disabled")
        return None


class HybridLangEmbedding:
    """Hybrid language embedding that uses precomputed when available, fallback to real-time encoding"""

    def __init__(
        self,
        precomputed_dir: Optional[str] = None,
        text_encoder_path: str = "google/t5-v1_1-xxl",
        device: str = "cuda",
        max_length: int = 77
    ):
        self.device = device
        self.max_length = max_length

        # Initialize precomputed embeddings if available
        self.precomputed = None
        if precomputed_dir and os.path.exists(precomputed_dir):
            try:
                self.precomputed = PrecomputedLangEmbedding(precomputed_dir, device)
                print(f"✓ Loaded precomputed embeddings from {precomputed_dir}")
            except Exception as e:
                print(f"Warning: Failed to load precomputed embeddings: {e}")

        # Initialize fallback embedder
        self.fallback_embedder = None
        if self.precomputed is None or True:  # Always create fallback for missing texts
            self.fallback_embedder = create_fallback_embedder(text_encoder_path, device, max_length)
            if self.fallback_embedder:
                print(f"✓ Fallback T5 embedder initialized")

    def get_text_embeddings(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get embeddings for batch of texts

        Args:
            texts: List of texts to encode

        Returns:
            Tuple of (embeddings, masks)
            embeddings: [B, max_length, hidden_dim]
            masks: [B, max_length]
        """
        if self.precomputed:
            # Try precomputed first
            try:
                embeddings, masks = self.precomputed.get_batch_embeddings(
                    texts, self.max_length, return_mask=True
                )

                # Check if any embeddings are missing (zero embeddings indicate missing)
                missing_indices = []
                for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                    if torch.allclose(embedding, torch.zeros_like(embedding)):
                        missing_indices.append(i)

                # If some embeddings are missing and we have fallback, encode them
                if missing_indices and self.fallback_embedder:
                    missing_texts = [texts[i] for i in missing_indices]
                    fallback_embeddings, fallback_masks = self.fallback_embedder.get_text_embeddings(missing_texts)

                    # Replace missing embeddings
                    for i, idx in enumerate(missing_indices):
                        embeddings[idx] = fallback_embeddings[i]
                        masks[idx] = fallback_masks[i]

                return embeddings, masks

            except Exception as e:
                print(f"Warning: Precomputed embedding failed: {e}, falling back to real-time encoding")

        # Fallback to real-time encoding
        if self.fallback_embedder:
            return self.fallback_embedder.get_text_embeddings(texts)
        else:
            # Last resort: return zero embeddings
            B = len(texts)
            embeddings = torch.zeros(B, self.max_length, 4096, device=self.device)  # T5-XXL dim
            masks = torch.zeros(B, self.max_length, dtype=torch.bool, device=self.device)
            print("Warning: No embedding method available, returning zero embeddings")
            return embeddings, masks

    def get_stats(self) -> Dict:
        """Get statistics about embedding usage"""
        stats = {'fallback_available': self.fallback_embedder is not None}
        if self.precomputed:
            stats.update(self.precomputed.get_stats())
        return stats