import torch
from torch.utils.data import Dataset
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BilingualDataset(Dataset):
    """
    A PyTorch Dataset for handling bilingual data in machine translation tasks.
    
    This dataset handles the preprocessing of source and target language pairs,
    including tokenization, padding, and mask generation for transformer models.
    
    Attributes:
        dataset: The raw dataset containing translation pairs
        tokenizer_enc: Tokenizer for the source/encoder language
        tokenizer_dec: Tokenizer for the target/decoder language
        enc_language: Source language identifier (e.g., 'en', 'fr')
        dec_language: Target language identifier
        seq_len: Maximum sequence length for both source and target
    """
    
    def __init__(
        self,
        dataset: Any,
        tokenizer_enc: Any,
        tokenizer_dec: Any,
        enc_language: str,
        dec_language: str,
        seq_len: int
    ):
        """
        Initialize the BilingualDataset.
        
        Args:
            dataset: Dataset containing translation pairs
            tokenizer_enc: Tokenizer for source language
            tokenizer_dec: Tokenizer for target language
            enc_language: Source language code
            dec_language: Target language code
            seq_len: Maximum sequence length
        """
        super().__init__()
        
        # Validate inputs
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        if not dataset:
            raise ValueError("Dataset cannot be empty")
            
        self.dataset = dataset
        self.tokenizer_enc = tokenizer_enc
        self.tokenizer_dec = tokenizer_dec
        self.enc_language = enc_language
        self.dec_language = dec_language
        self.seq_len = seq_len
        
        # Initialize special tokens
        try:
            self.sos_token = torch.tensor([tokenizer_dec.token_to_id("<sos>")], dtype=torch.int64)
            self.eos_token = torch.tensor([tokenizer_dec.token_to_id("<eos>")], dtype=torch.int64)
            self.pad_token = torch.tensor([tokenizer_dec.token_to_id("<pad>")], dtype=torch.int64)
        except Exception as e:
            raise ValueError(f"Failed to initialize special tokens: {str(e)}")
            
    def __len__(self) -> int:
        """Return the number of translation pairs in the dataset."""
        return len(self.dataset)
        
    def _tokenize_and_validate(self, text: str, tokenizer: Any, is_encoder: bool = True) -> torch.Tensor:
        """
        Tokenize text and validate token length.
        
        Args:
            text: Input text to tokenize
            tokenizer: Tokenizer to use
            is_encoder: Whether this is for encoder input (True) or decoder input (False)
            
        Returns:
            torch.Tensor: Tokenized input
        """
        try:
            tokens = tokenizer.encode(text).ids
            max_allowed = self.seq_len - (3 if is_encoder else 2)  # Account for special tokens
            
            if len(tokens) > max_allowed:
                logger.warning(f"Input sequence length {len(tokens)} exceeds maximum allowed length {max_allowed}")
                tokens = tokens[:max_allowed]
                
            return torch.tensor(tokens, dtype=torch.int64)
        except Exception as e:
            raise RuntimeError(f"Tokenization failed: {str(e)}")
            
    def _create_padding_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        """Create attention padding mask."""
        return (tensor != self.pad_token).unsqueeze(0).unsqueeze(0).int()
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a processed translation pair.
        
        Args:
            idx: Index of the translation pair
            
        Returns:
            Dictionary containing processed tensors and metadata
        """
        try:
            # Get translation pair
            enc_dec_pair = self.dataset[idx]
            enc_text = enc_dec_pair['translation'][self.enc_language]
            dec_text = enc_dec_pair['translation'][self.dec_language]
            
            # Tokenize inputs
            enc_tokens = self._tokenize_and_validate(enc_text, self.tokenizer_enc)
            dec_tokens = self._tokenize_and_validate(dec_text, self.tokenizer_dec, is_encoder=False)
            
            # Calculate padding lengths
            enc_pad_len = self.seq_len - len(enc_tokens) - 2  # For <sos> and <eos>
            dec_pad_len = self.seq_len - len(dec_tokens) - 1  # For <sos> only
            
            # Create padded sequences
            encoder_input = torch.cat([
                self.sos_token,
                enc_tokens,
                self.eos_token,
                self.pad_token.repeat(enc_pad_len)
            ])
            
            decoder_input = torch.cat([
                self.sos_token,
                dec_tokens,
                self.pad_token.repeat(dec_pad_len)
            ])
            
            label = torch.cat([
                dec_tokens,
                self.eos_token,
                self.pad_token.repeat(dec_pad_len)
            ])
            
            # Verify tensor shapes
            assert encoder_input.size(0) == self.seq_len, f"Encoder input size mismatch: {encoder_input.size(0)}"
            assert decoder_input.size(0) == self.seq_len, f"Decoder input size mismatch: {decoder_input.size(0)}"
            assert label.size(0) == self.seq_len, f"Label size mismatch: {label.size(0)}"
            
            return {
                'encoder_input': encoder_input,
                'decoder_input': decoder_input,
                'encoder_mask': self._create_padding_mask(encoder_input),
                'decoder_mask': self._create_padding_mask(decoder_input) & self.causal_mask(decoder_input.size(0)),
                'label': label,
                'enc_text': enc_text,
                'dec_text': dec_text
            }
            
        except Exception as e:
            logger.error(f"Error processing item {idx}: {str(e)}")
            raise
            
    @staticmethod
    def causal_mask(size: int) -> torch.Tensor:
        """
        Create causal mask for decoder attention.
        
        Args:
            size: Size of the sequence
            
        Returns:
            torch.Tensor: Causal attention mask
        """
        mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.bool)
        return ~mask  # Invert mask to match attention convention
        
    def decode_tokens(self, tokens: torch.Tensor) -> str:
        """
        Decode token ids back to text (useful for debugging).
        
        Args:
            tokens: Tensor of token ids
            
        Returns:
            str: Decoded text
        """
        try:
            # Remove padding tokens
            mask = tokens != self.pad_token
            cleaned_tokens = tokens[mask].tolist()
            return self.tokenizer_dec.decode(cleaned_tokens)
        except Exception as e:
            logger.error(f"Failed to decode tokens: {str(e)}")
            return ""