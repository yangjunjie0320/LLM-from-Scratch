import torch, typing
from typing import Optional

def generate(model : torch.nn.Module, input_ids : torch.Tensor, 
             max_new_tokens : int = 20, temperature : float = 1.0, 
             repetition_penalty : Optional[float] = None, 
             do_sample : Optional[bool] = None, top_k : Optional[int] = None) -> torch.Tensor:
    """
    A simple text generation function.
    
    Args:
        model: The GPT model
        input_ids: Starting token ids, a integer tensor of shape (batch_size, seq_len)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Controls randomness (lower = more deterministic)
    
    Returns:
        Generated token ids including the input_ids
    """
    if repetition_penalty is not None:
        raise NotImplementedError("Repetition penalty is not implemented")
    
    if do_sample is not None:
        raise NotImplementedError("Do sample is not implemented")
    
    if top_k is not None:
        raise NotImplementedError("Top k is not implemented")

    # Set the model to evaluation mode
    model.eval()
    
    # Clone the input_ids, generated is a integer 
    # tensor of shape (batch_size, seq_len)
    generated : torch.Tensor = input_ids.clone()
    batch_size : int = generated.shape[0]
    seq_len : int = generated.shape[1]
    vocab_size : int = model.config.vocab_size # type: ignore
    assert generated.shape == (batch_size, seq_len)
    print(generated.shape)
    
    # Generate tokens one by one
    with torch.no_grad():
        for itoken in range(max_new_tokens):
            # Get logits from the model
            outputs = model(generated)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            print(logits.shape)

            # Logits is a float tensor of shape (batch_size, seq_len + itoken + 1, vocab_size)
            assert logits.shape == (batch_size, seq_len + itoken, vocab_size)
            assert logits.dtype == torch.float32

            # Focus on the last token's predictions
            # next_token_logits is a float tensor of shape (batch_size, vocab_size)
            next_token_logits = logits[:, -1, :]
            assert next_token_logits.shape == (batch_size, vocab_size)
            assert next_token_logits.dtype == torch.float32
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Convert logits to probabilities
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            assert probs.shape == (batch_size, vocab_size)
            assert probs.dtype == torch.float32
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            assert next_token.shape == (batch_size, 1)
            assert next_token.dtype == torch.int64
            
            # Append the new token to the sequence
            generated = torch.cat((generated, next_token), dim=1)
            assert generated.shape == (batch_size, seq_len + itoken + 1)
            assert generated.dtype == torch.int64
    
    return generated

if __name__ == "__main__":
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    text = "My name is"
    tokens = tokenizer.encode(text, return_tensors="pt")

    m_ref = GPT2LMHeadModel.from_pretrained("gpt2")
    m_ref.eval()

    out = generate(
        m_ref, tokens, max_new_tokens=20, temperature=1.0, 
        repetition_penalty=None, do_sample=None, top_k=None
        )
    print(tokenizer.decode(out[0].tolist()).replace("\n", ""))
