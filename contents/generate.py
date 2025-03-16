import torch, typing
from typing import Optional

def generate(model : torch.nn.Module, input_ids : torch.Tensor, 
             attention_mask : Optional[torch.Tensor] = None,
             max_new_tokens : int = 20, temperature : float = 1.0):

    # Set the model to evaluation mode
    model.eval()
    
    # Clone the input_ids, generated is a integer 
    # tensor of shape (batch_size, seq_len)
    generated : torch.Tensor = input_ids
    batch_size : int = generated.shape[0]
    seq_len : int = generated.shape[1]
    vocab_size : int = model.config.vocab_size # type: ignore
    assert generated.shape == (batch_size, seq_len)
    
    # Generate tokens one by one
    with torch.no_grad():
        for itoken in range(max_new_tokens):
            # Get logits from the model
            outputs = model(generated, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    m_ref = GPT2LMHeadModel.from_pretrained("gpt2")
    m_ref.eval()

    # get token and attention mask
    text = "My name is"
    tokens = tokenizer(text, return_tensors="pt")

    input_ids = tokens["input_ids"] 
    attention_mask = tokens["attention_mask"]

    out = m_ref.generate(input_ids, attention_mask=attention_mask, max_length=20, temperature=1.0)
    print(tokenizer.decode(out[0].tolist()).replace("\n", ""))

    out = generate(
        m_ref, input_ids, attention_mask=attention_mask,
        max_new_tokens=20, temperature=1.0
        )
    print(tokenizer.decode(out[0].tolist()).replace("\n", ""))
