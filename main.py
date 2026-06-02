from encoder.encoder import EncoderBlock
from components.embeddings import input_embeddings, output_embeddings, tokenizer
from decoder.decoder import DecoderBlock
from components.positional_embeddings import get_positional_embeddings
from components.output_generation import generate_output
from config import DEVICE, D_MODEL, SEQ_LEN, BATCH_SIZE

print(f'----> Input Embeddings:\n {input_embeddings}')
print(f'----> Output Embeddings:\n {output_embeddings}')

position = get_positional_embeddings(SEQ_LEN, DEVICE, D_MODEL)
print(f'----> 2D positional embeddings with shape: {position.shape}')
print(f'----> embeddings with shape:\t {input_embeddings.shape}')

final_encoder_embds = input_embeddings + position
print(f'----> The final embeddings after adding positional embeddings are:\n {final_encoder_embds}')

encoder_block = EncoderBlock().to(DEVICE)
encoder_output = encoder_block(final_encoder_embds)
print(f'----> Encoder Block output:\n {encoder_output}')
print(f'----> Encoder Block output shape: {encoder_output.shape}')


final_decoder_embds = output_embeddings + position
print(f'----> The final decoder embeddings after adding positional embeddings are:\n {final_decoder_embds}')

decoder_block = DecoderBlock(encoder_output=encoder_output).to(DEVICE)
decoder_output = decoder_block(final_decoder_embds, encoder_output)
print(f'----> Decoder Block output:\n {decoder_output}')
print(f'----> Decoder Block output shape: {decoder_output.shape}')

output_probs = generate_output(decoder_output)
print(f'----> Final output probabilities:\n {output_probs}')
print(f'----> Final output probabilities shape: {output_probs.shape}')

# Final output generation
tokenizer_output = output_probs.argmax(dim=-1)
print(f'----> Tokenizer output (predicted token IDs):\n {tokenizer_output}')
print(f'----> Tokenizer output shape: {tokenizer_output.shape}')

# Decoding token IDs back to words
# Assuming we have access to the tokenizer instance used for encoding
decoded_sentences = [tokenizer.decode(token_ids) for token_ids in tokenizer_output.tolist()]
print(f'----> Decoded sentences:\n {decoded_sentences}')






# # Encoder Block -------------------------------------------------------------

# # Multi-Head Attention
# multi_head_attn = MultiHeadAttention(D_MODEL, n_heads=N_HEADS).to(DEVICE)
# attn_output = multi_head_attn(final_embds)
# print(f'----> Multi Head Attention output:\n {attn_output}')
# print(f'----> Multi Head Attention output shape: {attn_output.shape}')

# # Residual Connection
# attn_output = attn_output + final_embds
# print(f'----> Output after adding Residual Connection:\n {attn_output}')
# print(f'----> Output after adding Residual Connection shape: {attn_output.shape}')

# # Layer Normalization
# layer_norm = LayerNorm(D_MODEL).to(DEVICE)
# norm_output = layer_norm(attn_output)
# print(f'----> Layer Normalized output:\n {norm_output}')
# print(f'----> Layer Normalized output shape: {norm_output.shape}')

# # Feed Forward Network and another Residual Connection + Layer Norm
# feed_forward = FeedForward().to(DEVICE)
# ff_output = feed_forward(norm_output)
# print(f'----> Feed Forward output:\n {ff_output}')
# print(f'----> Feed Forward output shape: {ff_output.shape}')

# # Final Residual Connection + Layer Norm
# E1_output = layer_norm(ff_output + norm_output)
# print(f'----> Final output after second Residual Connection and Layer Norm:\n {E1_output}')
# print(f'----> Final output shape: {E1_output.shape}')