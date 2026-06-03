import torch
import torch.nn.functional as F
from encoder.encoder import EncoderBlock
from components.embeddings import input_embeddings, output_embeddings, input_one_hot, output_one_hot, tokenizer, encoder_sentences, output_tensors, embedding_matrix
from decoder.decoder import DecoderBlock
from components.positional_embeddings import get_positional_embeddings
from components.output_generation import generate_output
from config import DEVICE, D_MODEL, SEQ_LEN, BATCH_SIZE, D_HEAD, LR


# print(f'----> Input Embeddings:\n {input_embeddings}')
# print(f'----> Output Embeddings:\n {output_embeddings}')

position = get_positional_embeddings(SEQ_LEN, DEVICE, D_MODEL)
# print(f'----> 2D positional embeddings with shape: {position.shape}')
# print(f'----> embeddings with shape:\t {input_embeddings.shape}')

# final_encoder_embds = input_embeddings + position
# print(f'----> The final embeddings after adding positional embeddings are:\n {final_encoder_embds}')

encoder_block = EncoderBlock().to(DEVICE)
# encoder_output = encoder_block(final_encoder_embds)
# print(f'----> Encoder Block output:\n {encoder_output}')
# print(f'----> Encoder Block output shape: {encoder_output.shape}')


# final_decoder_embds = output_embeddings + position
# print(f'----> The final decoder embeddings after adding positional embeddings are:\n {final_decoder_embds}')

decoder_block = DecoderBlock().to(DEVICE)
# decoder_output = decoder_block(final_decoder_embds, encoder_output)
# print(f'----> Decoder Block output:\n {decoder_output}')
# print(f'----> Decoder Block output shape: {decoder_output.shape}')

# output_probs = generate_output(decoder_output)
# print(f'----> Final output probabilities:\n {output_probs}')
# print(f'----> Final output probabilities shape: {output_probs.shape}')

# Final output generation
# tokenizer_output = output_probs.argmax(dim=-1)
# print(f'----> Tokenizer output (predicted token IDs):\n {tokenizer_output}')
# print(f'----> Tokenizer output shape: {tokenizer_output.shape}')

# Decoding token IDs back to words
# Assuming we have access to the tokenizer instance used for encoding
# decoded_sentences = [tokenizer.decode(token_ids) for token_ids in tokenizer_output.tolist()]
# print(f'----> Input Sentences:\n {encoder_sentences}')
# print(f'----> Decoded sentences:\n {decoded_sentences}')


# Loss calculation
# print(f'----> Output Tensors (target token IDs):\n {output_tensors}')
# print(f'----> Output probabilities shape: {output_probs.shape}') # torch.Size([3, 4, 14])
# print(f'----> Output tensors shape: {output_tensors.shape}') # torch.Size([3, 4])
# loss = -output_probs[torch.arange(output_probs.size(0)).unsqueeze(1), torch.arange(output_probs.size(1)).unsqueeze(0), output_tensors].log().mean()
# print(f'----> Loss value: {loss.item()}')


# listing all the parameters of the encoder and decoder blocks to check if they are being updated
# Embeddings Matrix, and all parameters of the encoder and decoder blocks should have gradients after backpropagation
# print("-" * 100)
# print(f'----> Before backpropagation, loss value: {loss.item()}')

# print(f'----> Parameters of the encoder block:')
# for name, param in encoder_block.named_parameters():
#     print(f'{name}: requires_grad={param.requires_grad}, grad={param.grad}')


# print(f'----> Parameters of the decoder block:')
# for name, param in decoder_block.named_parameters():
#     print(f'{name}: requires_grad={param.requires_grad}, grad={param.grad}')

# print(f'----> Parameters of the embedding matrix:')
# print(f'embedding_matrix: requires_grad={embedding_matrix.requires_grad}, grad={embedding_matrix.grad}')


# Now we will perform backpropagation to check if gradients are being computed for the parameters of the encoder and decoder blocks, as well as the embedding matrix.
# print("-" * 100)
# loss.backward()

# print("-" * 100)
# print(f'----> Before backpropagation, loss value: {loss.item()}')

# print(f'----> Parameters of the encoder block:')
# for name, param in encoder_block.named_parameters():
#     print(f'{name}: requires_grad={param.requires_grad}, grad={param.grad}')


# print(f'----> Parameters of the decoder block:')
# for name, param in decoder_block.named_parameters():
#     print(f'{name}: requires_grad={param.requires_grad}, grad={param.grad}')

# print(f'----> Parameters of the embedding matrix:')
# print(f'embedding_matrix: requires_grad={embedding_matrix.requires_grad}, grad={embedding_matrix.grad}')


# Training Starts From here ----------------------------------------------------------------------------------------------------------------------------------
for epoch in range(1000):
    # print(f'----> Epoch {epoch+1}')

    # Recompute embeddings inside the loop to build a new computation graph
    input_embeddings = torch.matmul(input_one_hot, embedding_matrix)
    output_embeddings = torch.matmul(output_one_hot, embedding_matrix)
    final_encoder_embds = input_embeddings + position
    final_decoder_embds = output_embeddings + position

    # Forward pass
    encoder_output = encoder_block(final_encoder_embds)
    # print("-" * 50)
    # print(f'----> Encoder Block output:\n {encoder_output}')
    # print(f'----> Encoder Block output shape: {encoder_output.shape}')
    decoder_output = decoder_block(final_decoder_embds, encoder_output)
    output_probs = generate_output(decoder_output)
    
    # Loss calculation
    loss = -output_probs[torch.arange(output_probs.size(0)).unsqueeze(1), torch.arange(output_probs.size(1)).unsqueeze(0), output_tensors].log().mean()
    if (epoch+1) % 10 == 0:
        print(f'---->Epoch {epoch+1}, Loss value: {loss.item()}')

    # Preventing gradients from previous epoch from accumulating
    encoder_block.zero_grad()
    decoder_block.zero_grad()
    embedding_matrix.grad = None
    
    # Backpropagation
    loss.backward()

    # Updating Parameters
    with torch.no_grad():
        for param in encoder_block.parameters():
            param -= LR * param.grad
        for param in decoder_block.parameters():
            param -= LR * param.grad
        embedding_matrix -= LR * embedding_matrix.grad






# loss.backward()
# print(f'----> Gradients for encoder block parameters:')
# for name, param in encoder_block.named_parameters():
#     if param.grad is not None:
#         print(f'{name}: {param.grad.norm().item()}')
#     else:
#         print(f'{name}: No gradient computed')



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