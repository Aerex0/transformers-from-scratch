import torch
import torch.nn.functional as F
from encoder.encoder import EncoderBlock
from components.embeddings import input_embeddings, output_embeddings, input_one_hot, output_one_hot, tokenizer, encoder_sentences, output_tensors, embedding_matrix
from decoder.decoder import DecoderBlock
from components.positional_embeddings import get_positional_embeddings
from components.output_generation import OutputGeneration
from config import DEVICE, D_MODEL, SEQ_LEN, BATCH_SIZE, D_HEAD, LR



# Initialize components

position = get_positional_embeddings(SEQ_LEN, DEVICE, D_MODEL)
encoder_block = EncoderBlock().to(DEVICE)
decoder_block = DecoderBlock().to(DEVICE)
output_gen = OutputGeneration().to(DEVICE)


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
    output_probs = output_gen(decoder_output)
    
    # Loss calculation
    loss = -output_probs[torch.arange(output_probs.size(0)).unsqueeze(1), torch.arange(output_probs.size(1)).unsqueeze(0), output_tensors].log().mean()
    if (epoch+1) % 10 == 0:
        print(f'---->Epoch {epoch+1}, Loss value: {loss.item()}')

    # Preventing gradients from previous epoch from accumulating
    encoder_block.zero_grad()
    decoder_block.zero_grad()
    output_gen.zero_grad()
    embedding_matrix.grad = None
    
    # Backpropagation
    loss.backward()

    # Updating Parameters
    with torch.no_grad():
        for param in encoder_block.parameters():
            param -= LR * param.grad
        for param in decoder_block.parameters():
            param -= LR * param.grad
        for param in output_gen.parameters():
            param -= LR * param.grad
        embedding_matrix -= LR * embedding_matrix.grad