import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from components.tokenizer import WordTokenizer
from components.dataset import encoder_train_input_tensors, decoder_train_input_tensors, encoder_val_input_tensors, decoder_val_input_tensors, encoder_test_input_tensors, decoder_test_input_tensors, tokenizer, token_ids, id_tokens, train_df, val_df, test_df
from components.embeddings import embedding_matrix, Embeddings
from encoder.encoder import EncoderBlock, TransformerEncoder
from decoder.decoder import DecoderBlock, TransformerDecoder
from components.positional_embeddings import get_positional_embeddings
from components.output_generation import OutputGeneration
from config import DEVICE, D_MODEL, SEQ_LEN, BATCH_SIZE, D_HEAD, TOTAL_STEPS, ENCODER_LAYERS, DECODER_LAYERS, N_HEADS, WARMUP_STEPS, BETA1, BETA2, EPS

print(f'Train set size: {len(train_df)}, Validation set size: {len(val_df)}, Test set size: {len(test_df)}')
print(f'Device being used for training: {DEVICE}')
print(f'Dimension of Model (D_MODEL): {D_MODEL}, Sequence Length (SEQ_LEN): {SEQ_LEN}, Batch Size: {BATCH_SIZE}')
print(f'Number of Encoder Layers: {ENCODER_LAYERS}, Number of Decoder Layers: {DECODER_LAYERS}, Number of Attention Heads: {N_HEADS}')
print(f'Learning Rate: ADAM, beta1: {BETA1}, beta2: {BETA2}, eps: {EPS}, Number of Epochs: {TOTAL_STEPS}, Number of Warmup Steps: {WARMUP_STEPS}')
print(f'----> Sample from Train set:\n {train_df.sample(5 if len(train_df) >= 5 else len(train_df))}')

# Initialize components

position = get_positional_embeddings(SEQ_LEN, DEVICE, D_MODEL)
embedding_layer = Embeddings().to(DEVICE)

shared_weight = embedding_layer.embedding_matrix.weight  # Get the shared weight from the embedding layer
# print(f' Positional Embedding: {position}')
# print(f'Positional Embeddings shape: {position.shape}')
encoder_block = TransformerEncoder().to(DEVICE)
decoder_block = TransformerDecoder().to(DEVICE)
output_gen = OutputGeneration(shared_weight=shared_weight).to(DEVICE)

# Adam optimizer
optimizer = optim.Adam(
    list(encoder_block.parameters()) + list(decoder_block.parameters()) + list(output_gen.parameters()),  # Include embedding layer parameters in optimization
    betas=(BETA1, BETA2),
    eps=EPS
)

print("-" * 100)
print("Components Initialized Successfully. Starting Training Loop...")

# Initialize history tracking before the training loop
history = {
    'epoch': [],
    'train_loss': [],
    'val_loss': []
}

start_time = time.time()
# Training Starts From here --------------------------------------------------
for epoch in range(TOTAL_STEPS):
    train_batches = (len(encoder_train_input_tensors) + BATCH_SIZE - 1) // BATCH_SIZE
    val_batches = (len(encoder_val_input_tensors) + BATCH_SIZE - 1) // BATCH_SIZE
    # print(f'Encoder train input tensors shape: {encoder_train_input_tensors.shape}, Decoder train input tensors shape: {decoder_train_input_tensors.shape}')

    # print(f'Encoder val input tensors shape: {encoder_val_input_tensors.shape}, Decoder val input tensors shape: {decoder_val_input_tensors.shape}')
    total_train_loss = 0.0
    total_val_loss = 0.0
    for batch in range(train_batches):
        start_idx = batch * BATCH_SIZE
        train_end_idx = min(start_idx + BATCH_SIZE, len(encoder_train_input_tensors))
        
        encoder_train_batch = encoder_train_input_tensors[start_idx:train_end_idx]
        decoder_train_batch = decoder_train_input_tensors[start_idx:train_end_idx]

        LR = D_MODEL ** (-0.5) * min((epoch+1) ** (-0.5), (epoch+1) * (WARMUP_STEPS ** (-1.5))) # Learning rate scheduling as per the original transformer paper

        optimizer.param_groups[0]['lr'] = LR  # Update the learning rate in the optimizer

        # print(f' Encoder train input tensors shape: {encoder_train_input_tensors.shape}')
        
        # Recompute embeddings inside the loop to build a new computation graph
        encoder_input_embeddings = embedding_layer(encoder_train_batch)
        # print(f'----> Encoder input embeddings shape: {encoder_input_embeddings.shape}')
        # print(f'----> Encoder input embeddings sample:\n {encoder_input_embeddings}')
        decoder_output_embeddings = embedding_layer(decoder_train_batch)
        # input_embeddings = torch.matmul(input_one_hot, embedding_matrix)
        # output_embeddings = torch.matmul(output_one_hot, embedding_matrix)

        # final_encoder_embds = encoder_input_embeddings + position
        # print(f'----> Final encoder embeddings after adding positional encodings shape: {final_encoder_embds.shape}')
        # print(f'----> Final encoder embeddings after adding positional encodings sample:\n {final_encoder_embds}')
        # final_decoder_embds = decoder_output_embeddings + position

        # Forward pass
        encoder_output = encoder_block(encoder_input_embeddings)
        # print("-" * 50)
        # print(f'----> Encoder Block output:\n {encoder_output}')
        # print(f'----> Encoder Block output shape: {encoder_output.shape}')
        decoder_output = decoder_block(decoder_output_embeddings, encoder_output)
        output_probs = output_gen(decoder_output)
        
        # Loss calculation
        loss = -output_probs[torch.arange(output_probs.size(0)).unsqueeze(1), torch.arange(output_probs.size(1)).unsqueeze(0), decoder_train_batch].log().mean()

        # Preventing gradients from previous epoch from accumulating
        # encoder_block.zero_grad()
        # decoder_block.zero_grad()
        # output_gen.zero_grad()
        # embedding_matrix.weight.grad = None
        optimizer.zero_grad()
        
        # Backpropagation
        loss.backward()

        # Update parameters using optimizer
        optimizer.step()
        total_train_loss += loss.item()
    
    # Calculate average training loss for the epoch
    avg_train_loss = total_train_loss / len(encoder_train_batch)
    history['epoch'].append(epoch + 1)
    history['train_loss'].append(avg_train_loss)


    for batch in range(val_batches):
        start_idx = batch * BATCH_SIZE
        val_end_idx = min(start_idx + BATCH_SIZE, len(encoder_val_input_tensors))
        
        encoder_val_batch = encoder_val_input_tensors[start_idx:val_end_idx]
        decoder_val_batch = decoder_val_input_tensors[start_idx:val_end_idx]

        # print(f' Val Encoder input tensors shape: {encoder_val_batch.shape}, Val Decoder input tensors shape: {decoder_val_batch.shape}')
    # Updating Parameters
        with torch.no_grad():
            # for param in encoder_block.parameters():
            #     param -= LR * param.grad
            # for param in decoder_block.parameters():
            #     param -= LR * param.grad
            # for param in output_gen.parameters():
            #     param -= LR * param.grad
            
            # Calculating validation loss at the end of each epoch
            encoder_block.eval()
            decoder_block.eval()
            output_gen.eval()
            
            val_encoder_input_embeddings = embedding_matrix(encoder_val_batch)
            val_decoder_output_embeddings = embedding_matrix(decoder_val_batch)
            val_final_encoder_embds = val_encoder_input_embeddings + position
            val_final_decoder_embds = val_decoder_output_embeddings + position
            val_encoder_output = encoder_block(val_final_encoder_embds)
            val_decoder_output = decoder_block(val_final_decoder_embds, val_encoder_output)
            val_output_probs = output_gen(val_decoder_output)
            val_loss = -val_output_probs[torch.arange(val_output_probs.size(0)).unsqueeze(1), torch.arange(val_output_probs.size(1)).unsqueeze(0), decoder_val_batch].log().mean()

            total_val_loss += val_loss.item()
    avg_val_loss = total_val_loss / len(encoder_val_batch)
    history['val_loss'].append(avg_val_loss)
    

    if (epoch+1) % 50 == 0:
        print(
            f"Epoch {epoch+1:4d} | "
            f"LR: {LR:.4e} | "
            f"Opt LR: {optimizer.param_groups[0]['lr']:.4e} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {val_loss.item():.6f}"
        )
        # embedding_matrix -= LR * embedding_matrix.weight.grad Since the weights of the linear layer in output generation are shared with the embedding matrix, we don't need to update them separately here. They will be updated through the gradients computed for the output generation linear layer.

end_time = time.time()
print(f'Training completed in {(end_time - start_time)/60:.2f} minutes.')


# Parameters Overview
print("\n" + "="*40 + "\n TRANSFORMER PARAMETERS SUMMARY \n" + "="*40)
encoder_params = encoder_block.print_parameters(Layer_values=False, print_values=False)
decoder_params = decoder_block.print_parameters(Layer_values=False, print_values=False)
output_params = output_gen.print_parameters(Layer_values=False, print_values=False)

total_params = encoder_params + decoder_params + output_params
print(f"\nTotal Trainable Parameters in the Transformer Model: {total_params:,}\n")


# Convert history dictionary to a DataFrame
df_history = pd.DataFrame(history)

# Save it to a CSV file in your working directory
df_history.to_csv("transformer_training_history.csv", index=False)
print("Training history successfully saved to disk!")