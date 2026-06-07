import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from components.tokenizer import WordTokenizer
from components.dataset import (
    tokenizer, token_ids, id_tokens,
    train_df, val_df, test_df,
    encoder_train_input_tensors, decoder_train_input_tensors, target_train_input_tensors,
    encoder_val_input_tensors, decoder_val_input_tensors, target_val_input_tensors,
    encoder_test_input_tensors, decoder_test_input_tensors, target_test_input_tensors,
)
from components.embeddings import Embeddings
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
encoder_block = TransformerEncoder().to(DEVICE)
decoder_block = TransformerDecoder().to(DEVICE)
output_gen = OutputGeneration(shared_weight=shared_weight).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=token_ids['<PAD>']).to(DEVICE)  # Cross-entropy loss with padding token ignored

# Adam optimizer
optimizer = optim.Adam(
    list(
        encoder_block.parameters()) +
        list(decoder_block.parameters()) +
        list(output_gen.parameters()
        ),  # Include embedding layer parameters in optimization
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

    total_train_loss = 0.0
    total_val_loss = 0.0
    for batch in range(train_batches):
        start_idx = batch * BATCH_SIZE
        train_end_idx = min(start_idx + BATCH_SIZE, len(encoder_train_input_tensors))
        
        encoder_train_batch = encoder_train_input_tensors[start_idx:train_end_idx]
        decoder_train_batch = decoder_train_input_tensors[start_idx:train_end_idx]
        target_train_batch = target_train_input_tensors[start_idx:train_end_idx]

        LR = D_MODEL ** (-0.5) * min((epoch+1) ** (-0.5), (epoch+1) * (WARMUP_STEPS ** (-1.5))) # Learning rate scheduling as per the original transformer paper

        optimizer.param_groups[0]['lr'] = LR  # Update the learning rate in the optimizer

        
        # Recompute embeddings inside the loop to build a new computation graph
        encoder_input_embeddings = embedding_layer(encoder_train_batch)
        decoder_output_embeddings = embedding_layer(decoder_train_batch)
        
        # Forward pass
        encoder_output = encoder_block(encoder_input_embeddings)
        decoder_output = decoder_block(decoder_output_embeddings, encoder_output)
        logits = output_gen(decoder_output)
        
        # Loss calculation
        loss = criterion(logits.view(-1, logits.size(-1)), target_train_batch.view(-1))

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
        target_val_batch = target_val_input_tensors[start_idx:val_end_idx]

        with torch.no_grad():
            # Calculating validation loss at the end of each epoch
            encoder_block.eval()
            decoder_block.eval()
            output_gen.eval()
            
            val_encoder_input_embeddings = embedding_layer(encoder_val_batch)
            val_decoder_output_embeddings = embedding_layer(decoder_val_batch)
            val_encoder_output = encoder_block(val_encoder_input_embeddings)
            val_decoder_output = decoder_block(val_decoder_output_embeddings, val_encoder_output)
            val_logits = output_gen(val_decoder_output)
            val_loss = criterion(val_logits.view(-1, val_logits.size(-1)), target_val_batch.view(-1))

            total_val_loss += val_loss.item()
    avg_val_loss = total_val_loss / len(encoder_val_batch)
    history['val_loss'].append(avg_val_loss)
    

    if (epoch+1) % 50 == 0:
        print(
            f"Epoch {epoch+1:4d} | "
            f"LR: {LR:.4e} | "
            f"Opt LR: {optimizer.param_groups[0]['lr']:.4e} | "
            f"Train Loss: {total_train_loss:.6f} | "
            f"Val Loss: {total_val_loss:.6f}"
        )
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