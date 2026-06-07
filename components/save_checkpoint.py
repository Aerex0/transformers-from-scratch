import os

def save_checkpoint(epoch, encoder, decoder, embeddings, output_gen, optimizer, history, filename="transformer_checkpoint.pt"):
    """
    Saves the entire state architecture of the model to disk.
    """
    checkpoint = {
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'embeddings_state_dict': embeddings.state_dict(),
        'output_gen_state_dict': output_gen.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }
    
    torch.save(checkpoint, filename)
    print(f"==> Checkpoint successfully saved to '{filename}' at epoch {epoch}!")