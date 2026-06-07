import re

def filter_invalid_sentences(df, src_col='de', trg_col='en', min_words=3):
    """
    Filters out rows that are empty, too short, or contain non-sentence junk.
    """
    df = df.dropna(subset=[src_col, trg_col])
    
    df[src_col] = df[src_col].astype(str).str.strip()
    df[trg_col] = df[trg_col].astype(str).str.strip()
    
    word_count_mask = (df[src_col].str.split().str.len() >= min_words) & \
                      (df[trg_col].str.split().str.len() >= min_words)
    df = df[word_count_mask]
    
    # This filters out rows that are purely numbers, timestamps, or punctuation lines like "---"
    has_letters_regex = r'[a-zA-Z-äöüßÄÖÜ]' # Broadened to catch German umlauts too
    letter_mask = df[src_col].str.contains(has_letters_regex, regex=True) & \
                  df[trg_col].str.contains(has_letters_regex, regex=True)
    df = df[letter_mask]
    
    return df.reset_index(drop=True)