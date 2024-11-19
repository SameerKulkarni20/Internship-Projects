# Sample text data (you can change this)
text = "simple example text"
chars = list(set(text))  # Unique characters
char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}

# Convert text to indices
sequence = [char_to_idx[ch] for ch in text]

# Initialize weights and hidden state
weights_input = 0.5
weights_hidden = 0.5
hidden_state = 0

# Simplified RNN forward pass
for idx in range(len(sequence) - 1):
    input_char = sequence[idx]

    # Update hidden state using a simple formula
    hidden_state = weights_input * input_char + weights_hidden * hidden_state

    # Predict the next character index
    predicted_idx = int(hidden_state) % len(chars)
    predicted_char = idx_to_char[predicted_idx]

    # Display the prediction
    print(f"Input: {idx_to_char[input_char]} -> Predicted: {predicted_char}")
