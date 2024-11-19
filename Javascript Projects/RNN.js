// Sample text data (you can change this)
let text = "simple example text";
let chars = [...new Set(text)]; // Unique characters

// Create mappings
let char_to_idx = {};
let idx_to_char = {};

chars.forEach((ch, idx) => {
    char_to_idx[ch] = idx;
    idx_to_char[idx] = ch;
});

// Convert text to indices
let sequence = text.split("").map(ch => char_to_idx[ch]);

// Initialize weights and hidden state
let weights_input = 0.5;
let weights_hidden = 0.5;
let hidden_state = 0;

// Simplified RNN forward pass
for (let idx = 0; idx < sequence.length - 1; idx++) {
    let input_char = sequence[idx];

    // Update hidden state using a simple formula
    hidden_state = weights_input * input_char + weights_hidden * hidden_state;

    // Predict the next character index
    let predicted_idx = Math.floor(hidden_state) % chars.length;
    let predicted_char = idx_to_char[predicted_idx];

    // Display the prediction
    console.log(`Input: ${idx_to_char[input_char]} -> Predicted: ${predicted_char}`);
}
