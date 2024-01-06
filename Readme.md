# End-to-End Language Translation.

### The goal of the project is to Implement Attention is all you need paper from scratch to perform language translation.

### overview of the approach as a algorithm:
1. Take the input data and tokenize them.
2. 

### Now below I am explaining how each and every step works.

### **step-1:** Take the input data and tokenize them.
* load the selected pair of languages from opus_books dataset using HuggingFace Datasets.
* using HuggingFace Tokenizer library to train and tokenize the input src and tgt languages

### **step-3** Build the Transformer model.
**Embedding layer:**
The job of this layer is to create d-dimensional embeddings by taking input tokens batches.

* Input: (Batch_size, Seq_len (or) Vocab_size)
* Output: (Batch_size, Seq_len, d_model)

**Positional Encoding Layer:**
This layer adds a Positional encodings to the output of Embeddings layer so the model will have information about the order of sequence while computing the self-attention.

* Input: (Batch_size, Seq_len, d_model).
* Pos_encoding: (Batch_size, Seq_len, d_model).
* output = Input + pos_encoding => (Batch, Seq_len, d_model).