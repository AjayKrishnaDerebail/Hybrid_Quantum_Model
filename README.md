# Hybrid_Quantum_Model
![Diagram dipicting the hybrid quantum model](https://github.com/AjayKrishnaDerebail/Hybrid_Quantum_Model/assets/85861443/1fafe0cc-8aef-4dd2-8db7-6b322629e2a7)
<div align="justify">
# PROPOSED MODEL

In our methodology, we have developed an architecture, as illustrated in the above figure.
Our proposed 
system consists of six main stages: 1) Data pre-processing, 2) Embedding tweets 
using BERT, 3) Dimensionality reduction using UMAP, 4) Building Hybrid Quantum 
Model, 5) Classification layer. 

# 1) Data pre-processing

In the hybrid quantum model for depression detection using tweets, data preprocessing plays a crucial role in ensuring accurate and reliable results. Two key aspects focused on are noise estimation and noise filtering. The model addresses various types of noise present in tweets, such as spelling errors, abbreviations, and emoticons, which can impact data quality. To overcome this, advanced techniques including Lemmatization, Sentence tokenization, and Word vector tokenization are employed. Lemmatization involves reducing words to their base or dictionary form, simplifying subsequent analyses and enhancing accuracy. Sentence tokenization identifies sentence boundaries, crucial for many natural language processing tasks. Word vector tokenization represents words as numerical vectors, capturing semantic relationships and enabling complex language processing tasks. These preprocessing techniques contribute to the effectiveness of the depression detection model.

# 2) Embedding tweets using BERT

BERT is a deep learning model that uses a transformer architecture to pre-train on large amounts of unlabeled text data and then fine-tune on specific natura language processing tasks. The pre-training involves training on two self-supervised learning Hybrid Quantum Model For Depression Detectiontasks: masked languagemodeling and next sentence prediction. During pre-training, BERT takes a sequence of input tokens and converts them into embeddings using a combination of token and segment embeddings. The embeddings are then passed through a series of transformer encoder layers that use self-attention mechanisms to compute a weighted sum of the input embeddings. The output of each layer serves as the input to the next layer

# 3) Dimensionality reduction using UMAP

UMAP (Uniform Manifold Approximation and Projection) is a machine learning algorithm used for dimensionality reduction. It is a nonlinear dimensionality reduction 
technique that can be used for visualization and clustering of high-dimensional data. Dimensionality reduction is the process of reducing the number of variables in a dataset while retaining important information. It is often used to simplify the analysis of complex datasets, where the number of variables can be very large.

# 4) Building Hybrid Quantum Model

VQC, or Variational Quantum Classifier, is a hybrid machine learning algorithm that combines classical machine learning with quantum computing. It uses a quantum 
circuit to encode input data and a classical model to perform classification. During training, the parameters of the quantum circuit are optimized to minimize cost function that measures the difference between the predicted and actual labels of the training data. VQC has several advantages over classical machine learning, including the potential to achieve better performance and process quantum information more efficiently.

# Algorithm
Input: A labeled tweet dataset, transformed into embeddings using BERT and 
dimensionality reduced with UMAP, for training a VQC circuit.
Output: A parameterized quantum circuit which can classify new input data.
1. Initialize θ randomly
2. Define ψ(θ) = M ∗ U(θ) ∗ ϕ(x)
3. For each (x_i, y_i) in the training set: 
4. ϕ(x_i) = ϕ(x_i)
5. ψ(θ) = U(θ) ∗ ϕ(x_i)
6. y_i′ = M(ψ(θ))
7. L(θ) = f(y_i′, y_i)
8. ∇L(θ) = ∇θL(θ) #Compute the gradient of the cost function with respect to θ
9. dψ/dθ = dU(θ)/dθ ∗ ϕ(x_i) # Derivative of ψ(θ) with respect to θ
10.dy′_i/dθ = dM(ψ(θ))/dθ # Derivative of y_i' with respect to θ
11.dL(θ)/dθ = ∇L(θ) ∗ dy′_i/dθ # Derivative of L(θ) with respect to θ
12.θ = θ − α ∗ dL(θ)/dθ # Update θ using optimizer with learning rate α
13.Return trained VQC circui

# 5) Classification layer

When constructing a multilayer perceptron (MLP) model for binary classification, one can use dense layers as the building blocks of the neural network. In contrast to recurrent neural networks (RNNs) like LSTMs, dense layers do not have a temporal component and are instead fully connected. Each neuron in a dense layer receives inputs from all the neurons in the previous layer, computes a weighted sum of those inputs, and passes them through an activation function to produce an output. In the context of binary classification, the last layer of the MLP typically has a single neuron with a sigmoid activation function, which outputs a value between 0 and 1 representing the predicted probability of belonging to the positive class.
</div>
