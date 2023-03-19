import tensorflow as tf
import numpy as np
from preprocess import get_data
from types import SimpleNamespace


class MyTrigram(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size=200, embed_size=64):
        """
        The Model class predicts the next words in a sequence.
        : param vocab_size : The number of unique words in the data
        : param hidden_size   : The size of your desired RNN
        : param embed_size : The size of your latent embedding
        """

        super().__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.lr = .001

        # TODO: define your trainable variables and/or layers here. This should include an
        # embedding component, and any other variables/layers you require.

        self.tf_embedding_table = tf.keras.layers.Embedding(self.vocab_size, self.embed_size)
        # tf.Variable(tf.random.normal(
        #     [vocab_size, embed_size], stddev=0.01, dtype=tf.float32))

        # initalize layers?
        self.dense = tf.keras.layers.Dense(
            self.hidden_size, activation='leaky_relu')
        self.dense2 = tf.keras.layers.Dense(self.vocab_size, activation='softmax')

        self.seq = tf.keras.Sequential([self.dense, self.dense2])

        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()

    def call(self, inputs):
        """
        You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup or tf.keras.layers.Embedding)
        :param inputs: word ids of shape (batch_size, 2)
        :return: logits: The batch element probabilities as a tensor of shape (batch_size, vocab_size)
        """
        # shape is none for first dimension

        tf_embedding_vectors_0 = self.tf_embedding_table(inputs[:, 0])

        tf_embedding_vectors_1 = self.tf_embedding_table(inputs[:, 1])

        tf_embedding_vectors = tf.concat(
            [tf_embedding_vectors_0, tf_embedding_vectors_1], 1)

        inputs = self.seq(tf_embedding_vectors)

        return inputs

    def generate_sentence(self, word1, word2, length, vocab):
        """
        Given initial 2 words, print out predicted sentence of targeted length.
        (NOTE: you shouldn't need to make any changes to this function).

        :param word1: string, first word
        :param word2: string, second word
        :param length: int, desired sentence length
        :param vocab: dictionary, word to id mapping

        """
        reverse_vocab = {idx: word for word, idx in vocab.items()}
        output_string = np.zeros((1, length), dtype=np.int32)
        output_string[:, :2] = vocab[word1], vocab[word2]

        for end in range(2, length):
            start = end - 2
            output_string[:, end] = np.argmax(
                self(output_string[:, start:end]), axis=1)
        text = [reverse_vocab[i] for i in list(output_string[0])]

        print(" ".join(text))


#########################################################################################

def get_text_model(vocab):
    '''
    Tell our autograder how to train and test your model!
    '''

    # TODO: Set up your implementation of the RNN

    # Optional: Feel free to change or add more arguments!
    model = MyTrigram(len(vocab))

    loss_metric = model.loss

    def perplexity(y_true, y_pred):
        #loss = tf.keras.losses.SparseCategoricalCrossentropy()
        return tf.math.exp(tf.reduce_mean(loss_metric(y_true, y_pred)))

    # TODO: Define your own loss and metric for your optimizer
    # specify if we want to use logit or probs
    
    acc_metric = perplexity

    # TODO: Compile your model using your choice of optimizer, loss, and metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=model.lr),
        loss=loss_metric,
        metrics=[acc_metric],
    )

    return SimpleNamespace(
        model=model,
        epochs=1,
        batch_size=100,
    )


#########################################################################################

def main():

    # TODO: Pre-process and vectorize the data
    # HINT: You might be able to find this somewhere...
    train_data, test_data, vocab = get_data("/Users/mikaylawalsh/Desktop/deep_learning/hw4-mikaylawalsh/data/test.txt",
                                            "/Users/mikaylawalsh/Desktop/deep_learning/hw4-mikaylawalsh/data/train.txt")
    def process_trigram_data(data):
        X = np.array(data[:-1])
        Y = np.array(data[2:])
        X = np.column_stack((X[:-1], X[1:]))
        return X, Y

    X0, Y0 = process_trigram_data(train_data)
    X1, Y1 = process_trigram_data(test_data)

    # TODO: Get your model that you'd like to use
    # args = get_text_model(vocab)

    # TODO: Implement get_text_model to return the model that you want to use.
    args = get_text_model(vocab)

    args.model.fit(
        X0, Y0,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X1, Y1)
    )

    # Feel free to mess around with the word list to see the model try to generate sentences
    words = 'speak to this brown deep learning student'.split()
    for word1, word2 in zip(words[:-1], words[1:]):
        if word1 not in vocab:
            print(f"{word1} not in vocabulary")
        if word2 not in vocab:
            print(f"{word2} not in vocabulary")
        else:
            args.model.generate_sentence(word1, word2, 20, vocab)


if __name__ == '__main__':
    main()
