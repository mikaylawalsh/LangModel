import tensorflow as tf
import numpy as np
from preprocess import get_data
from types import SimpleNamespace


class MyRNN(tf.keras.Model):

    ##########################################################################################

    def __init__(self, vocab_size, rnn_size=128, embed_size=64):
        """
        The Model class predicts the next words in a sequence.
        : param vocab_size : The number of unique words in the data
        : param rnn_size   : The size of your desired RNN
        : param embed_size : The size of your latent embedding
        """

        super().__init__()

        self.vocab_size = vocab_size
        self.rnn_size = rnn_size # window size 
        self.embed_size = embed_size

        self.lr = .001

        self.tf_embedding_table = tf.keras.layers.Embedding(self.vocab_size, self.embed_size)

        self.dense = tf.keras.layers.Dense(
            self.vocab_size, activation='leaky_relu') # make even bigger? 
        self.dense2 = tf.keras.layers.Dense(self.vocab_size, activation='softmax')

        self.seq = tf.keras.Sequential([self.dense, self.dense2])

        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()

        self.LSTM = tf.keras.layers.LSTM(self.embed_size, return_sequences=True) # what goes in here
       
        ## TODO:
        ## - Define an embedding component to embed the word indices into a trainable embedding space.
        ## - Define a recurrent component to reason with the sequence of data. 
        ## - You may also want a dense layer near the end...    

    def call(self, inputs):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup or tf.keras.layers.Embedding)
        - You must use an LSTM or GRU as the next layer.
        """

        # inputs = (None,)

        embed = self.tf_embedding_table(inputs) # (None, 128, 64)

        output = self.LSTM(embed, initial_state = None) # (None, 64)

        output = self.seq(output) # (None, self.vocab_size)

        return output

    ##########################################################################################

    def generate_sentence(self, word1, length, vocab, sample_n=10):
        """
        Takes a model, vocab, selects from the most likely next word from the model's distribution.
        (NOTE: you shouldn't need to make any changes to this function).
        """
        reverse_vocab = {idx: word for word, idx in vocab.items()}

        first_string = word1
        first_word_index = vocab[word1]
        next_input = np.array([[first_word_index]])
        text = [first_string]

        for i in range(length):
            logits = self.call(next_input)
            logits = np.array(logits[0,0,:])
            top_n = np.argsort(logits)[-sample_n:]
            n_logits = np.exp(logits[top_n])/np.exp(logits[top_n]).sum()
            out_index = np.random.choice(top_n,p=n_logits)

            text.append(reverse_vocab[out_index])
            next_input = np.array([[out_index]])

        print(" ".join(text))


#########################################################################################

def get_text_model(vocab):
    '''
    Tell our autograder how to train and test your model!
    '''

    ## TODO: Set up your implementation of the RNN

    ## Optional: Feel free to change or add more arguments!
    model = MyRNN(len(vocab))

    ## TODO: Define your own loss and metric for your optimizer
    loss_metric = model.loss 
    def perplexity(y_true, y_pred):
        #loss = tf.keras.losses.SparseCategoricalCrossentropy()
        return tf.math.exp(tf.reduce_mean(loss_metric(y_true, y_pred)))
    acc_metric = perplexity

    ## TODO: Compile your model using your choice of optimizer, loss, and metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=model.lr), 
        loss=loss_metric, 
        metrics=[acc_metric],
    )

    return SimpleNamespace(
        model = model,
        epochs = 1,
        batch_size = 100,
    )



#########################################################################################

def main():

    ## TODO: Pre-process and vectorize the data
    ##   HINT: Please note that you are predicting the next word at each timestep, so you want to remove the last element
    ##   from train_x and test_x. You also need to drop the first element from train_y and test_y.
    ##   If you don't do this, you will see very, very small perplexities.
    ##   HINT: You might be able to find this somewhere...
    train_id, test_id, vocab = get_data("../data/train.txt", "../data/test.txt")

    train_id = np.array(train_id)
    test_id  = np.array(test_id)
    X0, Y0 = train_id[:-1], train_id[1:] # (1465613,) (1465613,)
    X1, Y1  = test_id[:-1],  test_id[1:] # (361911,) (361911,)

    window_size = 20
    
    # is it okay to hard code this in
    X0, Y0 = X0[:-13], Y0[:-13]
    X1, Y1 = X1[:-11], Y1[:-11]

    X0 = tf.reshape(X0, [-1, window_size])
    Y0 = tf.reshape(Y0, [-1, window_size])
    X1 = tf.reshape(X1, [-1, window_size])
    Y1 = tf.reshape(Y1, [-1, window_size])

    print(X0.shape)
    print(Y0.shape)
    ## TODO: Get your model that you'd like to use
    args = get_text_model(vocab) # len(vocab) = 4962

    args.model.fit(
        X0, Y0,
        epochs=args.epochs, 
        batch_size=args.batch_size, # 100
        validation_data=(X1, Y1)
    )

    ## Feel free to mess around with the word list to see the model try to generate sentences
    for word1 in 'speak to this brown deep learning student'.split():
        if word1 not in vocab: print(f"{word1} not in vocabulary")            
        else: args.model.generate_sentence(word1, 20, vocab, 10)

if __name__ == '__main__':
    main()
