# Attention-based Neural-Machine-Translation

Neural Machine Translation is an approach to machine translation that uses Artificial
Neural Networks to predict likelihood of a sequence of words, often trained in an 
end-to-end fashion and has the ability to generalize well to very long word sequences.
Formally it can be defined as a NN that models the conditional probability p(y x) of
translating a sentence x1...xn into y1...yn.

# POC 1
Using transformers for Neural Machine Translation.
The high level idea behind transformers is to design the attention mechanism: the ability
to set attention at different parts of a sequence for calculating a sequence representation.
It presents several difference with respect to past sequential architectures.
 
 1. No supositions about temporal/spacial relationship in between data.Therefore, it
 doesn´t think about them about a sequence per se. 
 
 2. Layer outputs can be calcualted in parallel
 
 3. Learn long dependencies 



# Datasets

The project attempts to use the Spanish Dataset.
Other research include
French to English
German to English
Portuguese to English 
Russian to English

# POC 2?

Research has take the attention-based mechanism into this field. The exploration implements
the paper based on two simple and effective classes of attentional mechanism:

* global approach : attends to all source words
* local: looks at one sentence. Conceived as an iteration from the existing previous
research literature. The difference is that it is differenciable almost everywhere, computationally
less expensive. This is based moslty in LSTM architectures.
