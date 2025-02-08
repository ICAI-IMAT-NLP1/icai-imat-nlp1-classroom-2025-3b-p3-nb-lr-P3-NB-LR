from typing import List, Dict
from collections import Counter
import torch

try:
    from src.utils import SentimentExample, tokenize
except ImportError:
    from utils import SentimentExample, tokenize


def read_sentiment_examples(infile: str) -> List[SentimentExample]:
    """
    Reads sentiment examples from a file.

    Args:
        infile: Path to the file to read from.

    Returns:
        A list of SentimentExample objects parsed from the file.
    """
 
    # TODO: Open the file, go line by line, separate sentence and label, tokenize the sentence and create SentimentExample object
    examples: List[SentimentExample] = []
    with open(infile, "r") as file:
        for line in file:
            sentence, label = line.strip().split("\t")
            words = tokenize(sentence)
            example = SentimentExample(words, int(label))
            examples.append(example)

    return examples


def build_vocab(examples: List[SentimentExample]) -> Dict[str, int]:
    """
    Creates a vocabulary from a list of SentimentExample objects.

    The vocabulary is a dictionary where keys are unique words from the examples and values are their corresponding indices.

    Args:
        examples (List[SentimentExample]): A list of SentimentExample objects.

    Returns:
        Dict[str, int]: A dictionary representing the vocabulary, where each word is mapped to a unique index.
    """
    """
     The function should analyze the SentimentExample objects, 
     extract all unique words across all examples, 
     and then assign each word a unique index. 
    """
    # TODO: Count unique words in all the examples from the training set
    vocab: Dict[str, int] = {}
    idx = 0
    for example in examples:
        for word in example.words:
            if word not in vocab:
                vocab[word] = idx
                idx += 1
            # else:
            #     vocab[word] += 1
    return vocab


def bag_of_words(
    text: List[str], vocab: Dict[str, int], binary: bool = False
) -> torch.Tensor:
    """
    Converts a list of words into a bag-of-words vector based on the provided vocabulary.
    Supports both binary and full (frequency-based) bag-of-words representations.

    Args:
        text (List[str]): A list of words to be vectorized.
        vocab (Dict[str, int]): A dictionary representing the vocabulary with words as keys and indices as values.
        binary (bool): If True, use binary BoW representation; otherwise, use full BoW representation.

    Returns:
        torch.Tensor: A tensor representing the bag-of-words vector.
    """

    """
    Specifically, you will convert lists of words into bag-of-words vectors using the vocabulary you’ve built.
    
    A Bag of Words (BoW) representation involves counting the frequency of each word in the vocabulary
    within a given text or, alternatively, simply marking their presence or absence (binary representation).

    Your function should be capable of generating both these types of representations based on the binary
    parameter.
      If binary is True, the function should return a binary BoW representation, where each element
    is 1 if the corresponding word is present in the text and 0 otherwise. 
    If binary is False, the function
    should return a full BoW representation, where each element is the frequency of the corresponding word in the text.

    Hint #2:
    Remember to exclude words that are not present in the vocabulary.
    """
    # TODO: Converts list of words into BoW, take into account the binary vs full
    bow: torch.Tensor = torch.zeros(len(vocab))

    for word in text:
        if binary:
            if word in vocab:
                bow[vocab[word]] = 1
            else:
                bow[-1] = 0
        else:
            if word in vocab:
                bow[vocab[word]] = text.count(word)
            else:
                bow[-1] = 0
            
    return bow

