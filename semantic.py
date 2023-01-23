# Import spacy
import spacy

# Load "en_core_web_md" and store in variable "nlp".
nlp = spacy.load("en_core_web_md")

# Example similarity: cat, monkey, banana
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print(f"""\nSimilarity of {word1}, {word2} and {word3}:
{word1} {word2} - {word1.similarity(word2)}
{word3} {word2} - {word3.similarity(word2)}
{word3} {word1} - {word3.similarity(word1)}""")

# Use nested for loops to undertake a comparison of the words.
tokens = nlp('cat apple monkey banana')
print("\nSimilarity of cat, apple, monkey and banana:")
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

"""
Note #1
The similarity between cat and monkey is higher than between monkey and banana, the reason could be that they
are both words of animal. On the other hand, monkey and banana are different category. Although it is also 
interesting to see that there is a bigger similarity between monkey and banana than between cat and banana. 
It means that the fact that we would naturally associate bananas with monkeys more than we would associate cats with
bananas is being shown here.
"""

# My example: Similarity of taxi, bus and road
word1 = nlp("taxi")
word2 = nlp("bus")
word3 = nlp("road")
print(f"""
My example: Similarity of {word1}, {word2} and {word3}:
{word1} {word2} - {word1.similarity(word2)}
{word3} {word2} - {word3.similarity(word2)}
{word3} {word1} - {word3.similarity(word1)}\n""")

# Use nested for loops to undertake a comparison of the words.
tokens = nlp('taxi bus road passenger')
print("My example: Similarity of taxi, bus, road and passenger:")
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

# Similarity between longer sentences
sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my cat on my car",
             "I\'d like my boat back",
             "I will name my dog Diana"]
model_sentence = nlp(sentence_to_compare)
print("\nSimilarity between sentences:")
# Use for loop to display the similarity of the sentences.
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

"""
NOTE #2 Differences between models sm and md.
After running the example.py with both models "en_core_web_sm" and "en_core_web_md", results shown that 
the similarity from "en_core_web_sm" are much lower as opposed to when using the "en_core_web_md". It is because the 
size of sm is smaller than md (12mb vs 40mb). Also, warning message displayed that the model used is not loading 
vectors. There are 0 keys, 0 unique vectors (0 dimensions) in the sm model but 514k keys, 20k unique vectors 
(300 dimensions) in the md model. It maybe the reason of produce similarity judgments that are not useful.
"""