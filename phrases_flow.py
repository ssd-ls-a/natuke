# %%
import pandas as pd
import scispacy
import spacy
from tqdm import tqdm

path = ''


# spacy.prefer_gpu()
nlp = spacy.load("en_ner_bc5cdr_md")
# nlp = spacy.load("en_core_web_trf")
# nlp = spacy.load("en_core_sci_scibert")
# nlp = spacy.load("en_core_sci_lg")
# nlp = spacy.load("en_ner_craft_md")


# %%
texts_df = pd.read_parquet('{}texts_query03-05.parquet'.format(path))
# %%
tqdm.pandas()
texts_df['text'] = texts_df['text'].progress_apply(lambda x: x.replace('\n', ' '))

# %%
def phrases_slicing(x):
    # phrases = []
    # slices = round(len(x)/512)
    # for i in range(slices):
    #     if i + 1 == slices:
    #         phrases.append(x[i*512:])
    #     else:
    #         phrases.append(x[i*512:(i+1)*512])

    doc = nlp(x)

    return [sent.text for sent in doc.sents]   

# %%
texts_df['phrases'] = texts_df['text'].progress_apply(phrases_slicing)
print(texts_df)

# %%
texts_df.to_parquet("{}pdf-phrases_24-04.parquet".format(path))




