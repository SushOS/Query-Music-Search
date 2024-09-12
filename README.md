# Problem statement

With the recent rise of LLMs, a brilliant startup named Perplexity has surfaced, which replaces the “discrete” nature of Google’s link-based search outputs with a fast and descriptive continuous stream of relevant search results.

Like Google, Spotify has a rigid and flexible search mechanism wherein we, as users, don’t have any control whatsoever. We believe that they deserve much better, and hence, we want to create a personable, intelligent multimodal (text+audio) music search system by making LLMs “understand” music.

Currently, Spotify uses a lexico-semantic search mechanism that is incapable of answering queries like “a song which has the banjo instrument in it” or “a scary song which I can play in the background of my YouTube video”.

However, using deep models always poses the risk of high latencies and high memory usage. We undertook the challenge of making everything efficient, even when using models with over 180 million parameters. The approach will be described below in detail.

# Approach

## Aligning Audio and Text Modalities

When dealing with multimodal data retrieval, we need to address the problem of bridging the gap between different data modalities. The main problem to solve is to develop a model that can understand musical data in a way that is more aligned with a detailed, professional, human-like understanding based on natural language descriptions.
![HTSAT Audio Encoder](https://github.com/shahrushi2003/CLAP/assets/101819411/6321c697-812d-4eb7-9d36-2b2fa610846d)

We surveyed many different audio representation techniques but ended up choosing [CLAP](https://ieeexplore.ieee.org/abstract/document/10095889) by Microsoft in order to solve this alignment problem. CLAP involved training a GPT2-based text encoder and an HTSAT audio encoder to project both modalities to a unified representation space. This solved the alignment problem, helping information flow from both ends.

CLAP is inspired from the CLIP model by OpenAI, which utilised a 2D cross-entropy loss to train both the encoders. This contrastive loss function helps align both the encoders by increasing the similarity between correct image-text pairs and decrease the similarity between incorrect ones.

![GPT2 Architecture](https://github.com/shahrushi2003/CLAP/assets/101819411/00c6d68c-e503-4dac-b629-f5fe7f52cd6a)

![CLAP Loss](https://github.com/shahrushi2003/CLAP/assets/101819411/507e7a8c-7bcd-4fb1-9344-ccab1d0026e3)

CLAP was perfect for our use case because we want to deliver the best search results based on the **textual** query. Moreover, CLAP has been trained extensively on a mixture of multiple datasets and achieved SOTA in 26 audio downstream tasks, including classification, captioning, and, yes, retrieval! The best thing about CLAP is that it allows for zero-shot usage, expanding the space of things we can do without finetuning the model.

## Matryoshka Representation Learning

Another problem to solve is latency and memory usage. This is a very crucial thing for search since a search bar is the entry point to utilising any application, which SHOULDN’T go wrong in any case. Hence, we resorted to a new embedding learning technique called Matryoshka Represenation Learning (a quite clever name!). 

MRL enhances representation learning by introducing flexibility into high-dimensional vectors. While rich in information, traditional representations lack adaptability and incur high computational costs for downstream tasks. MRL addresses this by optimising $O(log(d))$ nested vectors within a single representation, akin to Russian nesting dolls (Matryoshka), allowing for coarse-to-fine granularity. These Matryoshka Representations adapt seamlessly across tasks and environments without additional training or deployment overhead. With MRL, tasks like classification and retrieval achieve up to $14×$ faster speeds without sacrificing accuracy, making it a game-changer for large-scale ML systems. Using MRL led to substantial gains when we used it with CLAP for music information retrieval.

![MRL Loss Function](https://github.com/shahrushi2003/CLAP/assets/101819411/b58d3b7e-905c-4862-9ae9-c369691bcfbe)

The MRL loss is a simple idea which involves repeating the loss function over a variety of different dimensions. This helps concentrate global information into the lower dimensions as compared to higher dimensions. This property will be put to good use in the next section.

## Adaptive Retrieval

A very simple yet powerful use-case that extends the simple web-scale search example above is to use MRL for retrieval *adaptively*, i.e.

1. Shortlist a large pool of 1000 song candidates with small *d*=32
2. Reorder/Rerank the shortlisted pool of songs with large *d*=512
3. Repeat the same for different dimensions.

The reason this is easy with MRL is that the embedding for shortlisting is just a chopped-off version of the embedding for re-ordering! We thus don’t need to store multiple large copies of full-dimensional embeddings or query multiple search indices built on different dimensions (very slow).

# Dataset

We used the [MusicCaps](https://huggingface.co/datasets/google/MusicCaps) dataset for this experiment.

# Major Innovations

We introduced several major innovations in the field of music search and retrieval.

Firstly, we used the CLAP model by Microsoft for aligning Audio and Text Modalities. This allowed us to create a unified representation space for both the audio and text data, thus making our search more accurate and efficient.

Secondly, we implemented Matryoshka Representation Learning (MRL) which enhanced our representation learning by introducing flexibility into high-dimensional vectors. This approach helped us to reduce latency and memory usage, which are crucial for any search application.

Lastly, we introduced an Adaptive Retrieval method which uses MRL for retrieval adaptively. This helped us to shortlist and reorder songs based on different dimensions, thus making our retrieval process more efficient and accurate, without costing more money!

# Contributions

# Results

We finetuned our model’s projection layers on the MusicCaps dataset. The below loss curves show that the model is learning well. 

![image](https://github.com/shahrushi2003/CLAP/assets/101819411/c46f6934-9984-4afc-b906-d4e0f4a75eb8)

# Literature Review and Other Resources

### Papers

1. [Attention is All You Need](https://arxiv.org/abs/1706.03762)
2. [FAISS](https://arxiv.org/abs/2401.08281)
3. [CLAP: Learning Audio Concepts from Natural Language Supervision](https://ieeexplore.ieee.org/abstract/document/10095889)
4. [A-JEPA: Joint-Embedding Predictive Architecture Can Listen](https://arxiv.org/abs/2311.15830)
5. [Mistral 7B](https://arxiv.org/abs/2310.06825)
6. [MRL Paper](https://arxiv.org/abs/2205.13147)

### Blogs

1. [How to make LLMs listen?](https://paul.mou.dev/posts/2023-12-31-listening-with-llm/#overview)
2. [Hacker Llama’s The Random Transformer](https://osanseviero.github.io/hackerllama/blog/posts/random_transformer/) - An in-depth blog about transformers, attention and LLMs.
3. [Chat with Your Audio Locally: A Guide to RAG with Whisper, Ollama, and FAISS](https://medium.com/@ingridwickstevens/chat-with-your-audio-locally-a-guide-to-rag-with-whisper-ollama-and-faiss-6656b0b40a68)
4. [Retrieval Augmented Generation on audio data with LangChain and Chroma](https://www.assemblyai.com/blog/retrieval-augmented-generation-audio-langchain/)
5. [Leveraging LLMs with Information Retrieval: A Simple Demo](https://towardsdatascience.com/leveraging-llms-with-information-retrieval-a-simple-demo-600825d3cb4c#bypass)
6. [LangChain's blog on Query Construction](https://blog.langchain.dev/query-construction/)
7. [A blog on MRL](https://aniketrege.github.io/blog/2024/mrl/)

### Others

1. [Perplexity](https://www.perplexity.ai/) - Our inspiration.
2. [LangChain](https://www.langchain.com/)
3. [LlamaIndex](https://www.llamaindex.ai/)
