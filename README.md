<!-- # LLM_Comment_Generation
Information-Augmented Prompting for Multi-Intent Comment Generation using Large Language Models

Our paper is available on -->


# De-duplication
This is the implementation of IGMIC ( Information-Augmented Prompting for Multi-Intent Comment Generation using Large Language Models).




## Abstract

In contrast to conventional code comment generation methods that predominantly aim to provide a generic overview of a code snippet, multi-intent comment generation endeavors to produce comments from diverse perspectives, encompassing functionality descriptions and usage instructions.
With the prevalent adoption of Large Language Models (LLMs) in code-centric tasks, these models have been leveraged for tackling the multi-intent comment generation challenge.
Despite their successes, the state-of-the-art LLM-based approaches encounter difficulties in fully leveraging the intricate relationships among intents, code, and comments embedded within demonstration examples. 
To mitigate this issue, we propose IGMIC, a framework that leverages explicit guidance through informative demonstration examples to facilitate LLMs in generating intent-specific comments.
Specifically, IGMIC focuses on automating the extraction of important statements that help derive the code comment specific to an intention. 
This is accomplished by using a search model, which captures associations between paired components effectively while being lightweight to construct.
Then, IGMIC augments each demonstration example with its intent-specific key statements, expecting LLMs to follow similar reasoning steps when generating desired comments.
We conduct extensive experiments to evaluate our approach, and the results demonstrate that our approach outperforms the state-of-the-art baselines by 19.49\%, 22.88\%, and 26.25\% in terms of BLEU, ROUGE-L and METEOR, respectively.


## Get Started
PyTorch 2.3.0 framework with Python 3.8. 
Experiments are conducted on a server equipped with four NVIDIA A800 GPU and 512 GB memory, running on Ubuntu 20.04.2.

OS: Ubuntu 20.04.  
package: PyTorch 2.3.0 framework with Python 3.8. 
The dataset we used are funcom.test, funcom.train, tlcodesum.test and tlcodesum.train.  

