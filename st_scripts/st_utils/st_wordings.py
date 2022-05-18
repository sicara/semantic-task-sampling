import streamlit as st

WORDINGS = {
    "app_intro": """
    Since 2018, 98 papers have used miniImageNet as a benchmark. 205 papers have used tieredImageNet. \n
    If you've done any academic research on Few-Shot Image Classification, it is likely that you have used them yourself. 
    You have probably tested some model on hundreds of randomly generated Few-Shot Classification tasks from miniImageNet or tieredImageNet. \n
    But do you know what these tasks look like? \n
    Have you ever wondered what kind of discrimination your model was asked to perform? \n
    If you have not, tis not too late.
    If you have, you're in the right place.
    """,
    "uniform_tasks": """
    Few-Shot Learning benchmarks such as miniImageNet or tieredImageNet evaluate methods on hundreds of Few-Shot Classification tasks. 
    These tasks are sampled uniformly at random from the set of all possible tasks. \n
    This induces a huge bias towards tasks composed of classes that have nothing to do with one another. 
    Classes that you would probably never have to distinguish in any real use case. \n
    See it for yourself. 
    """,
    "after_uniform_task": """
        If this task looks even remotely like a task you would need to solve ever, please [reach out to me](https://twitter.com/EBennequin).

        Because of this shift between those academic benchmark and real life applications of Few-Shot Learning, the performance of a method on those benchmarks is only a distant proxy of its performance on real use cases.
        """,
    "explain_semantic_sampling": """
        The classes of tieredImageNet are part of the WordNet graph. \n
        We use this graph to define a semantic distance between classes. \n
        We use this semantic distance to define the coarsity of a task as the mean square distance between the classes constituting the task. \n
        We use this coarsity to sample tasks made of classes that are semantically close to each other. \n
        Play with the coarsity slider. See what kind of tasks we can sample.
        """,
    "semantic_graph": """
    It seems that when you choose a low coarsity, you get a task composed of classes that are semantically close to each other.
    For instance, with the lowest coarsity (8.65), you get the task of discriminating between 5 breeds of dogs.
    On the other hand, when you increase the coarsity, the classes seem to get more distant from one another. \n
    An other way to see this distance is on the WordNet graph. Below you can see the subgraph of WordNet spanned by the classes of tieredImageNet.
    The blue dots are the classes. Highligted in pink, you have the classes that constitute the task you selected.
    Hover any node to see the associated words. \n
    The smaller the coarsity, the closer the classes in the graph.
    """,
    "app_conclusion": """
    This little dashboard is meant to highlight that common Few-Shot Learning benchmarks are strongly biased towards tasks composed of classes that have very distant from each other. \n
    At Sicara, we have seen a wide variety of industrial applications of Few-Shot Learning, but we never encountered a scenario that can be approached by benchmarks presenting this type of bias.
    In fact, in our experience, most applications involve discriminating between classes that are semantically close to each other: plates from plates, tools from tools, carpets from carpets, parts of cars from parts of cars, etc. \n
    There are other benchmarks for fine-grained classifications. And it's OK that some benchmarks contain tasks that are very coarse-grained. 
    But today, tieredImageNet and miniImageNet are wildly used in the literature, and it's important to know what's in there,
    and how to restore the balance. \n

    If you want to know more, check out our paper 
    [Few-Shot Image Classification Benchmarks are Too Far From Reality: Build Back Better with Semantic Task Sampling](https://arxiv.org/abs/2205.05155)
    (presented at the Vision Datasets Understanding Workshop at CVPR 2022).
    """,
}

st_divider = lambda: st.markdown("---------")
