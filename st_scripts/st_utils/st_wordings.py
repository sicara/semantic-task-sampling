import streamlit as st

from st_scripts.st_utils.st_constants import SECONDARY_APP_COLOR

WORDINGS = {
    "app_intro": """
        Today we're going to talk about the two most widely used Few-Shot-Learning benchmarks: ***tiered*ImageNet** and ***mini*ImageNet**. 
        Since 2018, these two benchmarks combined have been used more than **300 times** in peer-reviewed papers. \n
        The standard evaluation process in Few-Shot Learning is to sample hundreds of small few-shot tasks from the test set, 
        compute the accuracy of the model on each task, and report the mean and standard deviation of these accuracies. 
        But **never do we look at the tasks individually**. Only the aggregated results. \n
        So **what the hell is in these tasks?** Exactly what kind of problem did these hundreds of research papers try to solve? 
        Does it reflects **real-world problems**?
    """,
    "uniform_tasks": """
        Few-Shot Learning benchmarks such as *mini*ImageNet or *tiered*ImageNet evaluate methods on hundreds of Few-Shot Classification tasks. 
        These tasks are sampled **uniformly at random** from the set of all possible tasks. \n
        This induces a **huge bias** towards tasks composed of classes that have nothing to do with one another. 
        Classes that you would probably never have to distinguish in any real use case. \n
        See it for yourself. 
    """,
    "after_uniform_task": """
        If this task looks even remotely like a task you would need to solve ever, please [reach out to me](https://twitter.com/EBennequin).
        But most likely, these 5 classes describe concepts that are **so far from each other that it makes the task just absurd**. \n
        In fact, you can spam the above buttons if you like, you probably won't ever get a task with 5 classes that are close enough for the task to make sense. 
        The problem is that on a dataset like *tiered*ImageNet and with uniform sampling, **the probability of getting a quintuplet of close classes is basically zero**. \n
        In real life however, **you'll most likely want to distinguish between concepts that are somewhat relevant to one another**. 
        Because of this shift between those academic benchmark and real life applications of Few-Shot Learning, the performance of a method on those benchmarks **is only a distant proxy** of its performance on your real-life use cases.
    """,
    "explain_semantic_sampling": """
        The classes of *tiered*ImageNet are part of the WordNet semantic graph. 
        We can use this graph to define a **semantic distance** between classes, 
        and then we can define a measure for the **coarsity** of a task, as the mean square semantic distance between the classes that compose it. \n
        Thanks to this measure, we can confirm on the opposite figure that using **uniform task sampling** (as is usually done in the literature), 
        we can never get a task composed of classes that are close to each other. \n
        **But these tasks are not unreachable!** We can actually force our task sampler to sample together classes with a low coarsity.
        That's the pink histogram. The pink histogram makes the impossible possible. 
        It can reach coarsities that the blue histogram would never even dream of. \n
    """,
    "introduce_slider": """
        OK, but what does it really mean for a task to have a low coarsity? 
        Play with the slider to see what kind of tasks we can sample.
    """,
    "semantic_graph": """
        It seems that when you choose a **low coarsity**, you get a task composed of classes that are **semantically close** to each other.
        For instance, with the lowest coarsity (8.65), you get the task of discriminating between 5 breeds of dogs.
        On the other hand, when you increase the coarsity, the classes seem to get **more distant** from one another. \n
        An other way to see this distance is **directly on the WordNet graph**. Below you can see the subgraph of WordNet spanned by the classes of *tiered*ImageNet.
        The blue dots are the classes. Highligted in pink, you have the classes that constitute the task you selected.
        Hover any node to see the associated words. \n
        The smaller the coarsity, the closer the classes in the graph.
    """,
    "app_conclusion": """
        This little dashboard is meant to highlight that **common Few-Shot Learning benchmarks are strongly biased** towards tasks composed of classes that are **very distant** from each other. \n
        At Sicara, we have seen a wide variety of industrial applications of Few-Shot Learning, but **we never encountered a scenario that can be approached by benchmarks presenting this type of bias**.
        In fact, in our experience, most applications involve discriminating between classes that are semantically close to each other: plates from plates, tools from tools, carpets from carpets, parts of cars from parts of cars, etc. \n
        There are other benchmarks for fine-grained classifications. And it's OK that some benchmarks contain tasks that are very coarse-grained. 
        But today, *tiered*ImageNet and *mini*ImageNet are wildly used in the literature, and **it's important to know what's in there,
        and how to restore the balance**. \n
        If you want to know more about the biases of classical Few-Shot Learning benchmarks and about semantic task sampling, **check out our paper 
        [Few-Shot Image Classification Benchmarks are Too Far From Reality: Build Back Better with Semantic Task Sampling](https://arxiv.org/abs/2205.05155)**
        (presented at the Vision Datasets Understanding Workshop at CVPR 2022).
    """,
    "about": """
        I'm [Etienne Bennequin](https://ebennequin.github.io/) and I'm doing a PhD to try and bridge the gap between 
        academic research in Few-Shot Learning and the industrial applications that we encounter on a daily basis at Sicara. \n
        What's [Sicara](https://www.sicara.fr/), you ask? It's a French company that builds tailored Data Science and Data Engineering solutions for its clients. 
        In my opinion it's a great place to work and learn about cool stuff like Streamlit, the library I used to build this website in full Python. \n
        If what you've seen here, in [the paper](https://arxiv.org/abs/2205.05155), or in [the code](https://github.com/sicara/semantic-task-sampling) 
        interests you, please reach out to me, I'm always trying to learn from other researchers.
    """,
}

st_divider = lambda: st.markdown(f"""<hr style="width:15%">""", unsafe_allow_html=True)
