[![](https://img.shields.io/badge/Python-3.10.6-informational)](https://www.python.org/)
[![](https://img.shields.io/github/license/DominikBeese/FairGer?label=License)](/LICENSE)
# FairGer
Data and code for the paper ["FairGer: Using NLP to Measure Support for Women and Migrants in 155 Years of German Parliamentary Debates"](https://arxiv.org/abs/2210.04359) by Dominik Beese, Ole Pütz, and Steffen Eger, 2022.

<img src="https://user-images.githubusercontent.com/111588769/194097201-571564b8-f758-4daf-bb60-8bde8803c3cc.png" alt="Solidarity Distribution per Decade" width="550px">


## Content
The repository contains the following elements:
 * 📂 [Data](/Data)
   * 📂 [Datasets](/Data/Datasets) of sentences containing a woman or migrant keyword
   * 📂 [Human Annotated Data](/Data/Human%20Annotated%20Data) regarding solidarity towards women and migrants
   * 📂 [Model Predicted Data](/Data/Model%20Predicted%20Data) with predictions of our ensemble for all [Datasets](/Data/Datasets)
 * 📂 [Code](/Code) to train and apply models
 * 📂 [Analysis](/Analysis) code to generate the plots

See [DominikBeese/DeuParl-v2](https://github.com/DominikBeese/DeuParl-v2) for the full dataset of plenary protocols from the German _Reichstag_ and _Bundestag_.


## Citation
```
@article{FairGer,
          title = "{F}air{G}er: Using {NLP} to Measure Support for Women and Migrants in 155 Years of German Parliamentary Debates",
         author = "Dominik Beese and Ole P{\"u}tz and Steffen Eger",
           year = "2022",
         eprint = "2210.04359",
  archivePrefix = "arXiv",
   primaryClass = "cs.CL"
}
```
> **Abstract:** We measure support with women and migrants in German political debates over the last 155 years. To do so, we (1) provide a gold standard of 1205 text snippets in context, annotated for support with our target groups, (2) train a BERT model on our annotated data, with which (3) we infer large-scale trends. These show that support with women is stronger than support with migrants, but both have steadily increased over time. While we hardly find any direct anti-support with women, there is more polarization when it comes to migrants. We also discuss the difficulty of annotation as a result of ambiguity in political discourse and indirectness, i.e., politicians' tendency to relate stances attributed to political opponents. Overall, our results indicate that German society, as measured from its political elite, has become fairer over time.
