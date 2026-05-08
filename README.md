# CRAFT: An Antigen-Specific TCR Generative Model for Conditional de novo Design with Physicochemical 

T-cell receptors (TCRs) are essential to adaptive immunity and mediate the recognition and elimination of malignant or infected cells through antigen-specific interactions. Despite the clinical progress of TCR-based immunotherapies, computational de novo design of antigen-specific TCRs remains diﬀicult. The main diﬀiculties include the complexity of peptide–major histocompatibility complex (pMHC)-dependent recognition, strict physicochemical requirements at the binding interface, and the small amount of paired training data available for model development. Current generative approaches depend largely on one-dimensional sequence statistics and do not capture pMHC presentation context, interfacial physicochemical properties, and native receptor priors within a unified framework. We present conditional reconstruction of antigen fragments for TCRs (CRAFT), a decoupled generative model for pMHC-conditioned de novo design of antigen-specific TCR CDR3β sequences under physicochemical constraints. With large-scale pre-training, CRAFT learns base representations that help reduce the limitations imposed by data scarcity. We further incorporate antigen-derived contextual information to guide antigen-specific TCR CDR3β sequence design. The architecture comprises an antigen encoder (PrimeBERT) that captures the pMHC context and an autoregressive decoder (PrimeTCR) that preserves native receptor priors. The encoder and decoder are coupled through cross-attention and augmented with Atchley factor embeddings, allowing CRAFT to incorporate physicochemical constraints relevant to antigen-conditioned receptor design during sequence decoding. Across multiple evaluations, CRAFT demonstrates competitive sequence naturalness, together with favorable performance in specific binding, and predicted structural plausibility. Overall, CRAFT provides a computational framework for developing personalized tumor immunotherapies. It also holds potential to support clinical translation by enabling the eﬀicient, targeted discovery of candidate TCRs.

![Model Image](model/model.png)

## System requirements

This tool is supported for Linux. The tool has been tested on the following systems:

- Ubuntu Linux release 18.04

`CRAFT package requires only a standard computer with enough RAM and NVIDIA GPUs to support operations:

- RAM: 24 GB *2
- GPU: NVIDIA RTX6000 GPUs

## Installation

To install the required packages for running GRATCR, please use the following command:

`conda create -n <env_name> python==3.12`
`conda activate <env_name>`
`pip install -r requirements.txt`

## Model Pre-training:

To pre-train the PrimeBERT, run:

`accelerate launch PrimeBERT.py --data_path='path_to_epitope.csv' --vocab_path='path_to_allepi.csv' --model_path="path_to_model_file"` 

To pre-train the PrimeGPT, run:

`accelerate launch PrimeGPT.py --data_path='path_to_tcrb.csv' --vocab_path='path_to_allbeta.csv' --model_path="path_to_model_file"`

To train the CRAFT, run:

`accelerate launch CRAFT.py --data_path='path_to_tcr-pair.csv' --tcr_vocab_path='path_to_allbeta.csv' --pep_vocab_path='path_to_allepi.csv' --model_path="path_to_model_file" --bert_path="path_to_PrimeBERT_model_file" --gpt_path="path_to_PrimeGPT_model_file" --mode='train'`

## De NovoTCR Design

`python CRAFT.py --data_path='path_to_tcr-pair.csv' --tcr_vocab_path='path_to_allbeta.csv' --pep_vocab_path='path_to_allepi.csv' --model_path="path_to_model_file" --bert_path="path_to_PrimeBERT_model_file" --gpt_path="path_to_PrimeGPT_model_file" --mode='generate' --result_path='./result.csv'`

## Dataset description

The datasets used in this study are all publicly available. IEDB: https://www.iedb.org/home_v3.php; TCRdb 2.0: https://guolab.wchscu.cn/TCRdb2/; McPAS-TCR:https://friedmanlab.weizmann.ac.il/McPAS-TCR/; VDJdb: https://vdjdb.com/; MIRA: https://www.cancerrxgene.org/downloads/bulk_download}; STCRDab: https://opig.stats.ox.ac.uk/webapps/stcrdab-stcrpred/.


# Acknowledgments

The authors sincerely hope to receive any suggestions from you! If you have any questions, please contact us at [2241900@s.hlju.edu.cn](mailto:2231829@s.hlju.edu.cn).





























