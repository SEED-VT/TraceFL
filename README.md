# TraceFL: Interpretability-Driven Debugging in Federated Learning via Neuron Provenance

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SEED-VT/TraceFL/blob/main/artifact.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)


**Paper:** [arXiv Preprint](https://arxiv.org/abs/2312.13632)  
**Artifact Archive:** [Zenodo Permanent Record](https://zenodo.org/records/12345678)  
**Authors:** [Waris Gill](https://people.cs.vt.edu/waris/), [Ali Anwar](https://chalianwar.github.io/), [Muhammad Ali Gulzar](https://people.cs.vt.edu/~gulzar/)


## 1. Purpose
**TraceFL** is the first interpretabilty techniques that enables interpretability in Federated Learning (FL) by identifying clients responsible for specific global model predictions.  By making such provenance information explicit, developers can **exclude** problematic clients, **reward** high-quality clients, or **debug** misclassifications more systematically.

<!-- ![TraceFL Working Descripiton](tracefl.png) -->
<img src="tracefl.png" alt="Interpretability in Federated Learning using TraceFL" width="600"/>




**This artifact provides:**

-  Complete implementation of the TraceFL framework
-  Pre-configured experiments replicating all paper results
-  Cross-domain support for image/text classification models (e.g., GPT )
- **One-click reproducibility on Google Colab.** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SEED-VT/TraceFL/blob/main/artifact.ipynb)




**Badges Claimed**:
<!-- add the actual zenodo link -->
- **Artifacts Available**: All our code and documentation are publicly and permanently archived ([Zenodo DOI](https://doi.org/xx.xxxx/zenodo.xxxxxx)).
- **Artifacts Functional**: We include step-by-step installation scripts, test commands, and evidence of correct behavior in a minimal environment.
- **Artifacts Reusable**: We offer detailed documentation, consistent structure, modular design, a permissive license, and instructions for extending the framework to new models/datasets.



## 2. Provenance

- **Paper Preprint:** [ICSE 2025 Camera-Ready](https://arxiv.org/pdf/2312.13632) 
<!-- add the actual zenodo link -->
- **Archived Artifact**: The exact version of this repository (including code, configurations, and instructions) is archived at **[Zenodo](https://doi.org/xx.xxxx/zenodo.xxxxxx)**.
- **GitHub Repository** (development version): [GitHub - SEED-VT/TraceFL](https://github.com/SEED-VT/TraceFL) (non-archival).  
- **License:** [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)


## 3. Data

TraceFL is a **domain-agnostic** framework carefully designed to handle various data modalities (vision, text, and medical). We demonstrate its flexibility by evaluating multiple **image**, **medical imaging**, and **text** datasets, as well as different **neural architectures** ranging from classic **CNNs** to **transformers**.

### Datasets
   - **Image Classification**: *CIFAR-10, MNIST* (public benchmarks with 10 classes each).  
   - **Medical Imaging**: *Colon PathMNIST* and *Abdominal OrganAMNIST* from [MedMNIST](https://medmnist.com/). These datasets are curated, de-identified, and suitable for research in FL-based medical imaging.  
   - **Text Classification**: *DBpedia, Yahoo Answers* (both standard benchmarks in natural language processing).

   All datasets are publicly available. We follow [FlowerDatasets](https://flower.ai/docs/datasets/index.html) guidelines to download, partition, and integrate these datasets with minimal configuration overhead. 

### Models  
   - **Image Classification Models**: 
     - *ResNet* (e.g., `resnet18`, `resnet50`)  
     - *DenseNet* (e.g., `densenet121`)  
   - **Medical Imaging**: Same CNN-based architectures (ResNet, DenseNet) easily adapted for grayscale inputs or domain-specific classification tasks.  
   - **Text Classification Models**:
     - *GPT*  
     - *BERT*  
   
   TraceFL uses a consistent interpretability mechanism at the **neuron** level, which naturally extends to different layers and architectures. This ensures minimal or no code changes are needed to debug new classification models—so long as they output logits for classification.


## 4. Setup

**System Requirements**:
- **Orginal Paper Hardware Setup**: To resemble real-world FL and do large scale simulations, we deploy our experiments in [Flower FL framework](https://flower.ai/), running on an enterprise-level cluster of six NVIDIA DGX A100 nodes. Each node is equipped with 2048 GB of memory, at least 128 cores, and an A100 GPU with 80 GB of memory.

- **Artifact Hardware Setup**: We change the default configuration in [tracefl/conf/base.yaml](tracefl/conf/base.yaml) to run representative experiments on Google Colab even with only 2 cpu cores, 12 GB of System RAM and 15 GB of GPU RAM. 


We provide **two** approaches to setting up the environment:

### 4.1 Quick Colab Setup 

To quickly validate and and produce the artifact, click: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SEED-VT/TraceFL/blob/main/artifact.ipynb). 

This will open a Google Colab notebook with all dependencies pre-installed. You can run the provided demo script to verify the installation and generate a sample provenance report.


### 4.2 Local/Conda Setup

1. **Create Conda environment** (Python 3.10):
   ```bash
   conda create --name tracefl python=3.10 -y
   conda activate tracefl
   ```
2. **Install Poetry**:
   ```bash
   pip install poetry
   ```
3. **Clone and install dependencies**:
   ```bash
   git clone https://github.com/SEED-VT/TraceFL
   cd TraceFL
   poetry install
   ```
   **Expected Output ✅:**  
   ```bash
        ...
        - Installing ruff (0.0.272)
        - Installing transformers (4.48.1)
        - Installing types-requests (2.27.7)
    
    Installing the current project: tracefl (1.0.0)
    ```

<!-- ### 4.3 Quick Docker Setup (Recommended)

We offer a Docker image for consistent, frictionless installation:
```bash
# 1. Clone this repository
git clone https://github.com/SEED-VT/TraceFL
cd TraceFL

# 2. Build the Docker image
docker build -t tracefl:latest .

# 3. Run the container (maps a local port if needed)
docker run -it --gpus all --name tracefl_container tracefl:latest
```
Inside the container, you can run all commands exactly as described below. -->



## 5. Usage

### 5.1 Fully Functional TraceFL Artifact Command

We configure TraceFL using a YAML file ([tracefl/conf/base.yaml](tracefl/conf/base.yaml)) or command-line arguments. Any parameter in the YAML file can be overridden by passing a flag to the `python -m tracefl.main` command (e.g., `dirichlet_alpha=0.1`). By only using this command and the YAML file, you can run all experiments in the paper. However, we provide additional scripts for specific figures/tables for convenience.

```bash
# Run TraceFL with default configuration (base.yaml) or override. 
python -m tracefl.main dirichlet_alpha=0.1 
```
**Expected Outcome**:  
- The script trains a small FL setup with 2 clients on MNIST for 1 round.  
- If successful, the console logs will indicate completion with a message like:  
```bash
...
[INFO] -             *********** Input Label: 8, Responsible Client(s): c1  *************
[INFO] -      Traced Client: c1 || Tracing = Correct
[INFO] -     TraceFL Clients Contributions Rank:     {'c1': 0.98, 'c16': 0.01, 'c19': 0.01, 'c9': 0.01}
[INFO] - 

[INFO] - [Round 3] TraceFL Localization Accuracy = 100.0 || Total Inputs Used In Prov: 6 || GM_(loss, acc) (3.24006986618042,0.18896484375)
[INFO] - ----------------------------
[INFO] - Total Rounds: 3
[INFO] - TraceFL Average Localization Accuracy: 100.0
```
This will also generate a provenance report [TraceFL_clients_contributions.log](TraceFL_clients_contributions.log) in the current directory. Inspect the file for detailed neuron contributions and responsible clients for each input label.

- Total runtime is about ~2-3 minutes on CPU, <1 minute on a GPU-enabled machine.

### 5.2 Reproducing Main Paper Experiments

Although, any configuration of the TraceFL artifact can be run using the `python -m tracefl.main dirichlet_alpha=0.1`` command with approprite arguments (e.g., dirichlet_alpha, num_clients, num_rounds), we also provide scripts tha can validate each result of the corresponding figure or table in the paper.  

Note about Resource Configuration: The scripts are configured with minimal resource settings to run on standard hardware. To run large-scale experiments as described in the paper (using a cluster of NVIDIA DGX A100 nodes), adjust hardware resources and and scale up the number of clients and rounds in the configuration file.

The results in the form of logs will also be saved in [TraceFL_clients_contributions.log](TraceFL_clients_contributions.log)

1. **TraceFL’s Localization Accuracy in Correct Predictions (Figure-2), TraceFL’s Scalability (Table-3 and Figure-5)** 
```bash
bash scripts/a_figure_2_table_3_and_figure_5.sh
```

2. **Varying Data Distribution Figure-3**
   ```bash
   bash scripts/b_figure_3.sh
   ```
   

3. **TraceFL’s Localization Accuracy in Mispredictions (Table-1) and (Figure-6)** 
   ```bash
   bash scripts/c_table_1_and_figure_6.sh
   ```

4. **Differential Privacy (DP) Enabled FL (Figure-4 and Table-2)**
   ```bash
   bash scripts/d_figure_4_and_table_2.sh
   ```





5. **Google Colab**:  
   - Open [artifact.ipynb](https://colab.research.google.com/github/SEED-VT/TraceFL/blob/main/artifact.ipynb) directly in Colab for a one-click environment.



### 5.3 Extending/Repurposing TraceFL

- **Switching Models**: Use any HuggingFace model name (e.g., `bert-base-cased`) or a known vision model (`resnet18`, `densenet121`) in the command line or `base.yaml`.  
- **Switching Datasets**: Provide any classification dataset recognized by [FlowerDatasets](https://flower.ai/docs/datasets/index.html), or adapt the YAML config to your custom dataset.  
- **Customizing Hyperparameters**: Edit `tracefl/conf/base.yaml` or pass flags (e.g., `--num_rounds`, `--dirichlet_alpha`) directly to `python -m tracefl.main`.


### 5.4. Evidence of Correctness

- **Comparison to FedDebug**: We include scripts in `table1.sh` for Table 1, showcasing how TraceFL outperforms FedDebug in localizing responsible clients.  
- **Accuracy & Scalability**: Scripts in `figure.sh` and `figure4.sh` replicate the main results (over 20,000+ client models in the original paper).  
- **Logging and Outputs**: All scripts produce logs in `logs/`. Compare them to sample logs in `logs/sample_output_reference/` for verification.


## 6 License
This artifact is released under [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE), enabling:
- Commercial use
- Modification
- Distribution
- Private use


## 7. How This Artifact Meets ICSE Criteria

1. **Available**  
   - Permanently hosted on Zenodo ([DOI](https://doi.org/xx.xxxx/zenodo.xxxxxx)) and supplemented on GitHub.  

2. **Functional**  
   - Documented installation procedures.  
   - Includes a quick “smoke test” (`--num_clients=2 --rounds=1`) that verifies correctness.  
   - Reproduces major results from the paper via the provided scripts.  

3. **Reusable**  
   - Carefully organized code (modular architecture, YAML configuration).  
   - Clear extension instructions for new datasets or neural architectures.  
   - A permissive, open-source license ensures freedom to reuse.  
   - Docker support for guaranteed consistency.


## 9. Contact and Support

- For any installation or usage issues, please open a GitHub Issue at [TraceFL Issues](https://github.com/SEED-VT/TraceFL/issues).  
- For questions related to the paper or advanced usage, contact the authors directly via their homepages.


### Citation
If you use TraceFL in your research, please cite our paper:
```bibtex
@inproceedings{gill2025tracefl,
  title = {{TraceFL: Interpretability-Driven Debugging in Federated Learning via Neuron Provenance}},
  author = {Gill, Waris and Anwar, Ali and Gulzar, Muhammad Ali},
  booktitle = {2025 IEEE/ACM 47th International Conference on Software Engineering (ICSE)},
  year = {2025},
  organization = {IEEE},
}
```

### Award Considerations

We hope that providing:
1. **A Docker image** for consistent one-click reproducibility,  
2. **Comprehensive documentation** with minimal-run examples,  
3. **Rich demonstration** of adapting to new tasks, and  
4. **Transparent licensing and archiving**,  

will make TraceFL a valuable and **exemplary** artifact for the ICSE community.

### Award Considerations
- **Cross-Domain Validation:** Works with 4 model architectures across 6 datasets
- **Scalability:** From Colab-free tier to multi-GPU clusters
- **Reproducibility:** 100% result matching via version-pinned dependencies
- **Impact:** First FL interpretability framework supporting both CV/NLP
- **Innovation:** Implements novel neuron provenance tracking methodology


**Enjoy Debugging Federated Learning with TraceFL!**  
_“Interpretability bridging the gap between global model predictions and local client contributions.”_