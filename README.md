<div align=center>
<h1>SRPFN: One Sequential Recommendation Model Pretrained from Synthetic Priors Predicts Multiple Datasets</h1>

![GitHub Repo stars](https://img.shields.io/github/stars/woooosung/SRPFN)

<div>
    <a href="https://scholar.google.com/citations?user=-3qk_osAAAAJ" target="_blank">Woosung Kang</a>,
    <a href="https://scholar.google.com/citations?user=aKpwctQAAAAJ&hl=ko" target="_blank">Jiwon Jeong</a>,
    <a href="https://www.linkedin.com/in/jonghyeok-shin-b8ab06343/" target="_blank">Jonghyeok Shin</a>,
    <a href="https://www.jeongwhanchoi.com" target="_blank">Jeongwhan Choi</a>,
    <a href="https://sites.google.com/view/noseong" target="_blank">Noseong Park</a>,
    <div>
    Korea Advanced Institute of Science and Technology (KAIST)
    </div>
</div>
</div>

---

Official implementation of **SRPFN**, accepted at **KDD 2026**.

> **TL;DR:** A single model pretrained on synthetic priors predicts multiple sequential recommendation datasets without dataset-specific training.

---

## Environment

You can create the environment as follows:

```bash
conda create -n srpfn python=3.10 pip -y
conda activate srpfn
pip install -r requirements.txt
conda install -n srpfn -c conda-forge graph-tool
```

All experiments were conducted on a single NVIDIA RTX A6000 GPU.

---

## Training

Edit `train_config.json` if needed, then run:

```bash
bash shell/train.sh
```

`shell/train.sh` uses the Python executable from the `srpfn` conda environment by
default. Training logs are written under `logs/train/`, and checkpoints are saved
to the `environment.save_path` value in `train_config.json`.

---

## Inference

To evaluate datasets in the `data/` folder, specify the dataset name and evaluation protocol in `eval_config.json`.

Run inference with:

```bash
bash shell/eval.sh
```