# DCGAN
Model is trained on  [Kaggle](https://www.kaggle.com/datasets/splcher/animefacedataset/data) dataset.  You can download it from Kaggle and extract files to `data` dir or simply run
```bash
chmod +x setup.sh
./setup.sh
```
to download data with `gdown`. Then run 

```python
python3 main.py -k WANDB_KEY
```
to start trainig.

To test model download checkpoint with `test_setup.sh` and run

```python
python3 test.py -n NUMBER_OF_SAMPLES
```
to get model output.
