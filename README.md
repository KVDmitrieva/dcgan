# DCGAN
Model is trained on  [Kaggle](https://www.kaggle.com/datasets/splcher/animefacedataset/data) dataset. 
Run 
```bash
chmod +x setup.sh
./setup.sh
```
to download data. Then run 

```python
python3 main.py -k WANDB_KEY
```
to start trainig.

To test model download checkpoint with `test_setup.sh` and run

```python
python3 test.py -n NUMBER_OF_SAMPLES
```
to get model output.
