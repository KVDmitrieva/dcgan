echo "Install gdown"
pip install gdown

mkdir data

echo "Download data"
gdown "https://drive.google.com/u/0/uc?id=1BKByUUHmUKBK1_Shw7mi8VKwEdjO63wT" -O data/data.zip
unzip -q data/data.zip -d data
rm data/data.zip

echo "Done!"