# $ apt-get update -y
# $ apt-get upgrade -y
# $ pip install --upgrade pip
# $ pip3 install -r requirements.txt

tar zxvf data/ontonotes-release-5.0_LDC2013T19.tgz -C ./data
tar zxvf data/conll-formatted-ontonotes-5.0-12.tar.gz -C ./data

conda create --name py27 python=2.7.13
conda init bash
conda activate py27
chmod a+rx skeleton2conll.sh
./skeleton2conll.sh -D data/ontonotes-release-5.0/data/files/data data/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0

conda deactivate

python3 agg.py