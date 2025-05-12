# export  OPENSEARCH_INITIAL_ADMIN_PASSWORD=GAIR-scrl-1
# nohup sudo -u opensearch /usr/share/opensearch/bin/opensearch > /inspire/hdd/ws-950e6aa1-e29e-4266-bd8a-942fc09bb560/embodied-intelligence/liupengfei-24025/wuyz/Dataset-Research/opensearch.log 2>&1 &
# curl -X GET https://localhost:9200 -u 'admin:GAIR-scrl-1' --insecure

sudo env OPENSEARCH_INITIAL_ADMIN_PASSWORD=GAIR-scrl-1 dpkg -i /inspire/hdd/ws-950e6aa1-e29e-4266-bd8a-942fc09bb560/embodied-intelligence/liupengfei-24025/wysi/Dataset-Research/opensearch/opensearch-2.19.1-linux-x64.deb

nohup sudo -u opensearch /usr/share/opensearch/bin/opensearch > ./opensearch.log 2>&1 &
source /inspire/hdd/ws-950e6aa1-e29e-4266-bd8a-942fc09bb560/embodied-intelligence/liupengfei-24025/wysi/anaconda/bin/activate
bash ./hfd.sh intfloat/multilingual-e5-large-instruct --tool aria2c -x 16
