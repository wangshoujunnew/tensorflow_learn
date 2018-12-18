#encoding=utf-8
from kafka import KafkaClient, KafkaProducer, KafkaConsumer

# 给kafka集群发送数据,生产者
cluster_host = '172.31.11.221:9092,172.31.11.222:9092,172.31.11.223:9092'
topic = 'test_topic'
group_id = 'test-python-test_topic'
producter = KafkaProducer(bootstrap_servers=cluster_host)
producter.send(topic=topic, value='from shoujunw test ... ')
producter.flush()
