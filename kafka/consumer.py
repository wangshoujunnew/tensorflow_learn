#encoding=utf-8
from kafka import KafkaClient, KafkaProducer, KafkaConsumer

# 给kafka集群发送数据, 消费者
cluster_host = '172.31.11.221:9092,172.31.11.222:9092,172.31.11.223:9092'
topic = 'test_topic'
group_id = 'test-python-test_topicfasdfasdfs'  相同groupid 的消费者消费同一份数据,不会重复消费

# 消费kafka 数据
consumer = KafkaConsumer(topic, bootstrap_servers=cluster_host, group_id=group_id)
for message in consumer:
    print(message)
