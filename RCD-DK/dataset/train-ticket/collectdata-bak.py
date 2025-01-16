#!/usr/bin/env python3

# This script scrapes data for Train Ticket microservices and node metrics using Prometheus

import argparse
import requests
import numpy as np
import pandas as pd

HOST_TEMPLATE = "http://{}:30090/"
MAX_RESOLUTION = 11_000  # Maximum resolution of Prometheus
TRAIN_TICKET_NS = "train-ticket"  # Train Ticket namespace

STEP = 1  # In seconds
DURATION = "45s"
NODES = ["k8s-2", "k8s-3", "k8s-4"]
NODE_QUERIES = {
    # "node_cpu": f'100 - avg(rate(node_cpu_seconds_total{{instance=~"k8s-[2-4]", job="node-exporter", mode="idle"}}[{DURATION}])) by (instance) * 100',
    "node_cpu": f'(avg(rate(node_cpu_seconds_total{{instance=~"k8s-[2-4]", job="node-exporter", mode!="idle"}}[{DURATION}])) by (instance) * 100) - (sum(rate(container_cpu_usage_seconds_total{{namespace="{TRAIN_TICKET_NS}"}}[{DURATION}])) by (instance) * 100)',
    # "node_mem": f'100 - (node_memory_MemAvailable_bytes{{instance=~"k8s-[2-4]", job="node-exporter"}} / node_memory_MemTotal_bytes{{instance=~"k8s-[2-4]", job="node-exporter"}} * 100)'
    # "node_mem": f'100 - ((node_memory_MemAvailable_bytes{{instance=~"k8s-[2-4]", job="node-exporter"}} + sum(container_memory_usage_bytes{{namespace="{TRAIN_TICKET_NS}"}})) / node_memory_MemTotal_bytes{{instance=~"k8s-[2-4]", job="node-exporter"}} * 100)'
}
QUERIES = {
    "cpu": f"sum(rate(container_cpu_usage_seconds_total{{namespace='{TRAIN_TICKET_NS}', pod=~'ts-.*|tsdb-mysql-.*|rabbitmq.*'}}[{DURATION}])) by (pod) * 100",
    "mem": f"sum(container_memory_usage_bytes{{namespace='{TRAIN_TICKET_NS}', pod=~'ts-.*|tsdb-mysql-.*|rabbitmq.*'}}) by (pod) / 1024 / 1024",  # Convert bytes to MB
    "load": f"tcp_sendmsg_calls_total{{job='nodejseBPF', service_name=~'ts-.*|tsdb-mysql-.*|rabbitmq.*'}}"
}

def _merge(x, y):
    data = x
    for key in y:
        data[key] = x.get(key, []) + y[key]
    return data

def _exec_query(query, start_time, end_time, host):
    response = requests.get(host + '/api/v1/query_range',
                            params={
                                'query': query,
                                'start': start_time,
                                'end': end_time,
                                'step': f"{STEP}s"
                            })
    data = {}
    results = response.json()['data']['result']
    for result in results:
        if 'pod' in result['metric']:
            service_name = result['metric']['pod']
        else:
            service_name = result['metric']['service_name']
        data[service_name] = result['values']
    return data

def _exec_node_query(query, start_time, end_time, host):
    response = requests.get(host + '/api/v1/query_range',
                            params={
                                'query': query,
                                'start': start_time,
                                'end': end_time,
                                'step': f"{STEP}s"
                            })
    data = {}
    results = response.json()['data']['result']
    for result in results:
        instance = result['metric']['instance']
        data[instance] = result['values']
    return data

def exec_query(query, start, end, host):
    if not (end - start) / STEP > MAX_RESOLUTION:
        return _exec_query(query, start, end, host)

    data = {}
    start_time = start
    end_time = start
    while end_time < end:
        end_time = min(end_time + MAX_RESOLUTION, end)
        print(f"Querying data from {start_time} to {end_time}")
        d = _exec_query(query, start_time, end_time, host)
        data = _merge(data, d)
        start_time = end_time + 1
    return data

def exec_node_query(query, start, end, host):
    if not (end - start) / STEP > MAX_RESOLUTION:
        return _exec_node_query(query, start, end, host)

    data = {}
    start_time = start
    end_time = start
    while end_time < end:
        end_time = min(end_time + MAX_RESOLUTION, end)
        print(f"Querying node data from {start_time} to {end_time}")
        d = _exec_node_query(query, start_time, end_time, host)
        data = _merge(data, d)
        start_time = end_time + 1
    return data

def get_data(queries, start, end, host):
    data = {}
    for name, query in queries.items():
        print(f"Working on query for {name}...")
        data[name] = exec_query(query, start, end, host)

    columns = {}
    for m, containers in data.items():
        for c, info in containers.items():
            i = np.array(info)
            time = i[:, 0]
            values = i[:, 1]
            if len(columns) == 0:
                columns["time"] = time
            if len(columns["time"]) < len(time):
                columns["time"] = time
            if (c.startswith("ts-") and len(c.split('-')) > 3) or c.startswith("rabbitmq"):
                container_name = '-'.join(c.split('-')[:-2]) if c.startswith("ts-") else 'rabbitmq'
                # print(f"Container name: {container_name}")
            else:
                container_name = c
            columns[f"{container_name}_{m}"] = values
    return columns

def get_node_data(queries, start, end, host):
    data = {}
    for name, query in queries.items():
        print(f"Working on node query for {name}...")
        data[name] = exec_node_query(query, start, end, host)

    columns = {}
    for m, instances in data.items():
        for instance, info in instances.items():
            i = np.array(info)
            time = i[:, 0]
            values = i[:, 1].astype(float)  # Ensure values are in float
            if len(columns) == 0:
                columns["time"] = time
            if len(columns["time"]) < len(time):
                columns["time"] = time
            columns[f"{instance}_{m}"] = values
    return columns

def make_dict_list_equal(dict_list):
    l_min = float('inf')
    for key in dict_list:
        l_min = min(l_min, len(dict_list[key]))

    new_dict = {}
    for key, old_list in dict_list.items():
        new_list = old_list
        if len(old_list) > l_min:
            print(f"Discarding {len(old_list) - l_min} entries from the end of the column name {key}")
            new_list = old_list[:l_min]
        new_dict[key] = new_list
    return new_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect data from Prometheus for Train Ticket microservices and node metrics')

    parser.add_argument('--ip', type=str, required=True, help='The IP of VM/container running Prometheus')
    parser.add_argument('--start', type=int, required=True, help='The start time (UNIX timestamp)')
    parser.add_argument('--end', type=int, required=True, help='The end time (UNIX timestamp)')
    parser.add_argument('--name', type=str, default='data.csv', help='The name/path of the file')
    parser.add_argument('--append', action='store_true', help='Append to the file')

    args = parser.parse_args()
    ip = args.ip
    start = args.start
    end = args.end
    name = args.name
    append = args.append

    host = HOST_TEMPLATE.format(ip)
    microservice_data = get_data(QUERIES, start, end, host)
    node_data = get_node_data(NODE_QUERIES, start, end, host)
    
    # Merge microservice data and node data
    data = {**microservice_data, **node_data}
    
    df = pd.DataFrame(make_dict_list_equal(data))
    if append:
        df.to_csv(name, index=False, mode='a', header=False)
    else:
        df.to_csv(name, index=False)

    print(f"The timeseries data is saved in file name {name}!")
    print(f"Total number of records are {len(df)}")
