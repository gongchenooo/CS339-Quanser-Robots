import csv
env_name = 'Qube-100-v0'
path = 'logs/' + env_name + '/monitor.csv'
with open(path, 'r') as f:
    reader = csv.reader(f)
    print(type(reader))
    i=0
    for row in reader:
        i = i+1
        print(i,':',row)