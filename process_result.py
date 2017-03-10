import pandas as pd

fname = 'result.txt'

dict = {}
pay_array=[]
with open(fname) as f:
    content = f.readlines()
    for line in content:
        data = line.split(',')
        # day
        idx = data[2]
        # shop_id
        shop_id = data[1]
        # pay_count
        pay_count = round(float(data[0]))
        if pay_count < 0:
            pay_count = -1 * pay_count
        pay_array.append(int(pay_count))
        if int(idx) == 14:
            dict[int(shop_id)] = pay_array
            pay_array = []

print dict[1]
print dict[2]

df = pd.DataFrame(dict)
df = df.transpose()

print df.head()
print df.shape[0]

df.to_csv('predict.csv', header=False)


exit()

