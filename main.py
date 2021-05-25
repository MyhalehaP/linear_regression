import pandas as panda
import numpy as np
import matplotlib.pyplot as plot
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

data = panda.read_csv('train.csv')

plot.rcParams['figure.figsize'] = (8,8)
#
# plot.scatter(data["Foundation"],data["SalePrice"])
#
# plot.xticks(rotation=60)
# plot.xlabel("Тип основи")
# plot.ylabel("Ціна")
#
#
#
# plot.show()
# print(data.info)

df = panda.DataFrame(data,columns=['HeatingQC','SalePrice'])

test_data = panda.read_csv('test.csv')
test = panda.DataFrame(test_data,columns=['Id','OverallQual'])

dummies = panda.get_dummies(df['HeatingQC'])
print(dummies)
clms = []
k = 10
for i in range(0,len(dummies.columns)):
    clms.append(k)
    k+=10


values = []

for index,row in dummies.iterrows():
    for i in range(0,len(row)):
        if row.values[i] == 1:
            values.append(clms[i])
            break

values = np.array(values).reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(values, df['SalePrice'], test_size=0.2)

x_train = np.array(x_train).reshape(-1, 1)
x_test = np.array(x_test).reshape(-1, 1)
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)


ln_regression = linear_model.LinearRegression()
ln_regression.fit(x_train, y_train)

price_predict = ln_regression.predict(x_test)

print(r2_score(y_test, price_predict))

plot.scatter(x_test,y_test,1)
plot.plot(x_test,price_predict,color='red',linewidth=1)

plot.xlabel('Якість матеріалів')
plot.ylabel('Sale Price')
plot.show()

# #first change in test_3
# #second change in test_3
#
# output = []
#
# for index,row in test.iterrows():
#     quality = np.array(row['OverallQual']).reshape(-1,1)
#     predict = ln_regression.predict(quality)
#     output.append([row['Id'],predict[0][0]])
#
# output_df = panda.DataFrame(output,columns=['Id','SalePrice'])
# output_df.to_csv("prediction.csv",index=False)
#
