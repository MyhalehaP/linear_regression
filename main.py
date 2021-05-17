import pandas as panda
import numpy as np
import matplotlib.pyplot as plot
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

data = panda.read_csv('train.csv')

df = panda.DataFrame(data,columns=['OverallQual','SalePrice'])

test_data = panda.read_csv('test.csv')
test = panda.DataFrame(test_data,columns=['Id','OverallQual'])

x_train, x_test, y_train, y_test = train_test_split(df['OverallQual'], df['SalePrice'], test_size=0.2)

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

plot.xlabel('Overall Quality')
plot.ylabel('Sale Price')
plot.show()


output = []

for index,row in test.iterrows():
    quality = np.array(row['OverallQual']).reshape(-1,1)
    predict = ln_regression.predict(quality)
    output.append([row['Id'],predict[0][0]])

output_df = panda.DataFrame(output,columns=['Id','SalePrice'])
output_df.to_csv("prediction.csv",index=False)

