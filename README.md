## EX-05-Feature-Generation
## AIM
To read the given data and perform Feature Generation process and save the data to a file.

## Explanation
Feature Generation (also known as feature construction, feature extraction or feature engineering) is the process of transforming features into new features that better relate to the target. It includes two process:

    1.Feature Encoding
    2.Feature Scaling
## FEATURE ENCODING:
Ordinal Encoding An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.

Label Encoding Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.

Binary Encoding Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.

One Hot Encoding We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

## FEATURE SCALING:
Standard Scaler It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1

MinMaxScaler It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is, 0.

Maximum absolute scaling Maximum absolute scaling scales the data to its maximum value; that is, it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.

RobustScaler RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

## ALGORITHM
## STEP 1
Read the given Data

## STEP 2
Clean the Data Set using Data Cleaning Process

## STEP 3
Apply Feature Generation techniques to all the feature of the data set

## STEP 4
Save the data to the file

## CODE
~~~
Data.csv :

import pandas as pd
df=pd.read_csv("data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
import category_encoders as ce
be=ce.BinaryEncoder()
ohe=OneHotEncoder(sparse=False)
le=LabelEncoder()
oe=OrdinalEncoder()


df1["City"] = ohe.fit_transform(df1[["City"]])

temp=['Cold','Warm','Hot','Very Hot']
oe1=OrdinalEncoder(categories=[temp])
df1['Ord_1'] = oe1.fit_transform(df1[["Ord_1"]])

edu=['High School','Diploma','Bachelors','Masters','PhD']
oe2=OrdinalEncoder(categories=[edu])
df1['Ord_2']= oe2.fit_transform(df1[["Ord_2"]])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df5
Encoding.csv :

import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
le=LabelEncoder()
oe=OrdinalEncoder()

df1["nom_0"] = oe.fit_transform(df1[["nom_0"]])
temp=['Cold','Warm','Hot']
oe2=OrdinalEncoder(categories=[temp])
df1['ord_2'] = oe2.fit_transform(df1[['ord_2']])

df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df0=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df0

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df2=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df2

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df3=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df3

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df4=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df4
Titanic.csv :

import pandas as pd
df=pd.read_csv("titanic_dataset.csv")
df

#removing unwanted data
df.drop("Name",axis=1,inplace=True)
df.drop("Ticket",axis=1,inplace=True)
df.drop("Cabin",axis=1,inplace=True)

#data cleaning
df.isnull().sum()

df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

df.isnull().sum()

df

#feature encoding
from category_encoders import BinaryEncoder
be=BinaryEncoder()
df["Sex"]=be.fit_transform(df[["Sex"]])
ndf=be.fit_transform(df["Sex"])
ndf

df1=df.copy()
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
embark=['S','C','Q']
e1=OrdinalEncoder(categories=[embark])
df1['Embarked'] = e1.fit_transform(df[['Embarked']])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df5
~~~
## OUTPUT:

## DATA.csv
![o1](https://user-images.githubusercontent.com/95342910/167538918-3cbd81b3-83f5-4b53-a788-f904f95603a8.png)
![04 (1)](https://user-images.githubusercontent.com/95342910/167538966-1d0b93c0-79a9-42a0-ac64-2d1d44574167.png)
![o5](https://user-images.githubusercontent.com/95342910/167538991-ff359982-5da2-4014-be82-baebdf56ac46.png)
![o6](https://user-images.githubusercontent.com/95342910/167539014-ad7f8735-976b-44c3-98cb-12901c254489.png)
![o8](https://user-images.githubusercontent.com/95342910/167539042-3a3100c4-a4e7-4b14-a573-c984b45d9695.png)
![o9](https://user-images.githubusercontent.com/95342910/167539079-955a64aa-2849-400e-ab1c-e66f2ce76a92.png)
![o10](https://user-images.githubusercontent.com/95342910/167539104-fd13bc73-f7b8-470e-879f-b3bd7f7b3817.png)
![o11](https://user-images.githubusercontent.com/95342910/167539127-16524e0a-a843-4d6b-8d31-aba20f8f4dd2.png)
![o12](https://user-images.githubusercontent.com/95342910/167539139-3a09ba11-2363-41c0-980c-25894666599b.png)

## ENCODING.csv
![o13](https://user-images.githubusercontent.com/95342910/167539312-a9344ae0-027e-4b3b-86e5-208d53052e52.png)
![o14](https://user-images.githubusercontent.com/95342910/167539324-a6ffce75-8d21-49b9-bc3a-c618dad925ef.png)
![o15](https://user-images.githubusercontent.com/95342910/167539344-e8af4638-55b9-436d-8785-c7bc5d94d660.png)
![o16](https://user-images.githubusercontent.com/95342910/167539353-b6f36fcd-7df0-4bea-9315-f8563f007e82.png)
![o18](https://user-images.githubusercontent.com/95342910/167539374-ab337500-dd51-4313-ac12-4dfdf4a6dea7.png)
![o20](https://user-images.githubusercontent.com/95342910/167539396-d8ef3431-fad6-4f4c-bff6-10d1d9bc6477.png)
![o21](https://user-images.githubusercontent.com/95342910/167539418-0649f2f1-e54a-41d7-8458-4595980c197c.png)

## TITANIC.csv
![o22](https://user-images.githubusercontent.com/95342910/167539525-db282cbb-a45a-4bb0-981e-e209259a21a2.png)
![o23](https://user-images.githubusercontent.com/95342910/167539552-b7a90523-fc1c-40b8-ba6c-bda85b939c38.png)
![o24](https://user-images.githubusercontent.com/95342910/167539572-8bbe3116-7072-44e7-b484-7e52b22edc80.png)
![o25](https://user-images.githubusercontent.com/95342910/167539583-437531d6-95d6-49ef-a95d-fa06791dd64a.png)
![o26](https://user-images.githubusercontent.com/95342910/167539606-0cd1b1b8-4671-4331-b68a-a06cddad7000.png)
![o27](https://user-images.githubusercontent.com/95342910/167539627-6be867c8-fe61-4b93-a98e-ba56cbb9e8a4.png)

## Result:
Feature Generation process and Feature Scaling process is applied to the given data frames sucessfully.
