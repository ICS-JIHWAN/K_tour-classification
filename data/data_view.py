import pandas as pd
import glob
import cv2

from sklearn import preprocessing

tr_df = pd.read_csv('/storage/jhchoi/tour/open/train.csv')
te_df = pd.read_csv('/storage/jhchoi/tour/open/test.csv')
img_tr_paths = sorted(glob.glob('/storage/jhchoi/tour/open/image/train/*.jpg'))
img_te_paths = sorted(glob.glob('/storage/jhchoi/tour/open/image/test/*.jpg'))

# ======= Data leakage check =======
tr_df.isnull().sum()
te_df.isnull().sum()

# ======= Data head [top 5] =======
tr_df.drop(['id', 'img_path'], axis=1).head()
te_df.drop(['id'], axis=1).head()
tr_df.describe()
te_df.describe()

# ======= set cat 1, 2, 3 & length =======
tr_cat1, tr_cat2, tr_cat3 = set(tr_df['cat1'].values), set(tr_df['cat2'].values), set(tr_df['cat3'].values)
print("cat1 : {}\ncat2 : {} \ncat3 : {}".format(len(tr_cat1), len(tr_cat2), len(tr_cat3)))

# ======= Label-Encoding =======
le = preprocessing.LabelEncoder()
le.fit(tr_df['cat3'].values)
tr_df['cat3'] = le.transform(tr_df['cat3'].values)

# ======= View image =======
for img_tr_path in img_tr_paths:
    img = cv2.imread(img_tr_path)
    print(img.shape)
