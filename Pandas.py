import numpy as np # linear algebra
import pandas as pd # import in pandas

import os

from sklearn import preprocessing

enc = preprocessing.OneHotEncoder()

enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])    # fit来学习编码
enc.transform([[0, 1, 3]]).toarray()    # 进行编码
print(enc.transform([[0, 1, 3]]).toarray())
