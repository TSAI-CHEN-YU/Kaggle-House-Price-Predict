# Kaggle-House-Price-Predict
2021/9  房價預測練習

## 一、導入資料(Input Data)

- 資料來源
    - Kaggle房價預測競賽
    
    [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
    
- 問題定義
    - 從資料集中挑出重要的特徵，訓練模型並預測該房屋的售價
    
    [5W1H分析](https://www.notion.so/bae752165de1454e90695ec494ff5cfe)
    

---

## 二、資料理解(Data Understanding)

- **資料理解**
    - 80個特徵+1個目標變數(SalePrice)，共1460個訓練樣本。
    - 根據特徵，預測房屋價格：迴歸演算法
- **欄位特性**
    
    觀察每個特徵的特點，離散、連續的；與目標變數間的關聯。
    
    [房屋專案：(79)變數分析](https://www.notion.so/86893d63460549f9b86f56c4ead01ba6)
    

---

## 三、探索性資料分析(Exploratory Data Analysis, EDA)

資料視覺化， 判斷如何以最佳方式操作資料

- **數值變數**
    
    [**描述性統計分析**](https://www.notion.so/98004c7fd8f74595a86f3b002a1663d7)
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d54fc597-df4e-4817-ba4d-878d5c49b1c6/Untitled.png)
    
    [一樓面積]：左偏分佈，可做Log轉換
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/398fa55a-949f-4086-978e-f5c113c63cad/Untitled.png)
    
    [建造年分]：年份特徵(∈類別)具有較多的類別，直接one hot encoding會造成過於稀疏的資料，因此需要做特徵轉換
    

 

[數值特徵 vs Y (散佈圖、箱型圖)](https://www.notion.so/vs-Y-a6e7d819bd49487e9f594c2485af59f0)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/45ee03e3-2f2c-4a23-8191-1cfbc37af5e9/Untitled.png)

[地下室面積]：極端值，可以刪除

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/41d6b290-2ae9-4786-b494-394f34389b7e/Untitled.png)

[總體評價]：與價格明顯正相關，優先保留

[特徵分佈圖(直方圖+KDE曲線)](https://www.notion.so/KDE-576c84b4dc2c41bd9478f1db468573ff)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/164029f5-32bd-42aa-a7ff-1a46e8374c3a/Untitled.png)

[地下二樓面積]：欄位0過多，轉為類別(1/0)

- **類別變數**
    - 排除過於單一類別變數
    
    [類別變數分布](https://www.notion.so/28b67a6d4a044357936d0cb591040da2)
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/020d8c52-5350-4d49-b487-2e9e7a12359b/Untitled.png)
    
    [屋頂材料]：分佈類別過於單一，可以刪除
    

 

[類別特徵 vs Y (箱型圖)](https://www.notion.so/vs-Y-2e742c51488b4cb78a4e4df12a4ca686)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cd5e969f-ccd8-40f1-bd76-ae7613e3fe24/Untitled.png)

[外部材料質量]：順序資料，Label Encoding

---

## 四、資料清理(Data Cleaning)

### 缺失值處理

[缺失率(長條圖)：33欄缺失](https://www.notion.so/33-08095ec54183498c874c0f18268ebf34)

- 移除 缺失值太多(0.7% up)的欄位：-4
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6350e96a-3387-4de5-921d-63a6664f5141/Untitled.png)
    
    ```python
    Missing_1 = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']
    data_all.drop(Missing_1, axis = 1, inplace = True)
    ```
    
- 填補 none & 0：-29
    
    缺失原因通常是那間房子沒有該設備而不是漏記，故直接補none & 0。
    
    年份看作類別的缺失值，直接補none
    
    [缺失值填補表格](https://www.notion.so/fba0d51b26294ce4aea952079624dc87)
    

### 極端值處理

- 類別特徵
    
    移除 類別過於單一的欄位： -5
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c1cda863-0972-40ec-bfaf-7f3e11861313/Untitled.png)
    
    ```python
    Extr_cate = ['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating']
    data_all.drop(Extr_cate, axis = 1, inplace = True)
    ```
    
- 數值特徵
    
    根據散佈圖，手動設定閥值，排除極端值。
    
    [數值特徵 vs Y (散佈圖)](https://www.notion.so/vs-Y-71dd923b51e741ef88dc4af23744c51c)
    
    排除7個樣本(2919→2912)
    
    ```python
    out_thre = {'LotFrontage':300, 'LotArea':200000, 'BsmtFinSF1':5000, 'BsmtFinSF2':1400, 'TotalBsmtSF':6000,
                '1stFlrSF':4000, 'GrLivArea':5000, 'BedroomAbvGr':8, 'EnclosedPorch':500, 'MiscVal':15000}
    
    data_all_train = data_all[~(pd.isnull(data_all.SalePrice))]
    data_all_test = data_all[(pd.isnull(data_all.SalePrice))]
    
    for i in out_thre:
        col = i; thre = out_thre[i]
        data_all_train=data_all_train[data_all_train[col] < thre]
    
    print(f'before: {len(data_all)}')
    data_all = pd.concat((data_all_train, data_all_test), sort=False).reset_index(drop=True)
    print(f'after:  {len(data_all)}')
    ```
    

---

## 五、特徵工程

### 特徵工程

- 特徵轉換
    - 1.欄位0的數目超過7成→轉為二分類(1/0)
        
        ex:地下二樓面積→有無地下二樓
        
        ```python
        for i in many_zero:
            new_name = i + '_cate'
            data_all[new_name]  = (data_all[i] != 0).astype(int)
        ```
        
    - 2.車庫建造時間 → 有無車庫(1/0)
        
        ```python
        data_all['Garage'] = (data_all['GarageYrBlt'] == 'none').astype(int)
        ```
        
    - 3.落成年分 → 落成年分～售出年分(/年)
        
        ```python
        data_all['HousetoSale_year'] = data_all['YrSold'] - data_all['YearBuilt']
        ```
        
    - 4.改建年分 → 改建年分～售出年分(/年)
        
        ```python
        data_all['RemodtoSale_year'] = data_all['YrSold'] - data_all['YearRemodAdd']
        ```
        

- Log轉換：標準化數據~normal
    
    [Log轉換前後比較](https://www.notion.so/Log-34e19d9069e94142bd37544dbfedde8a)
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d6a8e31f-0eb9-48b6-a01d-af4097971db8/Untitled.png)
    
    [一樓面積]
    

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0b0a615c-8a6a-44ab-b45b-3a75a561bab9/Untitled.png)

[一樓面積]

```jsx
data_all[i] = np.log1p(data_all[i])
```

- 數值特徵->類別特徵(label encoding)
    
    
    ```jsx
    # 銷售年分(YrSold)、銷售月份(MoSold)、建築種類(MSSubClass)
    type0 = ['YrSold', 'MoSold', 'MSSubClass'] 
    for i in type0:
        data_all[i]=data_all[i].astype(str)
    ```
    
- 類別特徵->順序特徵(label encoding)
    
    ```jsx
    data_all[i] = data_all[i].replace(['Ex','Gd','TA','Fa','Po', 'none'], [5, 4, 3, 2, 1, 0]).astype(int)
    
    data_all[i] = data_all[i].replace(['Gd','Av','Mn','No','none'], [4, 3, 2, 1, 0]).astype(int)
    
    data_all[i] = data_all[i].replace(['GLQ','ALQ','BLQ','Rec','LwQ','Unf','none'], [6, 5, 4, 3, 2, 1, 0]).astype(int)
    
    data_all[i] = data_all[i].replace(['Sev','Mod','Gtl'], [2, 1, 0]).astype(int)
    ```
    
- 類別特徵->數值特徵(one hot encoding)
    
    ```jsx
    pd.get_dummies(data_all)
    ```
    
    特徵工程 後特徵數目：242
    

### 特徵篩選

- 篩選變異數(<0.1)
    
    使用很小的閾值篩選 變異數，排除幾乎沒有差異，對於區分樣本沒有貢獻的特徵。
    
    ```jsx
    VarThre = VarianceThreshold(threshold=0.1)
    VarThre.fit_transform(data_final)
    ```
    
    篩選 變異數 後特徵數目：80
    
- 篩選相關性(<0.5)
    
    使用皮爾森相關係數篩選 相關性，排除與Y不相關的特徵。
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c70e15d2-1f98-4a1a-9dcb-a88e9e59d998/Untitled.png)
    
    ```jsx
    data_final.corr(),
    ```
    
    篩選 相關性 後特徵數目：14
    

---

## 六、建構模型

- 資料切割
    
    三種資料集(原始、y做log轉換、標準化)，分割為 7(訓練集)：3(驗證集)
    
    ```jsx
    # 1.original data
    train_test_split(x, y,test_size = 0.3, random_state=1996)
    
    # 2.y log trans data
    y_log = np.log(y)
    
    # 3.standard data
    x_std = StandardScaler().fit_transform(x)
    y_std = StandardScaler().fit_transform(y)
    
    ```
    
- 選擇模型
    
    3種回歸常用model
    
    ```python
    # 1.極限梯度提升法(XGBoost)
    from xgboost import XGBRegressor 
    # 2.隨機森林(Random Forest)
    from sklearn.ensemble import RandomForestRegressor 
    # 3.嶺回歸(Ridge regression)
    from sklearn.linear_model import Ridge 
    ```
    
- 調參
    
    選擇模型的重要參數，根據RandomizedSearchCV()
    
    [調參code](https://www.notion.so/code-a253f22741e64d309a31a1919186ee06)
    
    ```python
    def parameter_select(model, para, x_train, y_train):
        grid = RandomizedSearchCV(estimator = model, #訓練的學習器
                                  param_distributions = para, #字典，放入參數範圍
                                  cv = 5, #5折交叉驗證
                                  scoring = 'neg_root_mean_squared_error', #評估方式為RMSE
                                  n_iter = 100, #訓練10次
                                  n_jobs = -1, #使用所有的CPU進行訓練
                                  verbose = 2 #控制詳細程度
                                 )
        grid.fit(x_train, y_train)
        #最優參數組合
        return grid.best_params_
    ```
    
- 建模
    
    根據調參結果訓練模型
    
    - 判定係數(coefficient of determination) or R平方(R squared)
        
        模型對因變量y產生變化的可解釋程度
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8928d322-6c16-4365-b6cf-f058a3f6df3b/Untitled.png)
        
        模型沒有出現過度配適
        
    - 均方誤差(RMSE)
        
        預測值&實際值的 平均差異，估計預測模型預測目標值的準確度。
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/69521518-8092-4652-a158-e5ddfdd7b533/Untitled.png)
        
        y_log的預測誤差較小
        
- 集成模型
    
    三個model的預測值做平均
    
    ```jsx
    from sklearn.ensemble import VotingRegressor
    ```
    
    [model in y log data](https://www.notion.so/b781a94423b24b0bb00a016a0cdc4364)
    

---

## 七、結論

### 預測成果

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8911b398-0eaf-41b0-b293-e32e71e1b87a/Untitled.png)

### 面臨的資料議題

- 模型泛用性
    
    房價資料可能有地域性，訓練出的模型，不一定適用於所有地區。
    
    解決：蒐集不同地區資料，增加資料信息，尋找模式和關聯性。
    
- 特徵數量
    
    影響房價的因素很多，79個絕對不是全部。例如：政府政策(房地合一2.0)、國際局勢(疫情)...等。
    
    解決：蒐集、連結不同領域的資料做分析。
    
- 疫情時代
    
    疫情衝擊，政府超低利率刺激經濟，家庭儲蓄增加，導致全球房價上漲。
    
    這種房價變化，在過去是沒有資料可供參考的，難以訓練預測模型。
    
    解決：疫情爆發到現在快2年，可以利用這些資料做預測。
    

### 未來模型提升可能方向

- AutoML(模型自動化)：
    
    原因：機器學習模型開發中，許多步驟反覆耗時。
    
    解決：資料分析、特徵選擇、特徵工程、模型調參、比較，一切自動化。(input→output)
    
- 深度學習模型：
    
    原因：機器學習多仰賴人工的處理特徵，耗時費力。
    
    解決：降低人工處理特徵時間，透過複雜的神經網路尋找特徵，訓練模型。
    

### Source Code
