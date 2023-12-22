import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import  KFold, train_test_split
from sklearn.pipeline import Pipeline

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)

import warnings
warnings.filterwarnings("ignore")

#
# plt.style.use("./input/rose-pine-dawn.mplstyle.txt")
# 读取训练集、测试集和样本提交文件
path = "./input/"
train = pd.read_csv(path+"train.csv").drop("Id",axis=1)
test = pd.read_csv(path+"test.csv").drop("Id",axis=1)
sub = pd.read_csv(path+"sample_submission.csv")


train["group"] = "train"
test["group"] = "test"
# 将训练集和测试集合并为一个df
df = pd.concat([train, test], axis=0)

train.drop("group", axis=1, inplace=True)
test.drop("group", axis=1, inplace=True)
target = "SalePrice"

# 找到数值和分类变量的列
def find_col_dtypes(df, target):
    num_cols = df.select_dtypes("number").columns.to_list()
    cat_cols = df.select_dtypes("object").columns.to_list()
    num_cols = [col for col in num_cols if col not in [target]]
    cat_cols = [col for col in cat_cols if  col not in [target]]
    return num_cols, cat_cols

num_cols, cat_cols = find_col_dtypes(train, target)

print(f"Num Cols: {num_cols}", end="\n\n")
print(f"Cat Cols: {cat_cols}")



#############  数据分析  ########################
# 根据SalePrice进行划分
train["SalePrice_Range"] = pd.cut(train["SalePrice"],
                                 bins=np.array([-np.inf, 100, 150, 200, np.inf])*1000,
                                 labels=["0-100k","100k-150k","150k-200k","200k+"])

# 工具函数 ,主用来绘制目标变量的分布图
def plot_target(df: pd.DataFrame, col: str, title: str, pie_colors:list) -> None:
    fig, ax = plt.subplots(1,2,figsize=(15, 6),  gridspec_kw={'width_ratios': [2,1]})

    textprops={'fontsize': 12, 'weight': 'bold',"color": "black"}
    ax[0].pie(df[col].value_counts().to_list(),
            colors=pie_colors,
            labels=df[col].value_counts().index.to_list(),
            autopct='%1.f%%',
            explode=([.05]*(df[col].nunique()-1)+[.5]),
            pctdistance=0.5,
            wedgeprops={'linewidth' : 1, 'edgecolor' : 'black'},
            textprops=textprops)

    sns.countplot(x = col, data=df, palette = "viridis", order=df[col].value_counts().to_dict().keys())
    for p, count in enumerate(df[col].value_counts().to_dict().values(),0):
        ax[1].text(p-0.17, count+(np.sqrt(count)), count, color='black', fontsize=13)
    plt.setp(ax[1].get_xticklabels(), fontweight="bold")
    plt.yticks([])
    plt.box(False)
    fig.suptitle(x=0.56, t=f'► {title} Distribution ◄', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()
plot_target(train,
            col="SalePrice_Range",
            title="SalePrice",
            pie_colors=["#4a4c7e","#367182","#33977c","#80c161","#f3f3af","#c0ebe9"])
num_cols, cat_cols = find_col_dtypes(train,target)
cat_cols = train[cat_cols].columns[train[cat_cols].nunique() < 10]

# 绘制分布图
plt.figure(figsize=(14,len(cat_cols)*2))
for idx,column in enumerate(cat_cols):
    plt.subplot(len(cat_cols)//2+1,2,idx+1)
    sns.countplot(hue="SalePrice_Range", x=column, data=train,palette="viridis")
    plt.title(f"{column} Distribution")
    plt.tight_layout()

plt.show()

num_cols, cat_cols = find_col_dtypes(train, target)

# 绘制箱线图
num_cols = train[num_cols].columns[train[num_cols].nunique() > 15]
plt.figure(figsize=(14,len(num_cols)*3))
for idx,column in enumerate(num_cols):
    plt.subplot(len(num_cols)//2+1,2,idx+1)
    sns.boxplot(x="SalePrice_Range", y=column, data=train,palette="viridis")
    plt.title(f"{column} Distribution")
    plt.tight_layout()
plt.show()

num_cols, cat_cols = find_col_dtypes(train, target)
num_cols = train[num_cols].columns[train[num_cols].nunique() > 15]
# 绘制直方图
plt.figure(figsize=(14,len(num_cols)*3))
for idx,column in enumerate(num_cols):
    plt.subplot(len(num_cols)//2+1,2,idx+1)
    sns.histplot(x=column, hue="SalePrice_Range", data=train,bins=30,kde=True, palette="viridis")
    plt.title(f"{column} Distribution")
    plt.tight_layout()
plt.show()
num_cols, cat_cols = find_col_dtypes(train, target)
num_cols = train[num_cols].columns[train[num_cols].nunique() > 25]
# 绘制热力图
plt.figure(figsize=(12,10))
corr=train[num_cols].corr(numeric_only=True)
mask= np.triu(np.ones_like(corr))
sns.heatmap(corr, annot=True, fmt=".1f", linewidths=1, mask=mask, cmap=sns.color_palette("icefire"));
plt.show()



#### 绘制缺失值
# 导入missingno库
import missingno as msno

# 绘制缺失值矩阵图
msno.matrix(train)

# 绘制缺失值条形图
msno.bar(train)

# 绘制缺失值热力图
msno.heatmap(train)



#############Feature Engineering 特征工程########################
# 填充 "MasVnrArea" 列的缺失值，并根据填充后的值更新 "MasVnrType" 列
df.loc[:,"MasVnrArea"] = df["MasVnrArea"].fillna(0)
df.loc[df["MasVnrArea"] == 0,"MasVnrType"] = "None"

def drop_feature(data,columns, percentage = 95):
    data = data.copy()
    for col in columns:
        if data[col].value_counts().sort_values(ascending=False).iloc[0] > percentage*len(df)/100:
            print(f"Feature {col} is Nonsense, Dropped")
            data.drop(col, axis=1, inplace=True)
    return data

df = drop_feature(df,cat_cols,percentage = 95)
num_cols, cat_cols = find_col_dtypes(df, target)



####处理缺失值后再次绘制

# 导入missingno库
import missingno as msno

# 绘制缺失值矩阵图
msno.matrix(df)

# 绘制缺失值条形图
msno.bar(df)

# 绘制缺失值热力图
msno.heatmap(df)



# 对分类变量进行稀有类别编码
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])
    return temp_df

df = rare_encoder(df, 0.05)

# 定义新的特征
def new_features(df):
    df['TotalArea'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['NewBathrooms'] = df['FullBath'] + df['HalfBath']
    df['TotalRooms'] = df['BedroomAbvGr'] + df['TotRmsAbvGrd']

new_features(train)
new_features(test)

# 对数据进行处理，去除异常值并处理高相关性的特征
def corr_outliner(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.01)
        Q3 = df[col].quantile(0.99)
        df.loc[df[col] < Q1, col] = round(Q1)
        df.loc[df[col] > Q3, col] = round(Q3)

    correlation = df[cols].corr().abs()
    triangle = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
    drop_list = [column for column in triangle.columns if any(triangle[column] > 0.91)]
    df.drop(drop_list, axis=1, inplace=True)

    return df


num_cols, cat_cols = find_col_dtypes(df, target)
num_cols = df[num_cols].columns[df[num_cols].nunique() > 25]

df = corr_outliner(df,num_cols)


plt.figure(figsize=(12,10))
corr=df[num_cols].corr(numeric_only=True)
mask= np.triu(np.ones_like(corr))
sns.heatmap(corr, annot=True, fmt=".1f", linewidths=1, mask=mask, cmap=sns.color_palette("icefire"));
plt.show()


# 分割数据集为训练集和测试集
train = df[df["group"] == "train"].drop("group", axis = 1)
test = df[df["group"] == "test"].drop(["group","SalePrice"], axis = 1)

num_cols, cat_cols = find_col_dtypes(train, target)
# 对分类变量进行独热编码
train = pd.get_dummies(train, dummy_na=True, columns=cat_cols, drop_first= True, dtype=int)
test = pd.get_dummies(test, dummy_na=True, columns=cat_cols, drop_first= True, dtype=int)

train.columns = [col.replace(" ", "_") for col in train.columns]
test.columns = [col.replace(" ", "_") for col in test.columns]



    #################### Modeling 建 模 ############################
from sklearn.metrics import mean_absolute_error, median_absolute_error
from sklearn.metrics import r2_score
# 定义特征和目标变量
X = train.drop(["SalePrice"], axis=1)
y = np.log(train["SalePrice"])

# 用来记录模型和对应的
model = []
score = []

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state = 1210)


# k-折交叉验证  求取超参数
RANDOM_SEED = 1 # 给个种子，方便复现
# 10-fold CV
kfolds = KFold(n_splits=10,shuffle=True,random_state=RANDOM_SEED)
def tune(objective):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    params = study.best_params
    best_score = study.best_value
    print(f"Best score: {best_score} \nOptimized parameters: {params}")
    return params

################   General Linear Models（一般线性模型）   ########################
# 线性回归模型
# 使用管道训练 Linear模型，无需单独处理NaN值
pipe = Pipeline([("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
                 ("ridge", LinearRegression())])
pipe.fit(X_train,y_train)
linear_model_predict = pipe.predict(X_test)
print("linear_model_Score: ",mean_absolute_error(linear_model_predict,y_test))
model.append("Multi Linear Regression")
score.append(mean_absolute_error(linear_model_predict,y_test))


# 岭回归模型
# 使用管道训练Ridge模型，无需单独处理NaN值
pipe = Pipeline([("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
                 ("ridge", Ridge(random_state=RANDOM_SEED))])
pipe.fit(X_train,y_train)
ridge_model_predict = pipe.predict(X_test)
print("ridge_model_Score: ",mean_absolute_error(ridge_model_predict,y_test))
model.append("Ridge Regression")
score.append(mean_absolute_error(ridge_model_predict,y_test))


# 套索回归模型
# 使用管道训练lasso模型，无需单独处理NaN值
pipe = Pipeline([("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
                 ("lasso", Lasso(random_state=RANDOM_SEED))])
pipe.fit(X_train,y_train)
lasso_model_predict = pipe.predict(X_test)
print("lasso_model_Score: ",mean_absolute_error(lasso_model_predict,y_test))
model.append("Lasso Regression")
score.append(mean_absolute_error(lasso_model_predict,y_test))

################   Ensemble Model（集成模型）   ########################
# 将LightGBM Regressor，XGBoost Regressor，Catboost Regressor使用voting

# 训练 LightGBM 模型
import lightgbm
lgb = lightgbm.LGBMRegressor(objective = 'mae')
lgb.fit(X_train, y_train)
lightgbm.plot_importance(lgb, max_num_features = 15);
mean_absolute_error(y_test,lgb.predict(X_test))
plt.show()
# 训练 XGBoost 模型
import xgboost
xgb = xgboost.XGBRegressor()
xgb.fit(X_train, y_train)
xgboost.plot_importance(xgb,max_num_features = 15);
mean_absolute_error(y_test,xgb.predict(X_test))
plt.show()

# 对 LightGBM 和 XGBoost 的特征重要性进行归一化处理并合并
from sklearn.preprocessing import MinMaxScaler
lgb_importances = pd.DataFrame(dict(lgbm = lgb.feature_importances_), index=lgb.feature_name_)
xgb_importances = pd.DataFrame(dict(xgb = xgb.feature_importances_), index=xgb.feature_names_in_)
importances = pd.concat([lgb_importances,xgb_importances],axis=1)
min_max = MinMaxScaler((0,1))
importances["cross"] = min_max.fit_transform(importances[["lgbm"]]) * min_max.fit_transform(importances[["xgb"]])
sorted = importances.sort_values(by="cross", ascending=False)
sorted.head(10)

# 删除排名较低的特征
X_train.drop(sorted.tail(100).index,axis=1, inplace=True)
X_test.drop(sorted.tail(100).index,axis=1, inplace=True)
test.drop(sorted.tail(100).index,axis=1, inplace=True)

################ LightGbm Regressor LightGbm 回归器  ##########################
from lightgbm import LGBMRegressor
import optuna
def objective_lgb(trial):
    """Define the objective function"""

    params = {
        'objective': trial.suggest_categorical('objective', ['mae']),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 15),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.05),
        'n_estimators': trial.suggest_int('n_estimators', 300, 700),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.1, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
        "seed" : trial.suggest_categorical('seed', [42]),
    }

    model_lgb = LGBMRegressor(**params)
    model_lgb.fit(X_train, y_train)
    y_pred = model_lgb.predict(X_test)
    return mean_absolute_error(y_test,y_pred)

study_lgb = optuna.create_study(direction='minimize')
optuna.logging.set_verbosity(optuna.logging.WARNING)
# 进行优化，n_trials是优化的迭代次数
study_lgb.optimize(objective_lgb, n_trials=50,show_progress_bar=True)

print('Best parameters', study_lgb.best_params)

# 使用最佳参数创建 LightGBM 模型并进行训练
lgb = LGBMRegressor(**study_lgb.best_params)
lgb.fit(X_train, y_train)
y_pred = lgb.predict(X_test)
# 打印模型性能评价指标
model.append("xgboost Regression")
score.append(mean_absolute_error(y_test, y_pred))
print('LGB MAE Error: ', mean_absolute_error(y_test, y_pred))

################### Xgboost Regressor Xgboost 回归器  ########################

from xgboost import XGBRegressor
import optuna

def objective_xg(trial):
    """Define the objective function"""

    params = {
        'booster': trial.suggest_categorical('booster', ['gbtree']),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.05),
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'subsample': trial.suggest_loguniform('subsample', 0.3, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
        "seed" : trial.suggest_categorical('seed', [42]),
        'objective': trial.suggest_categorical('objective', ['reg:absoluteerror']),
    }
    model_xgb = XGBRegressor(**params)
    model_xgb.fit(X_train, y_train)
    y_pred = model_xgb.predict(X_test)
    return mean_absolute_error(y_test,y_pred)

study_xgb = optuna.create_study(direction='minimize')
optuna.logging.set_verbosity(optuna.logging.WARNING)
# 进行优化，n_trials是优化的迭代次数
study_xgb.optimize(objective_xg, n_trials=50,show_progress_bar=True)
print('Best parameters', study_xgb.best_params)
xgb = XGBRegressor(**study_xgb.best_params)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
model.append("xgboost Regression")
score.append(mean_absolute_error(y_test, y_pred))
print('Xgboost MAE Error: ', mean_absolute_error(y_test, y_pred))

################ Catboost Regressor 回归器  ##########################
from catboost import CatBoostRegressor
import optuna
def objective_cat(trial):
    """Define the objective function"""

    params = {
        'objective': trial.suggest_categorical('objective', ['MAE']),
        'logging_level': trial.suggest_categorical('logging_level', ['Silent']),
        "random_seed": trial.suggest_categorical('random_seed', [42]),
        "iterations": trial.suggest_int("iterations", 500, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
        "depth": trial.suggest_int("depth", 5, 8),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 15),
        'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 0.01, 1),
        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 15),
        'reg_lambda': trial.suggest_uniform('reg_lambda', 50, 100),

    }

    model_cat = CatBoostRegressor(**params)
    model_cat.fit(X_train, y_train)
    y_pred = model_cat.predict(X_test)
    return mean_absolute_error(y_test, y_pred)



study_cat = optuna.create_study(direction='minimize')
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_cat.optimize(objective_cat, n_trials=50,show_progress_bar=True)

# Print the best parameters
print('Best parameters', study_cat.best_params)

cat = CatBoostRegressor(**study_cat.best_params)
cat.fit(X_train, y_train)
y_pred = cat.predict(X_test)
model.append("catboost Regression")
score.append(mean_absolute_error(y_test, y_pred))
print('Cat MAE Error: ', mean_absolute_error(y_test, y_pred))



############### Voting Regressor 投票回归器  ####################
from sklego.linear_model import LADRegression
# 创建 DataFrame 存储模型的预测结果
models = pd.DataFrame()
models["cat"] = cat.predict(X_test)
models["lgbm"] = lgb.predict(X_test)
models["xgb"] = xgb.predict(X_test)
# 使用 LADRegression 计算模型的权重
weights = LADRegression().fit(models, y_test).coef_
pd.DataFrame(weights, index = models.columns, columns = ["weights"])
plt.show()


# 整合优化后的 LightGBM 和 XGBoost 模型 和 CatBoost 模型
voting = VotingRegressor(estimators=[
                                      ('cat', cat),
                                      ('lgbm', lgb),
                                      ('xgb', xgb)],weights=weights)
voting.fit(X_train,y_train)
voting_pred = voting.predict(X_test)
# 打印 VotingRegressor 的性能评价指标
model.append("voting boost Regression")
score.append(mean_absolute_error(y_test, voting_pred))
print('Voting MAE Error: ', mean_absolute_error(y_test, voting_pred))

######################  性能比较  ################################
plt.subplots(figsize=(15, 5))
sns.barplot(x=score,y=model,palette = sns.cubehelix_palette(len(score)))
plt.xlabel("Score")
plt.ylabel("Regression")
plt.title('Regression mean_absolute_error')
plt.show()

#  由上图可知  voting的预测结果正确率更好 故选择 voting 模型来 预测
# 生成最终预测结果，并保存到'submission.csv'文件中
sub["SalePrice"]=np.exp(voting.predict(test))
sub.to_csv('submission.csv',index=False)
sub