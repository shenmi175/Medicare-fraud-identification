import random
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import os
import logging

# 设置日志文件的路径
log_directory = '../param_optim/logs'
log_filename = os.path.join(log_directory, 'logfile.log')

# 检查目录是否存在，如果不存在则创建
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# 配置日志
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

data = pd.read_csv('../data/process_data.csv')
X = data.drop(['RES'],axis=1)
y = data['RES']

x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=42)


# 定义超参数空间
# param_space = {
#     # 'eta': [0.01, 0.015, 0.025, 0.05, 0.1],
#     # 'gamma': [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
#     'max_depth': [3, 5, 7, 9, 12, 15, 17, 25],
#     # 'n_estimators': [64,100, 128, 150, 200, 228]
#     # 'min_child_weight': [1, 3, 5, 7, 9, 12],
#     # 'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
#     # 'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
#     # 'lambda': [0.01, 0.1, 0.3, 0.5, 1.0],
#     # 'alpha': [0, 0.1, 0.3, 0.5, 1.0]
# }

# 单一决策树
param_space = {
    'max_depth': [3, 5, 7, 9, 12, 15, 17, 25],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 5, 10],
}

# 适应度函数
def eval(individual):
    try:
        params = {key: param_space[key][index] for key, index in zip(param_space.keys(), individual)}
        # model = XGBClassifier(**params,random_state=42,objective='binary:logistic',eval_metric='auc')
        # model = LGBMClassifier(**params,random_state=42)
        model = DecisionTreeClassifier(**params, random_state=42)
        skf = StratifiedKFold(n_splits=5)
        scores = cross_val_score(model, x_train, y_train, cv=skf, scoring='roc_auc')
        logging.info(f"Evaluating individual: {individual}, Score: {scores}")
        return (scores.mean(),)
    except IndexError as e:
        logging.error(f"IndexError in evalXGBoost with individual: {individual}. Error: {str(e)}")
        return (float('inf'),)

# 自适应变异函数
def adaptiveMutate(individual, param_space, rate):
    if random.random() < rate:
        for i in range(len(individual)):
            if random.random() < rate:
                # 选择一个超参数并获取它的范围
                param_key = list(param_space.keys())[i]
                param_range = param_space[param_key]

                # 随机选择新的索引值
                new_value = random.randint(0, len(param_range) - 1)
                individual[i] = new_value
                logging.info(f"Mutating individual: {individual}")



# 精英策略的实现
def modified_eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
                                 halloffame=None, verbose=__debug__):

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # 评估整个群体
    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # 使用新的的个人更新名人堂
    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)

    # 开始遗传过程
    for gen in range(1, ngen + 1):
        # 选择下一代个体
        offspring = toolbox.select(population, len(population) - len(halloffame.items))
        offspring = list(map(toolbox.clone, offspring))

        # 对后代应用交叉和变异
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                logging.info(f"Crossover between {child1} and {child2}")
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 评估适应度无效的个体
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 使用新的的个体更新名人堂（群体）
        if halloffame is not None:
            halloffame.update(offspring)

        # 用后代取代当前群体
        population[:] = offspring + halloffame.items

        # 将当前生成的统计数据附加到日志中
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # 更新最优解
        if halloffame is not None and len(halloffame.items) > 0:
            best_individual = halloffame[0]
            best_params = {key: param_space[key][index] for key, index in zip(param_space.keys(), best_individual)}
            logging.info(f"Generation {gen}: Best Params = {best_params}")

    return population, logbook



# 创建适应度类，例如最大化适应度
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# 创建个体类，继承自列表，并关联到适应度类
creator.create("Individual", list, fitness=creator.FitnessMax)

# 创建遗传算法的工具
toolbox = base.Toolbox()
max_indices = [len(param_space[key]) - 1 for key in param_space.keys()]
toolbox.register("indices", lambda: [random.randint(0, max_index) for max_index in max_indices])
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", eval)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", adaptiveMutate, param_space=param_space, rate=0.1)

# 锦标赛选择
toolbox.register("select", tools.selTournament, tournsize=3)

# 运行遗传算法
population = toolbox.population(n=50)  # 初始种群，其中包含了 50 个个体
halloffame = tools.HallOfFame(maxsize=5)  # 用于存储遗传算法运行过程中找到的最好的个体

# 交叉和变异的概率
cxpd = 0.8
mutpb = 0.2

# 繁衍代数
ngen = 30

result = modified_eaSimpleWithElitism(population, toolbox, cxpb=cxpd, mutpb=mutpb, ngen=ngen, halloffame=halloffame, verbose=True)

best_individual = tools.selBest(population, k=1)[0]
best_params = {key: param_space[key][min(index, len(param_space[key]) - 1)] for key, index in zip(param_space.keys(), best_individual)}

# print("Best Parameters:", best_params)
