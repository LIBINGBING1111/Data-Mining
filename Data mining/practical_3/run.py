# -*- coding: utf-8 -*-
# =============================================================================
# question1:
# =============================================================================
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth

GPA=pd.read_csv(".\specs\gpa_question1.csv")

GPA=GPA.drop(['count'],axis=1)
GPAdata=GPA.values

te = TransactionEncoder()
te_ary = te.fit(GPAdata).transform(GPAdata)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.15, use_colnames=True)

frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

frequent_itemsets.to_csv("./output/question1_out_apriori.csv")

rules1=association_rules(frequent_itemsets, metric="confidence", min_threshold=0.9)
rules1.to_csv("./output/question1_out_rules9.csv")

rules2=association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
rules2.to_csv("./output/question1_out_rules7.csv")

# =============================================================================
# questio2:
# =============================================================================
BANK=pd.read_csv(".\specs\data_question2.csv")
BANK=BANK.drop(['id'],axis=1)

BANK['age']=pd.cut(BANK['age'], 3).astype(str)
BANK['income']=pd.cut(BANK['income'], 3).astype(str)
BANK['children']=pd.cut(BANK['children'], 3).astype(str)

dataset = pd.get_dummies(BANK)
frequent_itemsets3=fpgrowth(dataset, min_support=0.20, use_colnames=True)
frequent_itemsets3.to_csv("./output/question2_out_fpgrowth.csv")


rules3=association_rules(frequent_itemsets3, metric="confidence", min_threshold=0.78)
rules3.to_csv("./output/question2_out_rules.csv")

