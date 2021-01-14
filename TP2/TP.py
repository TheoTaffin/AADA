import datetime
import pandas as pd

import efficient_apriori
import os

import spmf

lines = []

local_root = 'C:/Users/Theo/PycharmProjects/AADA/'
data_path = os.path.join(local_root, 'TP2/data/accident.txt')

with open(data_path,  'r') as f:
    for line in f.readlines():
        line_array = line.split()
        line_array = tuple(map(int, line_array))
        lines.append(line_array)

print(len(lines))
t1 = datetime.datetime.now()
item_sets_c05, rules_c05 = efficient_apriori.apriori(lines, min_support=0.1,
                                                     min_confidence=0.5,
                                                     verbosity=1, max_length=5)
t2 = datetime.datetime.now()
time_taken_for_this_shit = t2 - t1
print(time_taken_for_this_shit)

# Question 2
item_sets_3 = dict(sorted(item_sets_c05[3].items(), key=lambda item: item[1], reverse=True))
item_sets_4 = dict(sorted(item_sets_c05[4].items(), key=lambda item: item[1], reverse=True))
item_sets_5 = dict(sorted(item_sets_c05[5].items(), key=lambda item: item[1], reverse=True))

# r : left side, r : right side (conclusion), [a : above, None : strict, u : under]

# Question 3
rules_al2_r1 = list(filter(lambda rule: len(rule.lhs) >= 2 and len(rule.rhs) == 1, rules_c05))
sorted_rules_al2_r1 = list(sorted(rules_al2_r1, key=lambda key: key.confidence, reverse=True))
for rule in sorted_rules_al2_r1[0:5]:
    print(rule)

# Question 4
item_sets_c04, rules_c04 = efficient_apriori.apriori(lines[0:10000], min_support=0.1,
                                                     min_confidence=0.4,
                                                     verbosity=1, max_length=5)

rules_r1_c1 = list(filter(lambda rule: len(rule.lhs) == 1 and len(rule.rhs) == 1, rules_c04))
# self explanatory
results_r1_c1_inverted_conf_equal = {}
for rule_1 in rules_r1_c1:
    tmp = rule_1
    for rule_2 in rules_r1_c1:
        """
        Successfully found results if we round to maximum 2 digits, otherwise no results are yielded
        """
        if tmp.lhs == rule_2.rhs and tmp.rhs == rule_2.lhs and \
                round(tmp.confidence, 2) == round(rule_2.confidence, 2):
            if not list(filter(lambda r: r.lhs == tmp.rhs and r.rhs == tmp.lhs,
                        results_r1_c1_inverted_conf_equal.keys())):
                results_r1_c1_inverted_conf_equal[tmp] = tmp.confidence


# Question 5
results_rn_cn_inverted_conf_equal = {}
for i in range(2, 6, 1):

    rules_ri_ci = list(filter(lambda rule: len(rule.lhs) == i and len(rule.rhs) == i, rules_c04))
    # self explanatory
    results_ri_ci_inverted_conf_equal = {}
    for rule_1 in rules_r1_c1:
        tmp = rule_1
        for rule_2 in rules_r1_c1:
            if tmp.lhs == rule_2.rhs and tmp.rhs == rule_2.lhs:
                if not list(filter(lambda r: r[0] == tmp.rhs[0] and r[1] == tmp.lhs[0],
                                   results_ri_ci_inverted_conf_equal.keys())):
                    a = tmp.lhs[0]
                    b = tmp.rhs[0]
                    results_ri_ci_inverted_conf_equal[(a, b)] = (tmp.confidence, rule_2.confidence)

    results_rn_cn_inverted_conf_equal[i] = results_ri_ci_inverted_conf_equal

# More friendly visualisation
i_2 = pd.DataFrame.from_dict(results_rn_cn_inverted_conf_equal[2], orient='index')
i_3 = pd.DataFrame.from_dict(results_rn_cn_inverted_conf_equal[3], orient='index')
i_4 = pd.DataFrame.from_dict(results_rn_cn_inverted_conf_equal[4], orient='index')
i_5 = pd.DataFrame.from_dict(results_rn_cn_inverted_conf_equal[5], orient='index')
"""
Answer : Confidence proximity between itemsets pairs seems to be rather random
"""

# Question 6
item_sets_length = sum(map(len, map(lambda item: item[1], item_sets_c05.items())))
print(f"for a confidence of 0.5 and a minimum support of 0.1, we have {item_sets_length}, "
      f"we could sort them and shot the most frequent by decreasing support but I'm not even "
      f"sure that's the question (cf question 1)")


# Question 7
"""
mdr alz c'est bon
"""

# Question 8
# You need spmf.jar in your working directory
output_path = os.path.join(local_root, 'TP2/data/output.txt')
spmf_build = spmf.Spmf("AprioriClose", arguments=[0.3], input_filename=data_path,
                       spmf_bin_location_dir='C:/Users/Theo/PycharmProjects/AADA/TP2')

spmf_build.run()
