import pandas as pd
import numpy as np
def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0
    return score / min(len(actual), k)


def mean_reciprocal_rank(actual,predicted):
    predicted = np.array(predicted)
    rank = np.where( predicted== np.array(actual[0])) 
    result = (rank[0]+1)/len(predicted)
    return result

 
    
    
def frequency_mean_reciprocal_rank(predicted,actual):
    predicted =  np.array(predicted)
    predicted = predicted[~(predicted == 'nan')]
    actual =np.array(actual)
    actual = actual[~(actual == 'nan')]
    total_frequency_rank = 0
    for correct_response in actual:
        print(correct_response)
        print(predicted)
        rank = np.where( predicted== correct_response) 
        result = (rank[0]+1)/len(predicted)
        print(result)
        if len(result) == 0:
            return 0.0
        total_frequency_rank += result[0]
    return total_frequency_rank



original_df  = pd.read_excel('output.xlsx')
original_df.drop(['Unnamed: 0'],axis= 1,inplace = True)
original_dict = original_df.set_index('0_x').T.to_dict('list')

tfidf_results_df = pd.read_excel('TFIDF_results.xlsx')
tfidf_results_dict = tfidf_results_df.set_index('Unnamed: 0').T.to_dict('list')


score_list = {}
for query in tfidf_results_dict.keys():
    score_list[query] = mean_reciprocal_rank(tfidf_results_dict[query],original_dict[query])
    
score_list = {}
for query in tfidf_results_dict.keys():
    score_list[query] = frequency_mean_reciprocal_rank(tfidf_results_dict[query],original_dict[query])