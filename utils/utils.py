def holm_bonferroni(s):
	s_sorted = s.sort_values('p-value')
	corrected = s_sorted['p-value'].copy()
	m = len(s_sorted)
	bon_corrected = corrected*len(corrected)
	for i in range(len(s_sorted)):
		p = 0
		for j in range(i+1):
			temp = (m - j +1)*s_sorted.iloc[j,0]
			if temp > p:
				p = temp
		corrected.iloc[i] = np.min([p,1])
		
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from scipy.optimize import linear_sum_assignment

def multivariate_2samp(A, B, iters = 1000, random_state = 0):
    merged = pd.concat([pd.DataFrame(A), pd.DataFrame(B)]).values
    
    dist_matrix = cdist(A,B)
    A_idx, B_idx = linear_sum_assignment(dist_matrix)
    statistic = dist_matrix[A_idx, B_idx].sum()
    
    pvalue = 0
    for i in range(iters):
        A_boot, B_boot = train_test_split(merged, train_size=len(A), random_state=random_state+i)
        boot_dist_matrix = cdist(A_boot, B_boot)
        A_boot_idx, B_boot_idx = linear_sum_assignment(boot_dist_matrix)
        boot_statistic = boot_dist_matrix[A_boot_idx, B_boot_idx].sum()
        if statistic <= boot_statistic:
            pvalue += 1
    return pvalue/iters