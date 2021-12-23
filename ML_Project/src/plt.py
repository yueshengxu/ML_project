import matplotlib.pyplot as plt
import pickle
import numpy as np

l = pickle.load( open( "result.pkl", "rb" ) )
l_no_fair = pickle.load( open( "result_no_fair.pkl", "rb" ) )
l_xndcg = pickle.load( open( "result_xndcg.pkl", "rb" ) )
l_xndcg10 = pickle.load( open( "result_xndcg10.pkl", "rb" ) )
l_ratio1 = pickle.load( open( "result_ratio0.1.pkl", "rb" ) )

epochs = np.arange(len(l))
test_no_fair = pickle.load( open( "test_no_fair.pkl", "rb" ) )
ndcg_no_tr = []
ndcg0_no_tr = []
ndcg1_no_tr = []
for i in test_no_fair:
    ndcg_no_tr.append(i[0])
    ndcg0_no_tr.append(i[1])
    ndcg1_no_tr.append(i[2])
ndcg_no_tr = np.array(ndcg_no_tr)
ndcg0_no_tr = np.array(ndcg0_no_tr)
ndcg1_no_tr = np.array(ndcg1_no_tr)

epochs = np.arange(len(ndcg_no_tr))
epochs = epochs*4+1
plt.plot(epochs,ndcg_no_tr)