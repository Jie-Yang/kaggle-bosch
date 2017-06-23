
import time
from utils import load_training_subset_1110,read_variable,save_variable
import numpy as np
from sklearn.metrics import matthews_corrcoef


#%%
val_X,val_Y = load_training_subset_1110(range(1000,1010,1))


tr_X_1s = read_variable('model_stats/tr_pip_data_1s_1110.pkl')


#%%
'''
Model: LogisticRegression
'''
import os
from sklearn.linear_model import LogisticRegression



len_1s = tr_X_1s.shape[0]
# can not afford smaller tol which will take too long to finish
tol = 1e-3
for set_id in range(0,166,1):

    file_path = '7/logic_'+str(set_id)+'.pkl'
    if os.path.exists(file_path):
                print('already exist.',file_path)
    else:
    
        # 6 chunks give about the same 0s (tiny amount of 1s is ignored) as the 1s
        chunk_range = range(set_id,1000,166)
        t_X,t_Y = load_training_subset_1110(chunk_range)
        tr_X = np.concatenate([t_X,tr_X_1s])
        tr_Y = np.concatenate([t_Y,np.ones(len_1s)])
        
        '''
        based on experiment, smaller tol give better fit on training dataset
         tol : float, default: 1e-4
        Tolerance for stopping criteria.
        '''

        best_model = LogisticRegression(penalty='l1',tol=tol,n_jobs=3,
                                        solver='liblinear')	
        t0 = time.time()
        best_model =best_model.fit(tr_X,tr_Y)
        tr_Y_pred = best_model.predict(tr_X)
        best_tr_mcc = matthews_corrcoef(tr_Y , tr_Y_pred)
        print(set_id,'- i','(',tol,')','logic:',',tr:',best_tr_mcc,end='')
        
        print(',val:',end='')
        val_Y_pred = best_model.predict(val_X)
        print(matthews_corrcoef(val_Y, val_Y_pred),end='')
        print(',cost:',int(time.time()-t0),'sec')
        #print('val 1s:real',sum(val_Y),',pred',sum(val_Y_pred))

        save_variable(best_model,file_path)
#%%
'''
111 - i ( 0.001 ) logic: ,tr: 0.146291817756,val:0.0161041850828,cost: 4 sec
loading & preprocessing (hd, imputer, kbest) data... range(112, 1000, 166)
112 - i ( 0.001 ) logic: ,tr: 0.164291317745,val:0.0275685119895,cost: 72 sec
loading & preprocessing (hd, imputer, kbest) data... range(113, 1000, 166)
113 - i ( 0.001 ) logic: ,tr: 0.154070756427,val:0.0208983531819,cost: 5 sec
loading & preprocessing (hd, imputer, kbest) data... range(114, 1000, 166)
114 - i ( 0.001 ) logic: ,tr: 0.139766502306,val:0.0148543144396,cost: 3 sec
loading & preprocessing (hd, imputer, kbest) data... range(115, 1000, 166)
115 - i ( 0.001 ) logic: ,tr: 0.161565076274,val:0.0285054621172,cost: 3 sec
loading & preprocessing (hd, imputer, kbest) data... range(116, 1000, 166)
116 - i ( 0.001 ) logic: ,tr: 0.153309748104,val:0.0183308995112,cost: 22 sec
loading & preprocessing (hd, imputer, kbest) data... range(117, 1000, 166)
117 - i ( 0.001 ) logic: ,tr: 0.169121656326,val:0.0207511049556,cost: 3 sec
loading & preprocessing (hd, imputer, kbest) data... range(118, 1000, 166)
118 - i ( 0.001 ) logic: ,tr: 0.163186199076,val:0.0224774080407,cost: 24 sec
loading & preprocessing (hd, imputer, kbest) data... range(119, 1000, 166)
119 - i ( 0.001 ) logic: ,tr: 0.15840378733,val:0.0107273353766,cost: 18 sec
loading & preprocessing (hd, imputer, kbest) data... range(120, 1000, 166)
120 - i ( 0.001 ) logic: ,tr: 0.146602510808,val:0.0148399887173,cost: 12 sec
loading & preprocessing (hd, imputer, kbest) data... range(121, 1000, 166)
121 - i ( 0.001 ) logic: ,tr: 0.167611937635,val:0.0139054972223,cost: 15 sec
loading & preprocessing (hd, imputer, kbest) data... range(122, 1000, 166)
122 - i ( 0.001 ) logic: ,tr: 0.14822241953,val:0.0134150713262,cost: 12 sec
loading & preprocessing (hd, imputer, kbest) data... range(123, 1000, 166)
123 - i ( 0.001 ) logic: ,tr: 0.142651155839,val:0.0046518503457,cost: 4 sec
loading & preprocessing (hd, imputer, kbest) data... range(124, 1000, 166)
124 - i ( 0.001 ) logic: ,tr: 0.155069691762,val:0.0252375520174,cost: 3 sec
loading & preprocessing (hd, imputer, kbest) data... range(125, 1000, 166)
125 - i ( 0.001 ) logic: ,tr: 0.156755078354,val:0.00381296120692,cost: 10 sec
loading & preprocessing (hd, imputer, kbest) data... range(126, 1000, 166)
126 - i ( 0.001 ) logic: ,tr: 0.15503546915,val:0.0271002940945,cost: 13 sec
loading & preprocessing (hd, imputer, kbest) data... range(127, 1000, 166)
127 - i ( 0.001 ) logic: ,tr: 0.159619337306,val:0.0150750826253,cost: 4 sec
loading & preprocessing (hd, imputer, kbest) data... range(128, 1000, 166)
128 - i ( 0.001 ) logic: ,tr: 0.159667130175,val:0.0215444080081,cost: 3 sec
loading & preprocessing (hd, imputer, kbest) data... range(129, 1000, 166)
129 - i ( 0.001 ) logic: ,tr: 0.16949463931,val:0.00657809653853,cost: 12 sec
loading & preprocessing (hd, imputer, kbest) data... range(130, 1000, 166)
130 - i ( 0.001 ) logic: ,tr: 0.160612116769,val:0.00727672210368,cost: 9 sec
loading & preprocessing (hd, imputer, kbest) data... range(131, 1000, 166)
131 - i ( 0.001 ) logic: ,tr: 0.170164191374,val:0.0232736860593,cost: 3 sec
loading & preprocessing (hd, imputer, kbest) data... range(132, 1000, 166)
132 - i ( 0.001 ) logic: ,tr: 0.165554791741,val:0.00372337259139,cost: 8 sec
loading & preprocessing (hd, imputer, kbest) data... range(133, 1000, 166)
133 - i ( 0.001 ) logic: ,tr: 0.160006845204,val:0.0147848771498,cost: 3 sec
loading & preprocessing (hd, imputer, kbest) data... range(134, 1000, 166)
134 - i ( 0.001 ) logic: ,tr: 0.16060231861,val:0.00349240068619,cost: 11 sec
loading & preprocessing (hd, imputer, kbest) data... range(135, 1000, 166)
135 - i ( 0.001 ) logic: ,tr: 0.178362020591,val:0.0182267228354,cost: 2 sec
loading & preprocessing (hd, imputer, kbest) data... range(136, 1000, 166)
136 - i ( 0.001 ) logic: ,tr: 0.159119665674,val:0.0218099719156,cost: 3 sec
loading & preprocessing (hd, imputer, kbest) data... range(137, 1000, 166)
137 - i ( 0.001 ) logic: ,tr: 0.163665828712,val:0.0121056718232,cost: 7 sec
loading & preprocessing (hd, imputer, kbest) data... range(138, 1000, 166)
138 - i ( 0.001 ) logic: ,tr: 0.170368145238,val:0.0233971675155,cost: 3 sec
loading & preprocessing (hd, imputer, kbest) data... range(139, 1000, 166)
139 - i ( 0.001 ) logic: ,tr: 0.167535105715,val:0.00513602393695,cost: 6 sec
loading & preprocessing (hd, imputer, kbest) data... range(140, 1000, 166)
140 - i ( 0.001 ) logic: ,tr: 0.153959984389,val:0.0174421623067,cost: 7 sec
loading & preprocessing (hd, imputer, kbest) data... range(141, 1000, 166)
141 - i ( 0.001 ) logic: ,tr: 0.174603007568,val:0.0254865374959,cost: 9 sec
loading & preprocessing (hd, imputer, kbest) data... range(142, 1000, 166)
142 - i ( 0.001 ) logic: ,tr: 0.168706391685,val:0.00963064554515,cost: 6 sec
loading & preprocessing (hd, imputer, kbest) data... range(143, 1000, 166)
143 - i ( 0.001 ) logic: ,tr: 0.176641273917,val:0.00613980264826,cost: 6 sec
loading & preprocessing (hd, imputer, kbest) data... range(144, 1000, 166)
144 - i ( 0.001 ) logic: ,tr: 0.167097439824,val:0.0197249497831,cost: 8 sec
loading & preprocessing (hd, imputer, kbest) data... range(145, 1000, 166)
145 - i ( 0.001 ) logic: ,tr: 0.166016239354,val:0.0151927842614,cost: 2 sec
loading & preprocessing (hd, imputer, kbest) data... range(146, 1000, 166)
146 - i ( 0.001 ) logic: ,tr: 0.176232560673,val:-0.00201441634957,cost: 2 sec
loading & preprocessing (hd, imputer, kbest) data... range(147, 1000, 166)
147 - i ( 0.001 ) logic: ,tr: 0.169621728537,val:0.00130797283779,cost: 6 sec
loading & preprocessing (hd, imputer, kbest) data... range(148, 1000, 166)
148 - i ( 0.001 ) logic: ,tr: 0.169657150528,val:0.0190255562969,cost: 5 sec
loading & preprocessing (hd, imputer, kbest) data... range(149, 1000, 166)
149 - i ( 0.001 ) logic: ,tr: 0.171754653401,val:0.00426973908697,cost: 7 sec
loading & preprocessing (hd, imputer, kbest) data... range(150, 1000, 166)
150 - i ( 0.001 ) logic: ,tr: 0.167041644854,val:0.00906970489795,cost: 6 sec
loading & preprocessing (hd, imputer, kbest) data... range(151, 1000, 166)
151 - i ( 0.001 ) logic: ,tr: 0.175875306674,val:-0.00437080429168,cost: 2 sec
loading & preprocessing (hd, imputer, kbest) data... range(152, 1000, 166)
152 - i ( 0.001 ) logic: ,tr: 0.165927157283,val:-0.00511160238819,cost: 3 sec
loading & preprocessing (hd, imputer, kbest) data... range(153, 1000, 166)
153 - i ( 0.001 ) logic: ,tr: 0.18474239264,val:-0.0120107830295,cost: 6 sec
loading & preprocessing (hd, imputer, kbest) data... range(154, 1000, 166)
154 - i ( 0.001 ) logic: ,tr: 0.161099298419,val:0.00360831018508,cost: 7 sec
loading & preprocessing (hd, imputer, kbest) data... range(155, 1000, 166)
155 - i ( 0.001 ) logic: ,tr: 0.170390005843,val:0.00746149906941,cost: 7 sec
loading & preprocessing (hd, imputer, kbest) data... range(156, 1000, 166)
156 - i ( 0.001 ) logic: ,tr: 0.163701291023,val:0.0105248455278,cost: 5 sec
loading & preprocessing (hd, imputer, kbest) data... range(157, 1000, 166)
157 - i ( 0.001 ) logic: ,tr: 0.171218762764,val:-0.000816155118347,cost: 2 sec
loading & preprocessing (hd, imputer, kbest) data... range(158, 1000, 166)
158 - i ( 0.001 ) logic: ,tr: 0.182499961359,val:-0.0028036905887,cost: 6 sec
loading & preprocessing (hd, imputer, kbest) data... range(159, 1000, 166)
159 - i ( 0.001 ) logic: ,tr: 0.171584985056,val:-3.99143079002e-05,cost: 6 sec
loading & preprocessing (hd, imputer, kbest) data... range(160, 1000, 166)
160 - i ( 0.001 ) logic: ,tr: 0.180418754477,val:0.00441477528845,cost: 6 sec
loading & preprocessing (hd, imputer, kbest) data... range(161, 1000, 166)
161 - i ( 0.001 ) logic: ,tr: 0.170628209931,val:-0.00232368394411,cost: 2 sec
loading & preprocessing (hd, imputer, kbest) data... range(162, 1000, 166)
162 - i ( 0.001 ) logic: ,tr: 0.176823723506,val:0.0107198457645,cost: 2 sec
loading & preprocessing (hd, imputer, kbest) data... range(163, 1000, 166)
163 - i ( 0.001 ) logic: ,tr: 0.178297674021,val:0.00253843627056,cost: 4 sec
loading & preprocessing (hd, imputer, kbest) data... range(164, 1000, 166)
164 - i ( 0.001 ) logic: ,tr: 0.176862775374,val:0.0116789776541,cost: 5 sec
loading & preprocessing (hd, imputer, kbest) data... range(165, 1000, 166)
165 - i ( 0.001 ) logic: ,tr: 0.17525818358,val:0.0052852294804,cost: 6 sec
'''