import pandas as pd
import dill
import numpy as np
from collections import defaultdict


##### process medications #####
# load med data
def med_process(med_file):
    """读取MIMIC原数据文件，保留pid、adm_id、data以及NDC，以DF类型返回"""
    # 读取药物文件，NDC（National Drug Code）以类别类型存储
    med_pd = pd.read_csv(med_file, dtype={'NDC':'category'})

    # drop不用的数据
    med_pd.drop(columns=['ROW_ID','DRUG_TYPE','DRUG_NAME_POE','DRUG_NAME_GENERIC',
                        'FORMULARY_DRUG_CD','PROD_STRENGTH','DOSE_VAL_RX',
                        'DOSE_UNIT_RX','FORM_VAL_DISP','FORM_UNIT_DISP', 'GSN', 'FORM_UNIT_DISP',
                        'ROUTE','ENDDATE','DRUG'], axis=1, inplace=True)
    med_pd.drop(index = med_pd[med_pd['NDC'] == '0'].index, axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd['ICUSTAY_ID'] = med_pd['ICUSTAY_ID'].astype('int64')
    med_pd['STARTDATE'] = pd.to_datetime(med_pd['STARTDATE'], format='%Y-%m-%d %H:%M:%S')    
    med_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'], inplace=True)
    med_pd = med_pd.reset_index(drop=True)  # 重置索引，同时drop原索引

    med_pd = med_pd.drop(columns=['ICUSTAY_ID'])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)

    return med_pd

# medication mapping
def ndc2atc4(med_pd):
    """将NDC映射到ACT4"""
    with open(ndc_rxnorm_file, 'r') as f:
        ndc2rxnorm = eval(f.read())
    # 根据ndc_rxnorm_file文件读取ndc到xnorm的映射（这个xnorm似乎等同于下面的RXCUI）
    med_pd['RXCUI'] = med_pd['NDC'].map(ndc2rxnorm)
    med_pd.dropna(inplace=True) # 实际上啥也没删掉

    rxnorm2atc = pd.read_csv(ndc2atc_file)
    rxnorm2atc = rxnorm2atc.drop(columns=['YEAR','MONTH','NDC'])    # NDC删了，直接从RXCUI映射到ATC
    # 根据RXCUI删除重复列
    rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)

    med_pd.drop(index = med_pd[med_pd['RXCUI'].isin([''])].index, axis=0, inplace=True)     # 删除特定的RXCUI
    
    med_pd['RXCUI'] = med_pd['RXCUI'].astype('int64')   
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(rxnorm2atc, on=['RXCUI'])     # 合并两个表
    med_pd.drop(columns=['NDC', 'RXCUI'], inplace=True) # 干掉NDC\RXCUI，只剩ATC4了
    med_pd = med_pd.rename(columns={'ATC4':'NDC'})      # 重新命名为NDC
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: x[:4])  # 只保留前四位
    med_pd = med_pd.drop_duplicates()    
    med_pd = med_pd.reset_index(drop=True)
    return med_pd

# visit >= 2
def process_visit_lg2(med_pd):
    """筛除admission次数小于两次的患者数据"""
    a = med_pd[['SUBJECT_ID', 'HADM_ID']].groupby(by='SUBJECT_ID')['HADM_ID'].unique().reset_index()
    a['HADM_ID_Len'] = a['HADM_ID'].map(lambda x:len(x))
    a = a[a['HADM_ID_Len'] > 1]
    return a 

# most common medications
def filter_300_most_med(med_pd):
    # 按照NDC出现的次数降序排列，取前300
    med_count = med_pd.groupby(by=['NDC']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
    med_pd = med_pd[med_pd['NDC'].isin(med_count.loc[:299, 'NDC'])]
    
    return med_pd.reset_index(drop=True)

##### process diagnosis #####
def diag_process(diag_file):
    diag_pd = pd.read_csv(diag_file)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=['SEQ_NUM','ROW_ID'],inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['SUBJECT_ID','HADM_ID'], inplace=True)
    diag_pd = diag_pd.reset_index(drop=True)

    def filter_2000_most_diag(diag_pd):
        diag_count = diag_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
        diag_pd = diag_pd[diag_pd['ICD9_CODE'].isin(diag_count.loc[:1999, 'ICD9_CODE'])]
        
        return diag_pd.reset_index(drop=True)

    diag_pd = filter_2000_most_diag(diag_pd)

    return diag_pd

##### process procedure #####
def procedure_process(procedure_file):
    pro_pd = pd.read_csv(procedure_file, dtype={'ICD9_CODE':'category'})
    pro_pd.drop(columns=['ROW_ID'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'], inplace=True)
    pro_pd.drop(columns=['SEQ_NUM'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)

    return pro_pd

def filter_1000_most_pro(pro_pd):
    pro_count = pro_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
    pro_pd = pro_pd[pro_pd['ICD9_CODE'].isin(pro_count.loc[:1000, 'ICD9_CODE'])]
    
    return pro_pd.reset_index(drop=True) 

###### combine three tables #####
def combine_process(med_pd, diag_pd, pro_pd):
    """药物、症状、proc的数据结合"""

    med_pd_key = med_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    diag_pd_key = diag_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    pro_pd_key = pro_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()

    combined_key = med_pd_key.merge(diag_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    combined_key = combined_key.merge(pro_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    diag_pd = diag_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    med_pd = med_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    pro_pd = pro_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    # flatten and merge
    diag_pd = diag_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index()  
    med_pd = med_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['NDC'].unique().reset_index()
    pro_pd = pro_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index().rename(columns={'ICD9_CODE':'PRO_CODE'})  
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: list(x))
    pro_pd['PRO_CODE'] = pro_pd['PRO_CODE'].map(lambda x: list(x))
    data = diag_pd.merge(med_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data = data.merge(pro_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    #     data['ICD9_CODE_Len'] = data['ICD9_CODE'].map(lambda x: len(x))
    data['NDC_Len'] = data['NDC'].map(lambda x: len(x))

    return data

def statistics(data):
    print('#patients ', data['SUBJECT_ID'].unique().shape)
    print('#clinical events ', len(data))
    
    diag = data['ICD9_CODE'].values
    med = data['NDC'].values
    pro = data['PRO_CODE'].values
    
    unique_diag = set([j for i in diag for j in list(i)])
    unique_med = set([j for i in med for j in list(i)])
    unique_pro = set([j for i in pro for j in list(i)])
    
    print('#diagnosis ', len(unique_diag))
    print('#med ', len(unique_med))
    print('#procedure', len(unique_pro))
    
    avg_diag, avg_med, avg_pro, max_diag, max_med, max_pro, cnt, max_visit, avg_visit = [0 for i in range(9)]

    for subject_id in data['SUBJECT_ID'].unique():
        item_data = data[data['SUBJECT_ID'] == subject_id]
        x, y, z = [], [], []
        visit_cnt = 0
        for index, row in item_data.iterrows():
            visit_cnt += 1
            cnt += 1
            x.extend(list(row['ICD9_CODE']))
            y.extend(list(row['NDC']))
            z.extend(list(row['PRO_CODE']))
        x, y, z = set(x), set(y), set(z)
        avg_diag += len(x)
        avg_med += len(y)
        avg_pro += len(z)
        avg_visit += visit_cnt
        if len(x) > max_diag:
            max_diag = len(x)
        if len(y) > max_med:
            max_med = len(y) 
        if len(z) > max_pro:
            max_pro = len(z)
        if visit_cnt > max_visit:
            max_visit = visit_cnt
        
    print('#avg of diagnoses ', avg_diag/ cnt)
    print('#avg of medicines ', avg_med/ cnt)
    print('#avg of procedures ', avg_pro/ cnt)
    print('#avg of vists ', avg_visit/ len(data['SUBJECT_ID'].unique()))
    
    print('#max of diagnoses ', max_diag)
    print('#max of medicines ', max_med)
    print('#max of procedures ', max_pro)
    print('#max of visit ', max_visit)

##### indexing file and final record
class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)
                
# create voc set
def create_str_token_mapping(df):
    diag_voc = Voc()
    med_voc = Voc()
    pro_voc = Voc()
    
    for index, row in df.iterrows():
        diag_voc.add_sentence(row['ICD9_CODE'])
        med_voc.add_sentence(row['NDC'])
        pro_voc.add_sentence(row['PRO_CODE'])
    
    dill.dump(obj={'diag_voc':diag_voc, 'med_voc':med_voc ,'pro_voc':pro_voc}, file=open('voc_final.pkl','wb'))
    return diag_voc, med_voc, pro_voc

# create final records
def create_patient_record(df, diag_voc, med_voc, pro_voc):
    """
    保存list类型的记录
    每一项代表一个患者，患者中有多个visit，每个visit包含三者数组，按顺序分别表示诊断、proc与药物
    存储的均为编号，可以通过voc_final.pkl来查看对应的具体word
    """
    records = [] # (patient, code_kind:3, codes)  code_kind:diag, proc, med
    for subject_id in df['SUBJECT_ID'].unique():
        item_df = df[df['SUBJECT_ID'] == subject_id]
        patient = []
        for index, row in item_df.iterrows():
            admission = []
            admission.append([diag_voc.word2idx[i] for i in row['ICD9_CODE']])
            admission.append([pro_voc.word2idx[i] for i in row['PRO_CODE']])
            admission.append([med_voc.word2idx[i] for i in row['NDC']])
            patient.append(admission)
        records.append(patient) 
    dill.dump(obj=records, file=open('records_final.pkl', 'wb'))
    return records
        


# get ddi matrix
def get_ddi_matrix(records, med_voc, ddi_file):

    TOPK = 40 # topk drug-drug interaction
    cid2atc_dic = defaultdict(set)
    med_voc_size = len(med_voc.idx2word)
    med_unique_word = [med_voc.idx2word[i] for i in range(med_voc_size)]    # 所有的药物的ATC4
    atc3_atc4_dic = defaultdict(set)
    for item in med_unique_word:
        atc3_atc4_dic[item[:4]].add(item)   # 
    
    with open(cid_atc, 'r') as f:
        for line in f:
            line_ls = line[:-1].split(',')
            cid = line_ls[0]
            atcs = line_ls[1:]
            for atc in atcs:
                if len(atc3_atc4_dic[atc[:4]]) != 0:
                    cid2atc_dic[cid].add(atc[:4])
            
    # 加载DDI数据
    ddi_df = pd.read_csv(ddi_file)
    # fliter sever side effect，也是采取topK的形式
    ddi_most_pd = ddi_df.groupby(by=['Polypharmacy Side Effect', 'Side Effect Name']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
    ddi_most_pd = ddi_most_pd.iloc[-TOPK:,:]
    # ddi_most_pd = pd.DataFrame(columns=['Side Effect Name'], data=['as','asd','as'])
    fliter_ddi_df = ddi_df.merge(ddi_most_pd[['Side Effect Name']], how='inner', on=['Side Effect Name'])
    ddi_df = fliter_ddi_df[['STITCH 1','STITCH 2']].drop_duplicates().reset_index(drop=True)


    # weighted ehr adj 
    ehr_adj = np.zeros((med_voc_size, med_voc_size))
    for patient in records:
        for adm in patient:
            med_set = adm[2]
            for i, med_i in enumerate(med_set):
                for j, med_j in enumerate(med_set):
                    if j<=i:
                        continue
                    # ehr_adj[med_i, med_j] = 1
                    # ehr_adj[med_j, med_i] = 1
                    ehr_adj[med_i, med_j] += 1
                    ehr_adj[med_j, med_i] += 1
    dill.dump(ehr_adj, open('ehr_adj_final.pkl', 'wb'))

    # ddi adj，DDI表是CID编码的，因此需要将CID映射到ACT编码，才能记录数据集中药物之间的冲突信息
    ddi_adj = np.zeros((med_voc_size,med_voc_size))
    for index, row in ddi_df.iterrows():
        # ddi
        cid1 = row['STITCH 1']
        cid2 = row['STITCH 2']
        
        # cid -> atc_level3
        for atc_i in cid2atc_dic[cid1]:
            for atc_j in cid2atc_dic[cid2]:
                
                # atc_level3 -> atc_level4
                for i in atc3_atc4_dic[atc_i]:
                    for j in atc3_atc4_dic[atc_j]:
                        if med_voc.word2idx[i] != med_voc.word2idx[j]:
                            ddi_adj[med_voc.word2idx[i], med_voc.word2idx[j]] = 1
                            ddi_adj[med_voc.word2idx[j], med_voc.word2idx[i]] = 1
    dill.dump(ddi_adj, open('ddi_A_final.pkl', 'wb')) 

    return ddi_adj

if __name__ == '__main__':
    # MIMIC数据文件，分别包括药物、诊断和proc
    med_file = '/data/mimic-iii/PRESCRIPTIONS.csv'
    diag_file = '/data/mimic-iii/DIAGNOSES_ICD.csv'
    procedure_file = '/data/mimic-iii/PROCEDURES_ICD.csv'

    # 药物信息
    med_structure_file = './idx2drug.pkl'   # 药物到分子式的映射

    # drug code mapping files
    ndc2atc_file = './ndc2atc_level4.csv'   # NDC code to ATC-4 code mapping file，用于读取xnorm到ATC
    cid_atc = './drug-atc.csv'              # drug（CID） to ATC code mapping file，用于处理DDI表
    ndc_rxnorm_file = './ndc2rxnorm_mapping.txt'    # NDC to xnorm mapping file

    # ddi information
    # data example
    # STITCH 1,STITCH 2,Polypharmacy Side Effect,Side Effect Name
    # CID000002173,CID000003345,C0151714,hypermagnesemia
    # CID000002173,CID000003345,C0035344,retinopathy of prematurity
    # CID000002173,CID000003345,C0004144,atelectasis
    # CID000002173,CID000003345,C0002063,alkalosis
    # CID000002173,CID000003345,C0004604,Back Ache
    # CID000002173,CID000003345,C0034063,lung edema
    ddi_file = '/data/drug-DDI.csv'

    # 处理MIMIC中的药物数据
    med_pd = med_process(med_file)
    med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)   # 注意这里仅仅是针对med表中出现了两次以上admission的patient
    med_pd = med_pd.merge(med_pd_lg2[['SUBJECT_ID']], on='SUBJECT_ID', how='inner').reset_index(drop=True) 

    med_pd = ndc2atc4(med_pd)
    NDCList = dill.load(open(med_structure_file, 'rb'))
    med_pd = med_pd[med_pd.NDC.isin(list(NDCList.keys()))]
    med_pd = filter_300_most_med(med_pd)

    print ('complete medication processing')

    # for diagnosis
    diag_pd = diag_process(diag_file)

    print ('complete diagnosis processing')

    # for procedure
    pro_pd = procedure_process(procedure_file)
    # pro_pd = filter_1000_most_pro(pro_pd)

    print ('complete procedure processing')

    # combine
    data = combine_process(med_pd, diag_pd, pro_pd)
    statistics(data)
    data.to_pickle('data_final.pkl')

    print ('complete combining')


    # ddi_matrix
    diag_voc, med_voc, pro_voc = create_str_token_mapping(data)
    records = create_patient_record(data, diag_voc, med_voc, pro_voc)   # diag,proc,medication按顺序存储
    ddi_adj = get_ddi_matrix(records, med_voc, ddi_file)
