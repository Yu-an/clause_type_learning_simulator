import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics.cluster import rand_score
import scipy.special as sc


def xtab(*cols):
    if not all(len(col) == len(cols[0]) for col in cols[1:]):
        raise ValueError("all arguments must be same size")

    if len(cols) != 2:
        raise TypeError("xtab() requires 2 numpy arrays")

    fnx1 = lambda q: len(q.squeeze().shape)
    if not all([fnx1(col) == 1 for col in cols]):
        raise ValueError("all input arrays must be 1D")
    wt =1

    uniq_vals_all_cols, idx = zip( *(np.unique(col, return_inverse=True) for col in cols) )
    shape_xt = [uniq_vals_col.size for uniq_vals_col in uniq_vals_all_cols]
    xt = np.zeros(shape_xt)
    dtype_xt = 'float'
    np.add.at(xt, idx, wt)
    return  xt


def PivotTab(df, var1, var2):
    #var1 will be the col, var2 is the row
    df_target=df.groupby([var1,var2])[var2].agg(["count"]).reset_index()
    a = df_target.groupby([var1])["count"].sum().to_dict()
    x = []
    for i in df_target.index:
        n = a[df_target[var1][i]]
        x.append(df_target['count'][i]/n)
    df_target["freq"]=x
    df_targetp=pd.pivot_table(df_target, values = "freq", index=[var1],
                        columns=[var2], fill_value = 0, aggfunc=np.sum)
    return df_targetp

def lnB(alpha):
    nom = np.sum(sc.gammaln(alpha)) - sc.gammaln(np.sum(alpha))
    return nom

def LogJointProb(a,c,S):
    alpha = np.ones(len(np.unique(a)))
    beta_a = np.ones((len(np.unique(a)),len(np.unique(c))))
    gamma_0 = np.ones((len(np.unique(c)),2))
    a_values = np.unique(a)
    c_values = np.unique(c)
    num_f = len(S)
    num_n = len(a)
    n_a =np.unique(a,return_counts = True)[1]
    n_c_a = xtab(a,c)
    n_S_c = np.array([xtab(c,s) for s in S])    
    p_a_alpha = lnB(alpha + n_a) -lnB(alpha) 
    p_c_beta = np.sum(np.array([(lnB(beta_a[k]+n_c_a[k]) - lnB(beta_a[k]))for k in a_values]))
    P_S_set = []
    for f in range(num_f):
        p_S_gamma = np.sum(np.array([(lnB(gamma_0[m]+n_S_c[f][m]) - lnB(gamma_0[m]))for m in c_values]))
        P_S_set.append(p_S_gamma)
    P_S = np.sum(np.array(P_S_set))
    return(p_a_alpha+p_c_beta+ P_S)

def LogJointProb_base(c,S):
    beta = np.array([1,1,1])
    gamma_0 = np.array([
        [1,1],
        [1,1],
        [1,1]
    ])
    c_values = np.unique(c)
    num_f = len(S)
    num_n = len(c)
    n_c =np.unique(c,return_counts = True)[1]
    n_S_c = np.array([xtab(c,s) for s in S])    
    p_c_beta = lnB(beta + n_c) -lnB(beta) 
    P_S_set = []
    for f in range(num_f):
        p_S_gamma = np.sum(np.array([(lnB(gamma_0[m]+n_S_c[f][m]) - lnB(gamma_0[m]))for m in c_values]))
        P_S_set.append(p_S_gamma)
    P_S = np.sum(np.array(P_S_set))
    return(p_c_beta+ P_S)

def syn_cluster(df, cluster, color0,color1):
    df_part = df[df["C_sampled"]==cluster].reset_index()
    S = ["Subj","Aux", "AuxInvert", "InitFunction" ,"PreVFunction", "PostVFunction"]
    syn = []
    for s in S:
        N_c_s=pd.pivot_table(df_part, values = "n", index=['C_sampled'],
                        columns=[s], aggfunc="sum").fillna(0).to_numpy()
        syn.append(N_c_s)
    data = pd.DataFrame({#"features": ["Subj","Obj","Aux", "AuxInvert","Verb", "InitFunction","PreVFunction", "PostVFunction" ],
            "wo": [syn[i][0][0]for i in range(len(S))],
             "w":[len(df_part)-syn[i][0][0]for i in range(len(S))]
             }, index = S)
    index = data.index
    column0 = data["wo"]
    column1 = data["w"]
    title0 = '-'
    title1 = '+'
    fig, axes = plt.subplots(figsize=(2,3), ncols=2, sharey=True)
    fig.tight_layout()

    axes[0].barh(index, column0, align='center', color=color0, zorder=10)
    axes[0].set_title(title0, fontsize=12, pad=15, color=color0)
    axes[1].barh(index, column1, align='center', color=color1, zorder=10)
    axes[1].set_title(title1, fontsize=12, pad=15, color=color1)
    axes[0].invert_xaxis()    
    plt.gca().invert_yaxis()
    axes[0].set(yticks=data.index, yticklabels=data.index)
    axes[0].yaxis.tick_left()
    axes[0].set_xticks([ 800, 1600])
    axes[1].set_xticks([800,1600])
    for label in (axes[0].get_xticklabels() + axes[0].get_yticklabels()):
        label.set(fontsize=10, color="black")
    for label in (axes[1].get_xticklabels() + axes[1].get_yticklabels()):
        label.set(fontsize=10, color="black")
    plt.subplots_adjust(wspace=0, top=0.85, bottom=0.1, left=0.18, right=0.95)

    
def GraphClusterSyn(num, datas,values, colors):
    fig = plt.figure(figsize=(15, 8))
    outer = gridspec.GridSpec(1, num, wspace=0.1)
    titles = ["--","+"]
    axes = []
    for i in range(num):
        inner = gridspec.GridSpecFromSubplotSpec(1, 2,
                        subplot_spec=outer[i])

        for j in range(2):
                column = datas[i][j]
                index = datas[i].index
                ax = plt.Subplot(fig, inner[j])
                fig.add_subplot(ax)
                ax.barh(index, column, align='center', color=colors[i][j], zorder=1)
                ax.set_title(titles[j], fontsize=20, pad=15, color=colors[i][1])
                ax.set_xticks([400, 1000, 1600])
                ax.set_yticks([])
                plt.gca().invert_yaxis()
                for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                    label.set(fontsize=12, color="black")
                axes.append(ax)

        plt.text(-800, -1.8, "Cluster "+str(values[i]), color ="black", fontsize=20)
        plt.subplots_adjust(wspace=0, top=0.85, bottom=0.1, left=0.1, right=0.95)
    for x in range(num*2):
        if x%2 ==0:
            axes[x].invert_xaxis()
    axes[0].set(yticks=datas[0].index, yticklabels=datas[0].index)
    axes[0].yaxis.tick_left()
    # axes[5].yaxis.tick_right()
    # axes[5].set(yticks=data1.index, yticklabels=data1.index)
    for label in (axes[0].get_yticklabels()):
                    label.set(fontsize=20, color="black")    
def SynfeaturesData(df, value,S):
    df_part = df[df["C_sampled"]==value].reset_index()
    syn = []
    for s in S:
        feat = df_part[s].to_numpy()
        values, counts = np.unique(feat, return_counts =True)
        if values.shape[0] ==1:
            if values[0] ==1:
                counts = np.append(0,counts)
            if values[0] ==0:
                counts = np.append(counts,0) 
        syn.append(counts)
    data = pd.DataFrame.from_dict(syn)
    data.index = S
    data = data.rename(index = {'Subj':"Subject", 'Obj':"Object", 'Aux':"Auxiliary", 'AuxInvert':"Subj-aux Inversion", 'InitFunction':"Initial UFI", 'PreVFunction':"Pre-verbal UFI", 'PostVFunction':"Post-verbal UFI"})
    return data

    
class results:
    def __init__(self,lang, datadir,iter_num):
        self.lang = lang
        self.datadir = datadir
        self.iter_num = iter_num
        with open(self.datadir+self.iter_num, "rb") as f:
            self.c_sampled = np.load(f,allow_pickle=True)
            
        with open(self.datadir+"input_data/df_true","rb") as f:
            self.df = pickle.load(f)

        with open(self.datadir+'input_data/training_data', 'rb') as f:
            self.a = np.load(f,allow_pickle=True)
            self.c_true = np.load(f,allow_pickle=True)
        
        if "target" in self.iter_num:
            x = self.datadir+self.iter_num
            x_list = x.split("/")[:-2]
            self.simdir = "/".join(x_list)
            with open(self.simdir+"/a_sim","rb") as f:
                self.a_sim = np.load(f,allow_pickle=True)
            
        
        self.df["C_sampled"] = self.c_sampled
#         self.df["posterior"] = [[]]*len(self.df)
#         self.df["posterior"] = list(self.posterior)
#         self.df["likelihood"] = [[]]*len(self.df)
#         self.df["likelihood"] = list(self.likelihood)        
        self.df["n"] =1
        
        
        self.heat()
        self.purity()
        if self.lang == "man":
            self.mansyn()
        if self.lang == "eng":
            self.engsyn()
        self.rand_score()
        
    def heat(self):
        self.heat_proportion = PivotTab(self.df,"C_sampled", "ClauseType")
        self.heat_counts = pd.pivot_table(self.df, values = "n", index = ["C_sampled"], columns = ["ClauseType"], aggfunc = "sum")#.to_numpy()
        self.heat_counts = self.heat_counts.fillna(0)
        self.heat_reversed = PivotTab(self.df,"ClauseType","C_sampled") 
        return self.heat_proportion, self.heat_counts,  self.heat_reversed
    def purity(self):
        self.heat_counts_np = self.heat_counts.to_numpy()
        self.purity = np.sum([np.max(self.heat_counts_np[x])for x in range(3)])/np.sum(self.heat_counts_np) 
        return self.purity
    def rand_score(self):
        self.true_c = self.df["Ctrue"].to_numpy()
        self.clause_rand = rand_score(self.c_sampled, self.true_c)
        return self.clause_rand
    def engsyn(self):
        self.syn = []
        for s in ["Subj","Obj","Aux", "AuxInvert","Verb", "InitFunction","PreVFunction", "PostVFunction" ]:
            self.syn.append(PivotTab(self.df, "C_sampled",s))
        return self.syn
    def mansyn(self):
        self.syn = []
        for s in ["Subject","Object","Aux", "AuxInvert","Verb", "InitFunction","PreVFunction", "PostVFunction","SFP" ]:
            self.syn.append(PivotTab(self.df, "C_sampled",s))
        return self.syn
    def heatmap(self):
        self.heat_proportion = self.heat_proportion[['Declarative', 'Interrogative', 'Imperative' ]].copy()
        sns.heatmap(self.heat_proportion, annot = True, cmap="YlGnBu")
        plt.ylabel("Inferred clause category")
        plt.xlabel("Actual clause type labels")
    def heatmaprev(self):
        self.df_part = self.heat_reversed.reset_index()
        self.df_part = self.df_part[self.df_part["ClauseType"]!="Amb"]
        self.df_part = self.df_part[self.df_part["ClauseType"]!="Other"]
        self.heatrev = pd.pivot_table(self.df_part, index=["ClauseType"])        
        self.heatrev = self.heatrev.loc[['Declarative', 'Interrogative', 'Imperative']]
        sns.heatmap(self.heatrev, cmap="YlGnBu", annot=True)
        plt.xlabel("Inferred clause category")
        plt.ylabel("Actual clause type labels")  
       